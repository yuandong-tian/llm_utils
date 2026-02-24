"""OpenAI-compatible load balancing router.

Routes requests to multiple upstream LLM servers using least-busy selection.
Supports port range expansion (e.g., http://worker:30000-30007) and automatic
health probing on startup.

Usage:
    python llm_router.py --upstreams "http://server1:8000,http://server2:8000"
    python llm_router.py --upstreams "http://worker:30000-30007"
"""

import argparse
import asyncio
import json
import os
import re
import threading
from typing import Dict, List, Optional, Set

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse

from job_utils import get_jobs, parse_log_info


def _expand_api_base_range(api_base: str) -> List[str]:
    match = re.match(r"^(https?://[^:/]+:\d+)-(\d+)$", api_base.strip())
    if not match:
        return [api_base.strip()]

    start_url, end_port_str = match.groups()
    host_part, start_port_str = start_url.rsplit(":", 1)
    start_port = int(start_port_str)
    end_port = int(end_port_str)
    if end_port < start_port:
        raise ValueError("Invalid base URL range: end port < start port.")

    return [f"{host_part}:{port}" for port in range(start_port, end_port + 1)]


def _normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _parse_upstreams(value: Optional[str]) -> List[str]:
    if not value:
        raise ValueError("Missing upstream base URLs.")

    parts = [part.strip() for part in value.split(",") if part.strip()]
    expanded: List[str] = []
    for part in parts:
        expanded.extend(_expand_api_base_range(part))

    normalized = [_normalize_base_url(part) for part in expanded if part]
    if not normalized:
        raise ValueError("No valid upstream base URLs parsed.")
    return normalized


class _RouterSnapshot:
    """Immutable snapshot of router state, captured at acquire time.

    In-flight requests hold a reference to their snapshot so hot-reloads
    don't corrupt them.
    """
    __slots__ = ("upstreams", "in_flight", "model_map")

    def __init__(self, upstreams: List[str], model_map: Dict[str, List[int]]):
        self.upstreams = upstreams
        self.in_flight = [0] * len(upstreams)
        self.model_map = model_map


class LeastBusyRouter:
    """Thread-safe router with model-aware routing and safe hot-reload.

    Routes to the backend with the fewest in-flight requests.
    Uses a snapshot pattern so hot-reloads don't corrupt in-flight requests.
    """

    def __init__(self, upstreams: List[str], model_map: Optional[Dict[str, List[int]]] = None):
        self._lock = threading.Lock()
        self._snap = _RouterSnapshot(upstreams, model_map or {})

    def acquire_backend(self, exclude: Optional[set] = None,
                        model: Optional[str] = None) -> tuple:
        """Acquire the least-busy backend. Returns (handle, backend_url)."""
        with self._lock:
            snap = self._snap
            allowed = None
            if model and snap.model_map:
                allowed = set(snap.model_map.get(model, []))
                if not allowed:
                    allowed = None

            candidates = [
                (idx, count)
                for idx, count in enumerate(snap.in_flight)
                if (not exclude or idx not in exclude)
                and (allowed is None or idx in allowed)
            ]
            if not candidates:
                raise RuntimeError(f"No available upstreams for model '{model}'.")

            idx, _ = min(candidates, key=lambda item: item[1])
            snap.in_flight[idx] += 1
            return (snap, idx), snap.upstreams[idx]

    def release_backend(self, handle) -> None:
        """Release a backend acquired via acquire_backend()."""
        snap, idx = handle
        with self._lock:
            if idx < len(snap.in_flight):
                snap.in_flight[idx] = max(0, snap.in_flight[idx] - 1)

    def reload(self, new_upstreams: List[str],
               new_model_map: Optional[Dict[str, List[int]]] = None) -> None:
        """Hot-reload backends. In-flight requests on old snapshots are unaffected."""
        with self._lock:
            self._snap = _RouterSnapshot(new_upstreams, new_model_map or {})

    def backends_for_model(self, model: str) -> List[int]:
        """Return upstream indices that serve a given model."""
        snap = self._snap
        if snap.model_map:
            return snap.model_map.get(model, list(range(len(snap.upstreams))))
        return list(range(len(snap.upstreams)))

    @property
    def upstreams(self) -> List[str]:
        return self._snap.upstreams

    @property
    def model_map(self) -> Dict[str, List[int]]:
        return self._snap.model_map

    @property
    def all_models(self) -> List[str]:
        return sorted(self._snap.model_map.keys())

    @property
    def _in_flight(self):
        return self._snap.in_flight


def _probe_upstreams(upstreams: List[str]):
    """Probe upstreams, discover models. Returns (healthy_upstreams, model_map)."""
    healthy = []
    # model_name -> list of indices into the healthy list
    model_map: Dict[str, List[int]] = {}

    with httpx.Client(timeout=5.0) as client:
        for upstream in upstreams:
            try:
                response = client.get(f"{upstream}/models")
                if response.status_code < 500:
                    idx = len(healthy)
                    healthy.append(upstream)
                    # Parse models from response
                    models = []
                    try:
                        data = response.json()
                        for m in data.get("data", []):
                            mid = m.get("id", "")
                            if mid:
                                models.append(mid)
                    except Exception:
                        pass
                    if models:
                        print(f"[router] Upstream OK: {upstream} -> models: {models}")
                        for mid in models:
                            model_map.setdefault(mid, []).append(idx)
                    else:
                        print(f"[router] Upstream OK: {upstream} (no models discovered)")
                else:
                    print(f"[router] Upstream error {response.status_code}: {upstream}")
            except httpx.RequestError as exc:
                print(f"[router] Upstream unreachable: {upstream} ({exc})")

    return healthy, model_map


def _discover_from_squeue(log_dir: Optional[str] = None) -> tuple:
    """Auto-discover upstream URLs from running squeue jobs.

    Returns (healthy_upstreams, model_map) or ([], {}) if none found.
    """
    urls = []
    for job in get_jobs():
        if job.get("state") != "RUNNING":
            continue
        info = parse_log_info(job, log_dir=log_dir)
        port = info.get("port", "?")
        if port != "?" and info.get("status") in ("Ready", "Serving"):
            urls.append(f"http://{job['node']}:{port}/v1")
    return _probe_upstreams(urls) if urls else ([], {})


def _filter_by_models(
    healthy: List[str],
    model_map: Dict[str, List[int]],
    upstream_models: Optional[Set[str]],
) -> tuple:
    """Filter backends and model_map to only include covered models.

    Returns (filtered_upstreams, filtered_model_map) with re-indexed indices.
    """
    if not upstream_models or not model_map:
        return healthy, model_map

    # Find indices of backends that serve at least one covered model
    keep_indices = set()
    for mid, idxs in model_map.items():
        if mid in upstream_models:
            keep_indices.update(idxs)

    if not keep_indices:
        return [], {}

    # Build new list with re-indexed mapping
    old_to_new = {}
    new_upstreams = []
    for old_idx in sorted(keep_indices):
        old_to_new[old_idx] = len(new_upstreams)
        new_upstreams.append(healthy[old_idx])

    new_model_map = {}
    for mid, idxs in model_map.items():
        if mid in upstream_models:
            new_idxs = [old_to_new[i] for i in idxs if i in old_to_new]
            if new_idxs:
                new_model_map[mid] = new_idxs

    return new_upstreams, new_model_map


def create_app(upstreams: List[str], model_map: Optional[Dict[str, List[int]]] = None,
               upstream_models: Optional[Set[str]] = None,
               discover_interval: int = 0,
               log_dir: Optional[str] = None) -> FastAPI:
    router = LeastBusyRouter(upstreams, model_map=model_map)
    # Shared client: short connect timeout (5s), long read timeout (5min) for generations.
    # Large pool to handle many concurrent streaming requests without blocking.
    timeout = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=30.0)
    limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
    _client = [httpx.AsyncClient(timeout=timeout, limits=limits)]
    _CLIENT_RECYCLE_SECS = 1800  # recreate client every 30 min to avoid stale connections
    app = FastAPI()

    if model_map:
        print(f"[router] Model routing table:")
        for model, indices in sorted(model_map.items()):
            backends = [upstreams[i] for i in indices]
            print(f"  {model} -> {backends}")

    @app.on_event("shutdown")
    async def _shutdown():
        await _client[0].aclose()

    async def _recycle_client_loop():
        """Recreate HTTP client periodically to prevent connection pool staleness."""
        while True:
            await asyncio.sleep(_CLIENT_RECYCLE_SECS)
            try:
                old = _client[0]
                _client[0] = httpx.AsyncClient(timeout=timeout, limits=limits)
                # Grace period for in-flight requests on old client
                await asyncio.sleep(120)
                await old.aclose()
                print("[router] Recycled HTTP connection pool")
            except Exception as exc:
                print(f"[router] Client recycle error: {exc}")

    @app.on_event("startup")
    async def _start_recycle():
        asyncio.create_task(_recycle_client_loop())

    if discover_interval > 0:
        async def _auto_discover_loop():
            """Periodically discover backends from squeue and reload."""
            while True:
                await asyncio.sleep(discover_interval)
                try:
                    healthy, mmap = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: _discover_from_squeue(log_dir=log_dir)
                    )
                    if not healthy:
                        continue
                    healthy, mmap = _filter_by_models(healthy, mmap, upstream_models)
                    if not healthy:
                        continue
                    # Only reload if the set of backends changed
                    current = set(router.upstreams)
                    new_set = set(healthy)
                    if current != new_set:
                        router.reload(healthy, mmap)
                        print(f"[router] Auto-discover: {len(healthy)} backend(s), "
                              f"models={sorted(mmap.keys())}")
                except Exception as exc:
                    print(f"[router] Auto-discover error: {exc}")

        @app.on_event("startup")
        async def _start_discover():
            asyncio.create_task(_auto_discover_loop())

    async def _proxy_post(request: Request, endpoint: str):
        """Proxy a POST request to an upstream backend.

        Streaming requests (stream=true) use httpx streaming so SSE events
        are forwarded immediately instead of buffered, preventing connection
        pool exhaustion under sustained load.
        """
        payload = await request.json()
        model = payload.get("model")
        is_stream = payload.get("stream", False)
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        if model and router.model_map and model not in router.model_map:
            return Response(
                content=json.dumps({"error": {
                    "message": f"Model '{model}' not found. Available: {router.all_models}",
                    "type": "invalid_request_error",
                }}),
                status_code=404,
                media_type="application/json",
            )

        last_error = None
        tried = set()
        num_candidates = len(router.backends_for_model(model)) if model else len(router.upstreams)

        for _ in range(num_candidates):
            try:
                handle, backend = router.acquire_backend(exclude=tried, model=model)
            except RuntimeError:
                break
            _, idx = handle
            tried.add(idx)

            if is_stream:
                # Streaming path: open the upstream connection, check status,
                # then hand off to StreamingResponse which releases the handle
                # when the stream finishes (or client disconnects).
                try:
                    req = _client[0].build_request(
                        "POST", f"{backend}/{endpoint}",
                        json=payload, headers=headers,
                    )
                    upstream_resp = await _client[0].send(req, stream=True)
                except httpx.RequestError as exc:
                    router.release_backend(handle)
                    last_error = f"{backend}: {type(exc).__name__}: {exc}"
                    print(f"[router] {last_error}")
                    continue

                if upstream_resp.status_code >= 500:
                    await upstream_resp.aclose()
                    router.release_backend(handle)
                    last_error = f"{backend} returned {upstream_resp.status_code}"
                    print(f"[router] {last_error}")
                    continue

                # Success — stream response through. The generator's finally
                # block ensures cleanup even if the client disconnects early.
                async def _stream_body(resp=upstream_resp, h=handle):
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()
                        router.release_backend(h)

                return StreamingResponse(
                    _stream_body(),
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get(
                        "content-type", "text/event-stream"
                    ),
                )
            else:
                # Non-streaming path: read full response, release immediately.
                try:
                    response = await _client[0].post(
                        f"{backend}/{endpoint}", json=payload, headers=headers
                    )
                    if response.status_code >= 500:
                        last_error = f"{backend} returned {response.status_code}"
                        print(f"[router] {last_error}")
                        continue
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        media_type=response.headers.get("content-type"),
                    )
                except httpx.RequestError as exc:
                    last_error = f"{backend}: {type(exc).__name__}: {exc}"
                    print(f"[router] {last_error}")
                finally:
                    router.release_backend(handle)

        print(f"[router] All {len(tried)} backend(s) failed for {endpoint} model={model}")
        return Response(
            content=json.dumps({"error": last_error or "All upstreams failed."}),
            status_code=502,
            media_type="application/json",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_post(request, "chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_post(request, "completions")

    @app.get("/v1/models")
    async def list_models(request: Request):
        # If we have a model map, return the aggregated list directly
        if router.model_map:
            model_list = []
            for model_id in router.all_models:
                backend_indices = router.model_map[model_id]
                model_list.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "router",
                    "backends": len(backend_indices),
                })
            body = {"object": "list", "data": model_list}
            return Response(
                content=json.dumps(body),
                status_code=200,
                media_type="application/json",
            )

        # Fallback: proxy to a backend
        if not router.upstreams:
            return Response(
                content=json.dumps({"object": "list", "data": []}),
                status_code=200,
                media_type="application/json",
            )

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        last_error = None
        tried = set()

        for _ in range(len(router.upstreams)):
            handle, backend = router.acquire_backend(exclude=tried)
            _, idx = handle
            tried.add(idx)
            try:
                response = await _client[0].get(f"{backend}/models", headers=headers)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type"),
                )
            except httpx.RequestError as exc:
                last_error = str(exc)
            finally:
                router.release_backend(handle)

        error_body = {"error": last_error or "All upstreams failed."}
        return Response(
            content=json.dumps(error_body),
            status_code=502,
            media_type="application/json",
        )

    @app.post("/admin/reload")
    async def admin_reload(request: Request):
        """Hot-reload backends. Accepts {"upstreams": ["http://host:port/v1", ...]}."""
        try:
            payload = await request.json()
        except Exception:
            return Response(
                content=json.dumps({"error": "Invalid JSON body."}),
                status_code=400,
                media_type="application/json",
            )

        raw_upstreams = payload.get("upstreams", [])
        if not raw_upstreams:
            return Response(
                content=json.dumps({"error": "Missing 'upstreams' list."}),
                status_code=400,
                media_type="application/json",
            )

        # Normalize URLs
        normalized = [_normalize_base_url(u) for u in raw_upstreams]

        # Probe and discover models
        healthy, model_map = _probe_upstreams(normalized)

        if not healthy:
            return Response(
                content=json.dumps({
                    "error": "No healthy upstreams found.",
                    "probed": normalized,
                }),
                status_code=502,
                media_type="application/json",
            )

        # Filter to covered models only
        healthy, model_map = _filter_by_models(healthy, model_map, upstream_models)

        if not healthy:
            return Response(
                content=json.dumps({
                    "error": "No backends match covered models.",
                    "upstream_models": sorted(upstream_models) if upstream_models else "all",
                }),
                status_code=200,
                media_type="application/json",
            )

        router.reload(healthy, model_map)

        result = {
            "status": "reloaded",
            "upstreams": healthy,
            "models": {
                mid: [healthy[i] for i in idxs]
                for mid, idxs in model_map.items()
            },
        }
        print(f"[router] Admin reload: {len(healthy)} backends, "
              f"models={sorted(model_map.keys())}")
        return Response(
            content=json.dumps(result),
            status_code=200,
            media_type="application/json",
        )

    @app.get("/admin/status")
    async def admin_status():
        """Return current router state."""
        return Response(
            content=json.dumps({
                "upstreams": list(router.upstreams),
                "models": router.all_models,
                "model_map": {
                    mid: [router.upstreams[i] for i in idxs]
                    for mid, idxs in router.model_map.items()
                } if router.model_map else {},
                "upstream_models": sorted(upstream_models) if upstream_models else "all",
                "in_flight": list(router._in_flight),
            }),
            status_code=200,
            media_type="application/json",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI router")
    parser.add_argument(
        "--listen",
        default=os.getenv("LLM_ROUTER_HOST", "0.0.0.0"),
        help="Host to bind the router",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LLM_ROUTER_PORT", "8001")),
        help="Port to bind the router",
    )
    parser.add_argument(
        "--upstreams",
        default=os.getenv("LLM_ROUTER_UPSTREAMS") or os.getenv("LOCAL_LLM_BASE_URL"),
        help="Comma-separated base URLs or a port range like http://worker-0:30000-30007",
    )
    parser.add_argument(
        "--upstream-models",
        default=None,
        help='Comma-separated model names to cover, or "all" (default: all)',
    )
    parser.add_argument(
        "--discover-interval",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Auto-discover backends from squeue every N seconds (0 = disabled, default: 30)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory containing job .err logs (default: env LLM_LOG_DIR or ~/test_cluster/yuandong)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a quick test request to the running router and exit",
    )

    args = parser.parse_args()
    if args.test:
        if not args.upstreams:
            print("Missing --upstreams for test mode.")
            return

        upstreams = _parse_upstreams(args.upstreams)
        app = create_app(upstreams)

        async def _run_tests():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://router"
            ) as client:
                try:
                    models = await client.get("/v1/models", timeout=10.0)
                    print("GET /v1/models status:", models.status_code)
                    print(models.text)
                except Exception as exc:
                    print(f"Models request failed: {exc}")

                try:
                    payload = {
                        "model": os.getenv(
                            "LLM_MODEL", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
                        ),
                        "messages": [
                            {"role": "user", "content": "Hello from the router test."}
                        ],
                    }
                    resp = await client.post(
                        "/v1/chat/completions",
                        json=payload,
                        timeout=30.0,
                    )
                    print("POST /v1/chat/completions status:", resp.status_code)
                    print(resp.text)
                except Exception as exc:
                    print(f"Chat completion request failed: {exc}")

        asyncio.run(_run_tests())
        return

    # Parse --upstream-models
    upstream_models = None
    if args.upstream_models and args.upstream_models.lower() != "all":
        upstream_models = {m.strip() for m in args.upstream_models.split(",") if m.strip()}
        print(f"[router] Upstream models: {sorted(upstream_models)}")
    else:
        print("[router] All upstream models")

    if args.upstreams:
        healthy, model_map = _probe_upstreams(_parse_upstreams(args.upstreams))
    else:
        healthy, model_map = _discover_from_squeue(log_dir=args.log_dir)

    if healthy:
        healthy, model_map = _filter_by_models(healthy, model_map, upstream_models)

    if healthy:
        print(f"[router] {len(healthy)} backend(s), models={sorted(model_map.keys())}")
    else:
        print("[router] No backends found; use POST /admin/reload to add later.")
    if args.discover_interval > 0:
        print(f"[router] Auto-discover from squeue every {args.discover_interval}s")

    app = create_app(healthy, model_map=model_map, upstream_models=upstream_models,
                     discover_interval=args.discover_interval, log_dir=args.log_dir)

    uvicorn.run(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
