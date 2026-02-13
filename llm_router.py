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


class LeastBusyRouter:
    """Thread-safe router that directs requests to the least-busy upstream server.

    Supports model-aware routing: if a model_map is provided, only backends
    that serve the requested model are considered.
    """

    def __init__(self, upstreams: List[str], model_map: Optional[Dict[str, List[int]]] = None):
        self._upstreams = upstreams
        self._in_flight = [0] * len(upstreams)
        self._lock = threading.Lock()
        # model_name -> list of upstream indices that serve it
        self._model_map = model_map or {}

    def acquire_backend(self, exclude: Optional[set] = None, model: Optional[str] = None) -> int:
        with self._lock:
            # If model specified and we have a mapping, restrict to those backends
            allowed = None
            if model and self._model_map:
                allowed = set(self._model_map.get(model, []))
                if not allowed:
                    # Model not found in map; fall back to all backends
                    allowed = None

            candidates = [
                (idx, count)
                for idx, count in enumerate(self._in_flight)
                if (not exclude or idx not in exclude)
                and (allowed is None or idx in allowed)
            ]
            if not candidates:
                raise RuntimeError(f"No available upstreams for model '{model}'.")
            idx, _ = min(candidates, key=lambda item: item[1])
            self._in_flight[idx] += 1
            return idx

    def release_backend(self, idx: int) -> None:
        with self._lock:
            self._in_flight[idx] = max(0, self._in_flight[idx] - 1)

    def backends_for_model(self, model: str) -> List[int]:
        """Return upstream indices that serve a given model."""
        if self._model_map:
            return self._model_map.get(model, list(range(len(self._upstreams))))
        return list(range(len(self._upstreams)))

    @property
    def upstreams(self) -> List[str]:
        return self._upstreams

    @property
    def model_map(self) -> Dict[str, List[int]]:
        return self._model_map

    @property
    def all_models(self) -> List[str]:
        return sorted(self._model_map.keys())


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


def create_app(upstreams: List[str], model_map: Optional[Dict[str, List[int]]] = None) -> FastAPI:
    router = LeastBusyRouter(upstreams, model_map=model_map)
    app = FastAPI()

    if model_map:
        print(f"[router] Model routing table:")
        for model, indices in sorted(model_map.items()):
            backends = [upstreams[i] for i in indices]
            print(f"  {model} -> {backends}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        model = payload.get("model")
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        # Check if model is known
        if model and router.model_map and model not in router.model_map:
            error_body = {
                "error": {
                    "message": f"Model '{model}' not found. Available: {router.all_models}",
                    "type": "invalid_request_error",
                }
            }
            return Response(
                content=json.dumps(error_body),
                status_code=404,
                media_type="application/json",
            )

        last_error = None
        tried = set()
        num_candidates = len(router.backends_for_model(model)) if model else len(router.upstreams)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(num_candidates):
                try:
                    idx = router.acquire_backend(exclude=tried, model=model)
                except RuntimeError:
                    break
                backend = router.upstreams[idx]
                tried.add(idx)
                try:
                    upstream_url = f"{backend}/chat/completions"
                    response = await client.post(
                        upstream_url, json=payload, headers=headers
                    )
                    media_type = response.headers.get("content-type")
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        media_type=media_type,
                    )
                except httpx.RequestError as exc:
                    last_error = str(exc)
                finally:
                    router.release_backend(idx)

        error_body = {"error": last_error or "All upstreams failed."}
        return Response(
            content=json.dumps(error_body),
            status_code=502,
            media_type="application/json",
        )

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
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        last_error = None
        tried = set()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(len(router.upstreams)):
                idx = router.acquire_backend(exclude=tried)
                backend = router.upstreams[idx]
                tried.add(idx)
                try:
                    upstream_url = f"{backend}/models"
                    response = await client.get(upstream_url, headers=headers)
                    media_type = response.headers.get("content-type")
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        media_type=media_type,
                    )
                except httpx.RequestError as exc:
                    last_error = str(exc)
                finally:
                    router.release_backend(idx)

        error_body = {"error": last_error or "All upstreams failed."}
        return Response(
            content=json.dumps(error_body),
            status_code=502,
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

    upstreams = _parse_upstreams(args.upstreams)
    healthy, model_map = _probe_upstreams(upstreams)
    if not healthy:
        raise RuntimeError("No healthy upstreams available; aborting router startup.")
    if len(healthy) < len(upstreams):
        print("[router] Some upstreams are unhealthy and will be skipped.")
    if model_map:
        print(f"[router] Discovered {len(model_map)} model(s) across {len(healthy)} backend(s)")
    app = create_app(healthy, model_map=model_map)

    uvicorn.run(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
