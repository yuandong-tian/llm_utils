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
from typing import List, Optional

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
    """Thread-safe router that directs requests to the least-busy upstream server."""

    def __init__(self, upstreams: List[str]):
        self._upstreams = upstreams
        self._in_flight = [0] * len(upstreams)
        self._lock = threading.Lock()

    def acquire_backend(self, exclude: Optional[set] = None) -> int:
        with self._lock:
            candidates = [
                (idx, count)
                for idx, count in enumerate(self._in_flight)
                if not exclude or idx not in exclude
            ]
            if not candidates:
                raise RuntimeError("No available upstreams.")
            idx, _ = min(candidates, key=lambda item: item[1])
            self._in_flight[idx] += 1
            return idx

    def release_backend(self, idx: int) -> None:
        with self._lock:
            self._in_flight[idx] = max(0, self._in_flight[idx] - 1)

    @property
    def upstreams(self) -> List[str]:
        return self._upstreams


def _probe_upstreams(upstreams: List[str]) -> List[str]:
    healthy = []
    with httpx.Client(timeout=5.0) as client:
        for upstream in upstreams:
            try:
                response = client.get(f"{upstream}/models")
                if response.status_code < 500:
                    healthy.append(upstream)
                    print(f"[router] Upstream OK: {upstream}")
                else:
                    print(f"[router] Upstream error {response.status_code}: {upstream}")
            except httpx.RequestError as exc:
                print(f"[router] Upstream unreachable: {upstream} ({exc})")
    return healthy


def create_app(upstreams: List[str]) -> FastAPI:
    router = LeastBusyRouter(upstreams)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        last_error = None
        tried = set()

        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(len(router.upstreams)):
                idx = router.acquire_backend(exclude=tried)
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
    healthy = _probe_upstreams(upstreams)
    if not healthy:
        raise RuntimeError("No healthy upstreams available; aborting router startup.")
    if len(healthy) < len(upstreams):
        print("[router] Some upstreams are unhealthy and will be skipped.")
    app = create_app(healthy)

    uvicorn.run(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
