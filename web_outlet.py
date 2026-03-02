"""Web Outlet — Dynamic service registry portal with transparent proxying.

Runs a central hub on a fixed port (default 3772). Any uvicorn-based service
can register itself at runtime, making it instantly visible from the portal.
All traffic to registered services is relayed through the outlet — the
internal addresses are never exposed to the public.

Usage — start the outlet:
    python web_outlet.py
    python web_outlet.py --port 3772 --host 0.0.0.0

Usage — register your service (sync):
    from web_outlet import OutletClient

    client = OutletClient("http://localhost:3772")
    client.register("My API", "http://localhost:8080", description="Inference server", tags=["ml"])
    uvicorn.run(app, host="0.0.0.0", port=8080)
    client.deregister()

Usage — register with context manager (auto-deregisters on exit):
    with OutletClient("http://localhost:3772") as client:
        client.register("My API", "http://localhost:8080")
        uvicorn.run(app, host="0.0.0.0", port=8080)

Usage — async registration:
    async with AsyncOutletClient("http://localhost:3772") as client:
        await client.register("My API", "http://localhost:8080")
        ...

Registration payload (POST /api/register):
    { "name": str, "url": str, "description": str, "tags": [str], "ttl": int|null }

    ttl: optional time-to-live in seconds. If set, the entry expires unless
         re-registered before then. Useful for ephemeral processes.

Proxy:
    Every registered service is reachable at /proxy/{name}/ through the outlet.
    The outlet transparently relays all HTTP traffic (including streaming/SSE)
    to the service's internal URL, which is never exposed to the public.
"""

import argparse
import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# ─── Models ───────────────────────────────────────────────────────────────────


class ServiceRegistration(BaseModel):
    name: str
    url: str
    description: str = ""
    tags: List[str] = []
    ttl: Optional[int] = None  # seconds until expiry; None = permanent


class ServiceEntry(BaseModel):
    name: str
    url: str           # internal — never sent to the browser
    description: str = ""
    tags: List[str] = []
    registered_at: float = 0.0
    expires_at: Optional[float] = None  # None = permanent


# ─── Registry ─────────────────────────────────────────────────────────────────

_registry: dict[str, ServiceEntry] = {}

# Headers that must not be forwarded through a proxy
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


def _prune_expired() -> None:
    now = time.time()
    expired = [
        k for k, v in _registry.items() if v.expires_at is not None and v.expires_at <= now
    ]
    for k in expired:
        del _registry[k]


def _public_entry(s: ServiceEntry) -> dict:
    """Return a service entry safe to expose publicly (no internal URL)."""
    return {
        "name": s.name,
        "description": s.description,
        "tags": s.tags,
        "registered_at": s.registered_at,
        "expires_at": s.expires_at,
    }


# ─── Lifespan ─────────────────────────────────────────────────────────────────


async def _pruner():
    while True:
        await asyncio.sleep(5)
        _prune_expired()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_pruner())
    yield
    task.cancel()


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Web Outlet", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def portal():
    return HTMLResponse(_PORTAL_HTML)


@app.get("/api/services")
async def list_services():
    _prune_expired()
    return JSONResponse([_public_entry(s) for s in _registry.values()])


@app.post("/api/register", status_code=201)
async def register(svc: ServiceRegistration):
    expires_at = (time.time() + svc.ttl) if svc.ttl else None
    _registry[svc.name] = ServiceEntry(
        name=svc.name,
        url=svc.url,
        description=svc.description,
        tags=svc.tags,
        registered_at=time.time(),
        expires_at=expires_at,
    )
    return {"status": "registered", "name": svc.name}


@app.delete("/api/register/{name}")
async def deregister(name: str):
    if name not in _registry:
        raise HTTPException(404, f"Service '{name}' not found")
    del _registry[name]
    return {"status": "deregistered", "name": name}


# ─── Reverse Proxy ────────────────────────────────────────────────────────────

_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


@app.api_route("/proxy/{name}", methods=_PROXY_METHODS)
@app.api_route("/proxy/{name}/{path:path}", methods=_PROXY_METHODS)
async def proxy(request: Request, name: str, path: str = ""):
    _prune_expired()
    if name not in _registry:
        raise HTTPException(404, f"Service '{name}' is not registered")

    svc = _registry[name]
    target = svc.url.rstrip("/")
    if path:
        target += "/" + path
    if request.url.query:
        target += "?" + request.url.query

    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }

    body = await request.body()

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=120, write=60, pool=10),
        follow_redirects=True,
    )
    try:
        upstream_req = client.build_request(
            method=request.method,
            url=target,
            headers=fwd_headers,
            content=body,
        )
        resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise HTTPException(502, f"Cannot reach '{name}' — is it running?")
    except httpx.TimeoutException:
        await client.aclose()
        raise HTTPException(504, f"'{name}' timed out")
    except Exception as exc:
        await client.aclose()
        raise HTTPException(502, str(exc))

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "content-length"
    }

    async def stream_body():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=resp.status_code,
        headers=resp_headers,
    )


# ─── Client (sync) ────────────────────────────────────────────────────────────


class OutletClient:
    """Synchronous client for registering a service with the Web Outlet.

    Example:
        client = OutletClient("http://localhost:3772")
        client.register("My API", "http://localhost:8080", description="...", ttl=60)
        uvicorn.run(app)
        client.deregister()

    Or as a context manager:
        with OutletClient() as client:
            client.register("My API", "http://localhost:8080")
            uvicorn.run(app)   # blocks; deregister runs on exit
    """

    def __init__(self, outlet_url: str = "http://localhost:3772"):
        self.outlet_url = outlet_url.rstrip("/")
        self._registered_name: Optional[str] = None

    def register(
        self,
        name: str,
        url: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> "OutletClient":
        payload = {
            "name": name,
            "url": url,
            "description": description,
            "tags": tags or [],
            "ttl": ttl,
        }
        with httpx.Client(timeout=5) as client:
            r = client.post(f"{self.outlet_url}/api/register", json=payload)
            r.raise_for_status()
        self._registered_name = name
        return self

    def deregister(self, name: Optional[str] = None) -> None:
        target = name or self._registered_name
        if not target:
            return
        try:
            with httpx.Client(timeout=5) as client:
                client.delete(f"{self.outlet_url}/api/register/{target}")
        except Exception:
            pass
        if target == self._registered_name:
            self._registered_name = None

    def __enter__(self) -> "OutletClient":
        return self

    def __exit__(self, *_) -> None:
        self.deregister()


# ─── Client (async) ───────────────────────────────────────────────────────────


class AsyncOutletClient:
    """Async client for registering a service with the Web Outlet.

    Example:
        async with AsyncOutletClient() as client:
            await client.register("My API", "http://localhost:8080")
            ...
    """

    def __init__(self, outlet_url: str = "http://localhost:3772"):
        self.outlet_url = outlet_url.rstrip("/")
        self._registered_name: Optional[str] = None

    async def register(
        self,
        name: str,
        url: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> "AsyncOutletClient":
        payload = {
            "name": name,
            "url": url,
            "description": description,
            "tags": tags or [],
            "ttl": ttl,
        }
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(f"{self.outlet_url}/api/register", json=payload)
            r.raise_for_status()
        self._registered_name = name
        return self

    async def deregister(self, name: Optional[str] = None) -> None:
        target = name or self._registered_name
        if not target:
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.delete(f"{self.outlet_url}/api/register/{target}")
        except Exception:
            pass
        if target == self._registered_name:
            self._registered_name = None

    async def __aenter__(self) -> "AsyncOutletClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.deregister()


# ─── Portal HTML ──────────────────────────────────────────────────────────────

_PORTAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WEB OUTLET</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:        #05050a;
      --bg2:       #0b0b13;
      --bg3:       #10101a;
      --border:    #1c1c2e;
      --amber:     #e8a020;
      --amber-lo:  #7a5210;
      --amber-dim: #4a3208;
      --green:     #2ecc71;
      --green-lo:  #1a5c3a;
      --red:       #e84040;
      --text:      #c8b896;
      --text-lo:   #605040;
      --text-dim:  #3a3028;
      --mono: 'JetBrains Mono', 'Courier New', monospace;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--mono);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* CRT scanlines */
    body::before {
      content: '';
      position: fixed; inset: 0; z-index: 9999; pointer-events: none;
      background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.12) 2px, rgba(0,0,0,0.12) 4px
      );
    }
    /* Edge vignette */
    body::after {
      content: '';
      position: fixed; inset: 0; z-index: 9998; pointer-events: none;
      background: radial-gradient(ellipse 120% 80% at 50% 50%, transparent 55%, rgba(0,0,0,0.7) 100%);
    }

    /* ── Header ──────────────────────────────────────────────── */
    header {
      display: flex; align-items: center; gap: 2rem;
      padding: 1.5rem 2.5rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(232,160,32,0.03) 0%, transparent 100%);
    }
    .logo {
      display: flex; align-items: baseline; gap: 0.6rem;
      font-size: 1.3rem; font-weight: 700; letter-spacing: 0.22em;
      color: var(--amber);
      text-shadow: 0 0 30px rgba(232,160,32,0.35), 0 0 60px rgba(232,160,32,0.12);
      white-space: nowrap;
    }
    .logo-glyph { font-size: 1.1rem; opacity: 0.6; }
    .blink { animation: blink 1.1s step-end infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    .header-right { margin-left: auto; display: flex; align-items: center; gap: 2rem; }
    .stat-box { text-align: right; line-height: 1.2; }
    .stat-val {
      font-size: 1.4rem; font-weight: 700; color: var(--green);
      text-shadow: 0 0 16px rgba(46,204,113,0.4);
    }
    .stat-label { font-size: 0.6rem; letter-spacing: 0.25em; color: var(--text-lo); }

    .pulse-ring {
      position: relative; width: 28px; height: 28px;
      display: flex; align-items: center; justify-content: center;
    }
    .pulse-ring::before {
      content: '';
      position: absolute; inset: 0; border-radius: 50%;
      border: 1px solid var(--green);
      animation: ring 3s ease-out infinite;
    }
    .pulse-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 10px var(--green), 0 0 20px rgba(46,204,113,0.4);
    }
    @keyframes ring {
      0%   { transform: scale(0.6); opacity: 0.8; }
      100% { transform: scale(1.8); opacity: 0; }
    }

    /* ── Layout ──────────────────────────────────────────────── */
    .main { padding: 2rem 2.5rem 4rem; }

    .section-label {
      font-size: 0.58rem; letter-spacing: 0.35em; color: var(--text-lo);
      text-transform: uppercase;
      display: flex; align-items: center; gap: 1rem;
      margin-bottom: 1.25rem;
    }
    .section-label::after {
      content: ''; flex: 1; height: 1px;
      background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
    }

    /* ── Service Grid ─────────────────────────────────────────── */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 1px;
      border: 1px solid var(--border);
      background: var(--border);
    }

    .card {
      background: var(--bg);
      padding: 1.4rem 1.5rem;
      cursor: pointer;
      position: relative;
      overflow: hidden;
      transition: background 0.2s;
      text-decoration: none;
      display: block;
      color: inherit;
    }
    .card::after {
      content: '';
      position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(232,160,32,0.03) 0%, transparent 60%);
      opacity: 0; transition: opacity 0.25s;
      pointer-events: none;
    }
    .card:hover { background: var(--bg3); }
    .card:hover::after { opacity: 1; }

    .card-accent {
      position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, var(--amber) 0%, transparent 100%);
      transform: scaleX(0); transform-origin: left;
      transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .card:hover .card-accent { transform: scaleX(1); }

    .card-status {
      font-size: 0.55rem; letter-spacing: 0.25em;
      color: var(--green); margin-bottom: 0.5rem;
      display: flex; align-items: center; gap: 0.4rem;
    }
    .card-status-dot {
      width: 5px; height: 5px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 6px var(--green);
      flex-shrink: 0;
    }

    .card-name {
      font-size: 0.95rem; font-weight: 700;
      color: var(--amber); letter-spacing: 0.05em; margin-bottom: 0.3rem;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* proxy path badge — replaces exposed URL */
    .card-proxy {
      display: inline-flex; align-items: center; gap: 0.4rem;
      font-size: 0.65rem; color: var(--text-lo);
      margin-bottom: 0.85rem; letter-spacing: 0.05em;
    }
    .card-proxy-arrow { color: var(--green-lo); }

    .card-desc {
      font-size: 0.75rem; line-height: 1.65;
      color: var(--text); opacity: 0.6;
      margin-bottom: 0.9rem;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .card-footer {
      display: flex; align-items: flex-end; justify-content: space-between; gap: 0.5rem;
    }
    .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; }
    .tag {
      font-size: 0.56rem; letter-spacing: 0.12em;
      padding: 0.15rem 0.45rem;
      border: 1px solid var(--amber-lo); color: var(--amber-lo);
    }
    .card-time { font-size: 0.58rem; color: var(--text-dim); white-space: nowrap; }

    /* open arrow shown on hover */
    .card-open {
      position: absolute; top: 1.2rem; right: 1.2rem;
      font-size: 0.8rem; color: var(--amber-lo);
      opacity: 0; transform: translateX(-4px);
      transition: opacity 0.2s, transform 0.2s;
    }
    .card:hover .card-open { opacity: 1; transform: none; }

    /* expire bar */
    .expire-bar-wrap { height: 2px; background: var(--bg3); margin-top: 0.75rem; overflow: hidden; }
    .expire-bar { height: 100%; background: var(--amber-lo); transition: width 1s linear; }

    /* ── Empty state ──────────────────────────────────────────── */
    .empty {
      grid-column: 1 / -1; background: var(--bg);
      padding: 4rem 2rem; text-align: center;
    }
    .empty-title { font-size: 0.8rem; letter-spacing: 0.25em; color: var(--text-lo); margin-bottom: 1rem; }
    .empty-hint { font-size: 0.65rem; color: var(--text-dim); line-height: 2.2; }
    .empty-hint code { color: var(--amber-lo); background: var(--bg2); padding: 0.1rem 0.4rem; }

    /* entry animation */
    .card { animation: fadeIn 0.3s ease both; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  </style>
</head>
<body>

<header>
  <div class="logo">
    <span class="logo-glyph">⬡</span>
    WEB OUTLET
    <span class="blink" style="font-weight:300;color:var(--amber-lo)">_</span>
  </div>
  <div class="header-right">
    <div class="stat-box">
      <div class="stat-val" id="svc-count">0</div>
      <div class="stat-label">SERVICES</div>
    </div>
    <div class="pulse-ring" title="Live — polls every 3s">
      <div class="pulse-dot"></div>
    </div>
  </div>
</header>

<div class="main">
  <div class="section-label">REGISTERED SERVICES</div>
  <div class="grid" id="grid"></div>
</div>

<script>
  const REFRESH_MS = 3000;

  function esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function timeAgo(ts) {
    if (!ts) return '';
    const d = Math.floor(Date.now() / 1000 - ts);
    if (d < 5)    return 'just now';
    if (d < 60)   return d + 's ago';
    if (d < 3600) return Math.floor(d / 60) + 'm ago';
    return Math.floor(d / 3600) + 'h ago';
  }

  function ttlPct(entry) {
    if (!entry.expires_at) return null;
    const total = entry.expires_at - entry.registered_at;
    const left  = entry.expires_at - Date.now() / 1000;
    return Math.max(0, Math.min(100, (left / total) * 100));
  }

  function renderGrid(data) {
    const grid = document.getElementById('grid');
    document.getElementById('svc-count').textContent = data.length;

    if (data.length === 0) {
      grid.innerHTML = `
        <div class="empty">
          <div class="empty-title">NO SERVICES REGISTERED</div>
          <div class="empty-hint">
            From any uvicorn service:<br>
            <code>from web_outlet import OutletClient</code><br>
            <code>OutletClient().register("name", "http://localhost:PORT")</code>
          </div>
        </div>`;
      return;
    }

    grid.innerHTML = data.map((s, i) => {
      const proxyPath = `/proxy/${encodeURIComponent(s.name)}/`;
      const pct = ttlPct(s);
      const expireBar = pct !== null
        ? `<div class="expire-bar-wrap"><div class="expire-bar" style="width:${pct}%"></div></div>`
        : '';
      const tags = (s.tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join('');
      return `
        <a class="card" href="${proxyPath}" target="_blank" style="animation-delay:${i * 40}ms">
          <div class="card-accent"></div>
          <div class="card-open">↗</div>
          <div class="card-status"><span class="card-status-dot"></span>ONLINE</div>
          <div class="card-name">${esc(s.name)}</div>
          <div class="card-proxy">
            <span class="card-proxy-arrow">→</span>
            <span>${esc(proxyPath)}</span>
          </div>
          ${s.description ? `<div class="card-desc">${esc(s.description)}</div>` : ''}
          <div class="card-footer">
            <div class="tags">${tags}</div>
            <span class="card-time">${timeAgo(s.registered_at)}</span>
          </div>
          ${expireBar}
        </a>`;
    }).join('');
  }

  async function fetchServices() {
    try {
      const r = await fetch('/api/services');
      if (!r.ok) return;
      renderGrid(await r.json());
    } catch (_) {}
  }

  fetchServices();
  setInterval(fetchServices, REFRESH_MS);
</script>
</body>
</html>
"""


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web Outlet — dynamic service registry portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=3772, help="Bind port (default: 3772)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"  ⬡  Web Outlet  →  http://{args.host}:{args.port}")
    uvicorn.run(
        "web_outlet:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
