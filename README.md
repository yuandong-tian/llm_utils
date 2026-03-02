# llm_utils

Lightweight helper for calling multiple LLM backends with caching, plus an
optional OpenAI-compatible router for load balancing.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python llm_util.py "Best way to solve Rubik's cube"
python llm_util.py "Summarize this text" --model gemini-2.5-flash
python llm_util.py "Explain chain-of-thought" --model deepseek-r1 --print-thinking
```

### OpenAI-compatible server (single or multiple bases)

```bash
python llm_util.py "Hello" --model openai-api
```

Provide the API base when instantiating `LLMCaller`:

```python
from llm_util import LLMCaller

caller = LLMCaller(
    default_model="local-model",
    api_base="http://localhost:30000/v1,http://localhost:30001/v1",
)
response, thinking = caller.generate("Hello", model_family="openai-api")
```

### Image inputs (OpenAI-compatible)

```python
from llm_util import LLMCaller

caller = LLMCaller(
    default_model="gpt-5-mini",
    api_base="https://api.openai.com/v1",
)
response, thinking = caller.generate_with_images(
    "Describe this image.",
    image_paths=["/path/to/image.png"],
    model_family="openai-api",
)
```

### Supported model families

- Gemini: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro-preview-05-06`
- OpenAI: `gpt-5.1`, `gpt-5-mini`, `gpt-5-nano`, `o3`, `o4-mini`
- DeepSeek: `deepseek-v3`, `deepseek-r1`
- xAI: `grok-2`
- Moonshot: `kimi-k2-0905-preview`, `kimi-k2-turbo-preview`
- Ollama: `deepseek-r1:32b`, `gemma3:27b`, `qwen3:32b`, `qwen3:8b`, `qwen3:4b`, `qwen3:4b-instruct`
- OpenAI-compatible passthrough: `openai-api`

## OpenAI Router (Load Balancing)

`llm_router.py` exposes an OpenAI-compatible `/v1/chat/completions` endpoint and
`/v1/models`, load balances across multiple backends using least-busy routing,
and retries another upstream on network errors.

```bash
python llm_router.py \
  --listen 0.0.0.0 \
  --upstreams http://worker-0:30000-30007 \
  --port 8001
```

Then point your client at the router:

```bash
export LLM_API_BASE=http://localhost:8001
```

Quick sanity check against the in-process test router:

```bash
python llm_router.py --upstreams http://localhost:30000-30007 --test
```

If you want to send a raw request, `run_query.sh` can be used:

```bash
./run_query.sh http://localhost:8001/v1/chat/completions "Hello router"
```

## Web Outlet (Service Registry Portal)

`web_outlet.py` runs a lightweight portal on a fixed port. Any uvicorn-based
service can register itself at runtime, making it instantly visible and
linkable from the portal page.

### Start the outlet

```bash
python web_outlet.py --port PORT
```

Open `http://localhost:PORT` to see the portal. It polls for changes every 3s.

### Register your service

**Context manager (auto-deregisters on exit):**
```python
from web_outlet import OutletClient
import uvicorn

with OutletClient("http://localhost:PORT") as client:
    client.register("My API", "http://localhost:8080",
                    description="LLaMA server", tags=["ml", "llm"])
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Manual register / deregister:**
```python
client = OutletClient("http://localhost:PORT")
client.register("My API", "http://localhost:8080")
uvicorn.run(app, host="0.0.0.0", port=8080)
client.deregister()
```

**With TTL — entry auto-expires if the process dies:**
```python
client.register("Worker", "http://localhost:8080", ttl=30)
# re-call register() every ~20s to refresh
```

**Async:**
```python
from web_outlet import AsyncOutletClient

async with AsyncOutletClient("http://localhost:PORT") as client:
    await client.register("My API", "http://localhost:8080")
    ...
```

### Raw API

```bash
# Register
curl -X POST http://localhost:PORT/api/register \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-svc","url":"http://localhost:8080","tags":["api"],"ttl":60}'

# List
curl http://localhost:PORT/api/services

# Remove
curl -X DELETE http://localhost:PORT/api/register/my-svc
```

## Environment

`llm_util.py` will read `../config/bashrc` for keys if they are not already in the environment.

Supported variables (depending on model):
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`, `GEMINI_API_KEY2`, `GEMINI_API_KEY3`
- `KIMI_API_KEY`
- `DEEPSEEK_TOKEN`
- `XAI_API_KEY_X`

## Cache

Responses are cached in `~/llm_calls.db`.
