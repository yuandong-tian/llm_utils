# llm_utils

Lightweight helper for calling multiple LLM backends with caching.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python llm_util.py "Best way to solve Rubik's cube"
python llm_util.py "Summarize this text" --model gemini-2.5-flash
```

### OpenAI-compatible local server

```bash
python llm_util.py "Hello" --model openai-api
```

Provide the API base when instantiating `LLMCaller`:

```python
from llm_util import LLMCaller

caller = LLMCaller(default_model="local-model", api_base="http://localhost:30000/v1")
response, thinking = caller.generate("Hello", model_family="openai-api")
```

## OpenAI Router (Load Balancing)

`llm_router.py` exposes an OpenAI-compatible `/v1/chat/completions` endpoint and
load balances across multiple backends using least-busy routing.

```bash
python llm_router.py \
  --upstreams http://worker-0:30000-30007 \
  --port 8001
```

Then point your client at the router:

```bash
export LLM_API_BASE=http://localhost:8001
```

## Environment

`llm_util.py` will read `../config/bashrc` for keys if they are not already in the environment.

Supported variables (depending on model):
- `GEMINI_API_KEY`, `GEMINI_API_KEY2`, `GEMINI_API_KEY3`
- `KIMI_API_KEY`
- `DEEPSEEK_TOKEN`
- `XAI_API_KEY_X`

## Cache

Responses are cached in `~/llm_calls.db`.
