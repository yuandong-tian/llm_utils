#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <url> [prompt...]"
  exit 1
fi

url="$1"
shift

temperature="${TEMPERATURE:-0}"
if [[ $# -ge 1 && "$1" == --temperature=* ]]; then
  temperature="${1#--temperature=}"
  shift
elif [[ $# -ge 2 && "$1" == "--temperature" ]]; then
  temperature="$2"
  shift 2
fi

if [[ $# -eq 0 ]]; then
  prompt_text=$(cat)
  set -- "$prompt_text"
fi

payload=$(python3 - "$temperature" "$@" <<'PY'
import json, sys
temperature = float(sys.argv[1])
prompts = sys.argv[2:]
payload = {
    "model": "minimax-m2.5",
    "prompt": prompts,
    "max_tokens": 2048,
    "temperature": temperature,
}
print(json.dumps(payload))
PY
)

curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d "$payload" \
  "${url%/}/v1/completions"
