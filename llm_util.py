# Standard library imports
import argparse
import asyncio
import base64
import os
import random
import re
import sqlite3
import subprocess
import threading

# Third-party imports
from google import genai
from openai import OpenAI
import tolerantjson

try:
    import ollama
except ImportError:
    ollama = None

DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe this audio clip. The audio clip can be either in English or in "
    "Simplified Chinese, keep the language when outputting. "
    "When outputting Chinese, output Simplified Chinese rather than Traditional Chinese."
)

def load_bash_env(bashrc_path: str = None) -> None:
    """Loads environment variables from a bashrc file into os.environ."""
    if bashrc_path is None:
        bashrc_path = os.path.join(os.path.dirname(__file__), '../config/bashrc')
    
    if not os.path.exists(bashrc_path):
        return

    # Use bash to source the file and print the environment
    cmd = f"source '{bashrc_path}' > /dev/null 2>&1 && env"
    
    try:
        result = subprocess.run(
            ['bash', '-c', cmd], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except Exception as e:
        print(f"Error loading environment variables from {bashrc_path}: {e}")

_BASH_ENV_LOADED = False

def get_env_var(var_name: str) -> str:
    global _BASH_ENV_LOADED
    val = os.environ.get(var_name)
    if val is None and not _BASH_ENV_LOADED:
        load_bash_env()
        _BASH_ENV_LOADED = True
        val = os.environ.get(var_name)
    return val

# Database file name, save in a global place so that it can be accessed by other modules
DB_FILE = os.path.expanduser('~/llm_calls.db')

# If the database doesn't exist, create it
if not os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute('CREATE TABLE llm_calls (prompt TEXT, model_name TEXT, params TEXT, thinking TEXT, result TEXT, parse_json BOOLEAN)')
    conn.close()

class MoonShotAPI:
    """API wrapper for Moonshot AI (Kimi) models.

    Requires KIMI_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "kimi-k2-0905-preview"):
        if not get_env_var("KIMI_API_KEY"):
            raise RuntimeError("Missing KIMI_API_KEY for MoonShot API.")
        self.model_name = model_name
        self.client = OpenAI(
            api_key = get_env_var("KIMI_API_KEY"),
            base_url = "https://api.moonshot.ai/v1",
        )
        
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = [
                {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.6, # controls randomness of output
        )
 
        return response.choices[0].message.content.strip(), ""

class GeminiAPI:
    """API wrapper for Google Gemini models.

    Supports key rotation across multiple API keys (GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3)
    to distribute load and avoid rate limits.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", use_key_rotation: bool = True):
        ENV_KEYS = ['GEMINI_API_KEY', 'GEMINI_API_KEY2', 'GEMINI_API_KEY3']
        # Randomly pick one of the API keys
        if use_key_rotation:
            env_key = random.choice(ENV_KEYS)
        else:
            env_key = 'GEMINI_API_KEY'
        api_key = get_env_var(env_key)
        if not api_key:
            raise RuntimeError(f"Missing {env_key} for Gemini API.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        result = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        # No thinking process for Gemini
        return result.text.strip(), ""


def transcribe_audio_gemini(audio_path: str, model: str = "gemini-2.5-flash") -> str:
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY for Gemini transcription.")
    client = genai.Client(api_key=api_key)
    gemini_file = client.files.upload(file=audio_path)
    response = client.models.generate_content(
        model=model,
        contents=[DEFAULT_TRANSCRIBE_PROMPT, gemini_file],
    )
    return response.text.strip()
    
    
class DeepSeekAPI:
    """API wrapper for DeepSeek models.

    Supports both deepseek-chat and deepseek-reasoner models.
    The reasoner model returns thinking/reasoning content alongside the response.
    Requires DEEPSEEK_TOKEN environment variable.
    """

    def __init__(self, model_name: str = "deepseek-reasoner"):
        if not get_env_var("DEEPSEEK_TOKEN"):
            raise RuntimeError("Missing DEEPSEEK_TOKEN for DeepSeek API.")
        self.model = OpenAI(
            api_key=get_env_var("DEEPSEEK_TOKEN"), 
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False)
        
        result = response.choices[0].message.content

        if hasattr(response.choices[0].message, "reasoning_content"):  
            thinking = response.choices[0].message.reasoning_content
        else:
            thinking = ""
        return result, thinking

class GrokAPI:
    """API wrapper for xAI Grok models.

    Requires XAI_API_KEY_X environment variable.
    """

    def __init__(self, model_name: str = "grok-2-latest"):
        if not get_env_var("XAI_API_KEY_X"):
            raise RuntimeError("Missing XAI_API_KEY_X for Grok API.")
        self.model = OpenAI(
            api_key=get_env_var("XAI_API_KEY_X"),
            base_url="https://api.x.ai/v1",
        )
        self.model_name = model_name
        
    def generate(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        )

        return response.choices[0].message.content, ""

class OpenAIAPI:
    """API wrapper for OpenAI and OpenAI-compatible endpoints.

    Supports multiple API base URLs for load balancing across servers.
    Pass comma-separated URLs or a list to api_base for multi-backend support.
    Thread-safe with automatic client pooling.
    """

    def __init__(self, model_name: str, api_base: str = "https://api.openai.com", api_key_env: str = "OPENAI_API_KEY"):
        if not api_base:
            raise RuntimeError("Missing api_base for OpenAI-compatible API.")
        self.model_name = model_name
        self.api_key = get_env_var(api_key_env) or "EMPTY"
        self._api_bases = self._normalize_api_bases(api_base)
        self._clients = [None] * len(self._api_bases)
        self._idle_lock = threading.Lock()
        self._busy_flags = [False] * len(self._api_bases)
        self._idle_cond = threading.Condition(self._idle_lock)

    def _normalize_api_bases(self, api_base):
        if isinstance(api_base, (list, tuple)):
            parts = [str(base).strip() for base in api_base if str(base).strip()]
        elif isinstance(api_base, str):
            parts = [part.strip() for part in api_base.split(",") if part.strip()] or [api_base.strip()]
        else:
            parts = [str(api_base).strip()]

        normalized = []
        for base in parts:
            if not base:
                continue
            base = base.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            normalized.append(base)

        if not normalized:
            raise RuntimeError("Missing api_base for OpenAI-compatible API.")
        return normalized

    def _build_client(self, api_base: str):
        return OpenAI(
            api_key=self.api_key,
            base_url=api_base,
        )

    def _ensure_api_key(self) -> None:
        if not self.api_key or self.api_key == "EMPTY":
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI-compatible API.")

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def _build_image_messages(self, prompt: str, image_paths: list[str]) -> list[dict]:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            image_b64 = self._encode_image(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
        return [{"role": "user", "content": content}]

    def generate(self, prompt: str) -> str:
        if len(self._api_bases) == 1:
            if self._clients[0] is None:
                self._clients[0] = self._build_client(self._api_bases[0])
            client = self._clients[0]
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content, ""

        with self._idle_cond:
            while True:
                idle_indices = [i for i, busy in enumerate(self._busy_flags) if not busy]
                if idle_indices:
                    idx = idle_indices[0]
                    self._busy_flags[idx] = True
                    break
                self._idle_cond.wait()

            if self._clients[idx] is None:
                self._clients[idx] = self._build_client(self._api_bases[idx])
            client = self._clients[idx]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content, ""
        finally:
            with self._idle_cond:
                self._busy_flags[idx] = False
                self._idle_cond.notify()

    def generate_with_images(self, prompt: str, image_paths: list[str], max_tokens: int = 800) -> str:
        self._ensure_api_key()
        if len(self._api_bases) == 1:
            if self._clients[0] is None:
                self._clients[0] = self._build_client(self._api_bases[0])
            client = self._clients[0]
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self._build_image_messages(prompt, image_paths),
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content, ""

        with self._idle_cond:
            while True:
                idle_indices = [i for i, busy in enumerate(self._busy_flags) if not busy]
                if idle_indices:
                    idx = idle_indices[0]
                    self._busy_flags[idx] = True
                    break
                self._idle_cond.wait()

            if self._clients[idx] is None:
                self._clients[idx] = self._build_client(self._api_bases[idx])
            client = self._clients[idx]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self._build_image_messages(prompt, image_paths),
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content, ""
        finally:
            with self._idle_cond:
                self._busy_flags[idx] = False
                self._idle_cond.notify()


def get_default_api_base(model_name: str) -> str | None:
    if not model_name:
        return None
    name = model_name.strip().lower()

    rules = [
        ("https://api.openai.com/v1", ("gpt-", "o3", "o4-")),
        ("https://api.moonshot.ai/v1", ("kimi-",)),
        ("https://api.deepseek.com/v1", ("deepseek-",)),
        ("https://api.x.ai/v1", ("grok-",)),
    ]

    for api_base, prefixes in rules:
        for prefix in prefixes:
            if name.startswith(prefix):
                return api_base
    return None


class OllamaAPI:
    """API wrapper for local Ollama models.

    Supports models like deepseek-r1, gemma3, qwen3, etc. running locally via Ollama.
    Automatically extracts thinking content from <think> tags if present.
    """

    def __init__(self, model_name: str = "deepseek-r1:32b"):
        if ollama is None:
            raise ImportError("ollama package is not installed. Run: pip install ollama")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name, 
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        result = response["message"]["content"]
        # extract the thinking part from the response. The thinking part is surrounded by <think> and </think>
        m = re.search(r'<think>(.*?)</think>', result, re.DOTALL)
        if m:
            thinking = m.group(1)
            result = result.replace(m.group(0), "")
        else:
            thinking = ""

        return result.strip(), thinking.strip()
        

class LLMCaller:
    """Unified interface for calling multiple LLM providers with caching.

    Supports Gemini, OpenAI, DeepSeek, Grok, Moonshot, and local Ollama models.
    Results are cached in a SQLite database to avoid redundant API calls.

    Args:
        use_cache: Whether to cache and reuse previous responses.
        default_model: Default model to use when none is specified.
        api_base: Base URL for OpenAI-compatible endpoints.
        api_key_env: Environment variable name for the API key.
    """

    def __init__(
        self,
        use_cache: bool = True,
        default_model: str = "gemini-2.5-flash",
        api_base: str = None,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.use_cache = use_cache
        self.default_model = default_model
        self.api_base = api_base
        self.api_key_env = api_key_env

        self.model_factories = {
            "gemini-2.0-flash" : lambda: GeminiAPI("gemini-2.0-flash"),
            "gemini-2.5-flash-preview-04-17" : lambda: GeminiAPI("gemini-2.5-flash-preview-04-17"),
            "gemini-2.5-flash-preview-05-20" : lambda: GeminiAPI("gemini-2.5-flash-preview-05-20"),
            "gemini-2.5-pro-preview-05-06" : lambda: GeminiAPI("gemini-2.5-pro-preview-05-06"),
            "gemini-2.5-flash" : lambda: GeminiAPI("gemini-2.5-flash"),
            "gpt-5.1" : lambda: OpenAIAPI("gpt-5.1"),
            "gpt-5-mini" : lambda: OpenAIAPI("gpt-5-mini"),
            "gpt-5-nano" : lambda: OpenAIAPI("gpt-5-nano"),
            "o3" : lambda: OpenAIAPI("o3"),
            "o4-mini" : lambda: OpenAIAPI("o4-mini"),
            "deepseek-v3" : lambda: DeepSeekAPI("deepseek-chat"),
            "deepseek-r1" : lambda: DeepSeekAPI("deepseek-reasoner"),
            "grok-2" : lambda: GrokAPI(),
            "openai-api" : lambda: OpenAIAPI(
                self.default_model,
                api_base=self.api_base,
                api_key_env=self.api_key_env,
            ),
            "deepseek-r1:32b" : lambda: OllamaAPI("deepseek-r1:32b"),
            "gemma3:27b" : lambda: OllamaAPI("gemma3:27b"),
            "qwen3:32b" : lambda: OllamaAPI("qwen3:32b"),
            "qwen3:8b" : lambda: OllamaAPI("qwen3:8b"),
            "qwen3:4b" : lambda: OllamaAPI("qwen3:4b"),
            "qwen3:4b-instruct" : lambda: OllamaAPI("qwen3:4b-instruct"),
            "kimi-k2-0905-preview" : lambda: MoonShotAPI("kimi-k2-0905-preview"),
            "kimi-k2-turbo-preview" : lambda: MoonShotAPI("kimi-k2-turbo-preview"),
        }
        self.models = {}

        # Open the database with check_same_thread=False to allow cross-thread usage
        # Use threading lock for thread safety
        self.db_lock = threading.Lock()
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.params = ""

    def generate(self, prompt: str, model_family: str = None, parse_json: bool = False, max_retries: int = 3) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.generate_async(
                    prompt,
                    model_family=model_family,
                    parse_json=parse_json,
                    max_retries=max_retries,
                )
            )
        raise RuntimeError("LLMCaller.generate cannot run inside an event loop. Use generate_async instead.")

    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        model_family: str = None,
        max_tokens: int = 800,
    ) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.generate_with_images_async(
                    prompt,
                    image_paths=image_paths,
                    model_family=model_family,
                    max_tokens=max_tokens,
                )
            )
        raise RuntimeError(
            "LLMCaller.generate_with_images cannot run inside an event loop. "
            "Use generate_with_images_async instead."
        )

    async def generate_with_images_async(
        self,
        prompt: str,
        image_paths: list[str],
        model_family: str = None,
        max_tokens: int = 800,
    ) -> str:
        if model_family is None:
            model_family = self.default_model
        model = self._get_model(model_family)
        if not hasattr(model, "generate_with_images"):
            raise RuntimeError(f"Model family {model_family} does not support images.")
        result, thinking = await asyncio.to_thread(
            model.generate_with_images,
            prompt,
            image_paths,
            max_tokens,
        )
        return result, thinking
    async def generate_async(self, prompt: str, model_family: str = None, parse_json: bool = False, max_retries: int = 3) -> str:
        if model_family is None:
            model_family = self.default_model
            
        model = self._get_model(model_family)

        # Check if the result is already in the database (thread-safe)
        if self.use_cache:
            def _read_cache():
                with self.db_lock:
                    self.cursor.execute(
                        'SELECT result, thinking, parse_json FROM llm_calls WHERE prompt = ? AND model_name = ? AND params = ?',
                        (prompt, model.model_name, self.params),
                    )
                    return self.cursor.fetchone()

            result = await asyncio.to_thread(_read_cache)
            if result and parse_json == result[2]:
                # Cache hit with matching parse_json flag
                if parse_json:
                    return tolerantjson.tolerate(result[0]), result[1]
                return result[0], result[1]
            # If parse_json flag doesn't match, regenerate

        # If the result is not in the database, or we are not in deterministic mode, generate it
        last_error = None
        base_sleep_time = 1
        for attempt in range(max_retries):
            try:
                result, thinking = await asyncio.to_thread(model.generate, prompt)
                if parse_json:
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    result_json = tolerantjson.tolerate(result)
                else:
                    result = result.strip()
                thinking = thinking.strip()
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"Error generating: {e}, will try again")
                if attempt < max_retries - 1:
                    sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                    sleep_time = min(sleep_time, 30)
                    await asyncio.sleep(sleep_time)
                continue

        if last_error is not None:
            raise RuntimeError(f"LLM generation failed after {max_retries} attempts: {last_error}")

        # Save only the successful result. If the result is already in the database, it will be overwritten. (thread-safe)
        def _write_cache():
            with self.db_lock:
                self.cursor.execute(
                    'INSERT INTO llm_calls (prompt, model_name, params, result, thinking, parse_json) VALUES (?, ?, ?, ?, ?, ?)',
                    (prompt, model.model_name, self.params, result, thinking, parse_json),
                )
                self.conn.commit()

        await asyncio.to_thread(_write_cache)

        if parse_json:
            return result_json, thinking
        else:
            return result, thinking

    def _get_model(self, model_family: str):
        if model_family not in self.model_factories:
            raise KeyError(f"Unknown model family: {model_family}")
        if model_family not in self.models:
            self.models[model_family] = self.model_factories[model_family]()
        return self.models[model_family]


async def main_async():
    """Test function for LLMCaller."""
    parser = argparse.ArgumentParser(description="Test LLMCaller with a query")
    parser.add_argument("query", type=str, help="The query to send to the LLM")
    parser.add_argument("--model", type=str, help="The model to use", default="gemini-2.5-flash")
    parser.add_argument("--print-thinking", action="store_true", 
                       help="Print the thinking process if available")
    
    args = parser.parse_args()
    
    # Create an instance of LLMCaller
    caller = LLMCaller(use_cache=True, default_model=args.model)
    
    print(f"Query: {args.query}\n")
    print("Generating response...\n")
    
    # Call the generate method
    response, thinking = await caller.generate_async(args.query)
    
    # Print out the response
    print("Response:")
    print(response)
    
    # Print thinking if requested and available
    if args.print_thinking and thinking:
        print("\nThinking process:")
        print(thinking)


if __name__ == "__main__":
    asyncio.run(main_async())
