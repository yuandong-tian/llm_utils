import random
import time
import tolerantjson
from google import genai
from openai import OpenAI
import os
import re
import argparse

# create a database to store the results of LLM calls
import sqlite3
import subprocess
import threading

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
    def __init__(self, model_name: str = "gemini-2.0-flash", use_key_rotation: bool = True):
        ENV_KEYS = ['GEMINI_API_KEY', 'GEMINI_API_KEY2', 'GEMINI_API_KEY3']
        # Randomly pick one of the API keys
        if use_key_rotation:
            env_key = random.choice(ENV_KEYS)
        else:
            env_key = 'GEMINI_API_KEY'
        api_key = get_env_var(env_key)
        if not api_key:
            raise RuntimeError(f"Missing {env_key} for Gemini API.")
        print(f"Using Gemini API key: {env_key}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        result = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        # No thinking process for Gemini
        return result.text.strip(), ""
    
    
class DeepSeekAPI:
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

class OllamaAPI:
    def __init__(self, model_name: str = "deepseek-r1:32b"):
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        import ollama
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
    def __init__(self, use_cache: bool = True, default_model: str = "gemini-2.5-flash"):
        self.use_cache = use_cache
        self.default_model = default_model

        self.model_factories = {
            "gemini-2.0-flash" : lambda: GeminiAPI("gemini-2.0-flash"),
            "gemini-2.5-flash-preview-04-17" : lambda: GeminiAPI("gemini-2.5-flash-preview-04-17"),
            "gemini-2.5-flash-preview-05-20" : lambda: GeminiAPI("gemini-2.5-flash-preview-05-20"),
            "gemini-2.5-pro-preview-05-06" : lambda: GeminiAPI("gemini-2.5-pro-preview-05-06"),
            "gemini-2.5-flash" : lambda: GeminiAPI("gemini-2.5-flash"),
            "deepseek-v3" : lambda: DeepSeekAPI("deepseek-chat"),
            "deepseek-r1" : lambda: DeepSeekAPI("deepseek-reasoner"),
            "grok-2" : lambda: GrokAPI(),
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
        if model_family is None:
            model_family = self.default_model
            
        model = self._get_model(model_family)

        # Check if the result is already in the database (thread-safe)
        if self.use_cache:
            with self.db_lock:
                self.cursor.execute('SELECT result, thinking, parse_json FROM llm_calls WHERE prompt = ? AND model_name = ? AND params = ?', (prompt, model.model_name, self.params))
                result = self.cursor.fetchone()
                if result:
                    assert parse_json == result[2], f"Parse JSON flag mismatch: {parse_json} != {result[2]}"
                    if parse_json:
                        return tolerantjson.tolerate(result[0]), result[1]
                    else:
                        return result[0], result[1]

        # If the result is not in the database, or we are not in deterministic mode, generate it
        for _ in range(max_retries):
            try:
                result, thinking = model.generate(prompt)
                if parse_json:
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    result_json = tolerantjson.tolerate(result)
                else:
                    result = result.strip()
                thinking = thinking.strip()
                break
            except Exception as e:
                print(f"Error generating: {e}, will try again in 1 second")
                time.sleep(1)
                continue

        # Save only the successful result. If the result is already in the database, it will be overwritten. (thread-safe)
        with self.db_lock:
            self.cursor.execute('INSERT INTO llm_calls (prompt, model_name, params, result, thinking, parse_json) VALUES (?, ?, ?, ?, ?, ?)', (prompt, model.model_name, self.params, result, thinking, parse_json))
            self.conn.commit()

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


def main():
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
    response, thinking = caller.generate(args.query)
    
    # Print out the response
    print("Response:")
    print(response)
    
    # Print thinking if requested and available
    if args.print_thinking and thinking:
        print("\nThinking process:")
        print(thinking)


if __name__ == "__main__":
    main()
