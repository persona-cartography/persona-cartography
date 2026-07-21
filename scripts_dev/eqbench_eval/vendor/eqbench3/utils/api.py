import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List # Added List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI, Anthropic, or other).
    Supports 'test' and 'judge' configurations.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        # Load specific or default API credentials based on model_type
        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            # Judge model is used for ELO pairwise comparisons
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else: # Default/fallback
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        # Determine API provider for header/payload structure
        self.provider = "openai"  # Default
        if "anthropic.com" in self.base_url:
            self.provider = "anthropic"

        if not self.api_key:
            logging.warning(f"API Key for model_type '{self.model_type}' not found in environment variables.")
        self.headers = self._get_headers()

        logging.debug(f"Initialized {self.model_type} API client. Provider: {self.provider}, URL: {self.base_url}")

    def _get_headers(self):
        """Get headers based on API provider."""
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_anthropic_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Extract system prompt and format messages for Anthropic API."""
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Anthropic requires alternating user/assistant roles
                if not user_messages or user_messages[-1]["role"] != msg["role"]:
                    user_messages.append(msg)
                else:
                    # Merge consecutive messages of same role
                    user_messages[-1]["content"] += "\n\n" + msg["content"]
        return system_prompt, user_messages

    def _extract_anthropic_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from Anthropic API response."""
        if data.get("type") == "error":
            raise RuntimeError(f"Anthropic API Error: {data.get('error', {}).get('message')}")
        if isinstance(data.get("content"), list):
            text_block = next((block["text"] for block in data["content"] if block.get("type") == "text"), None)
            if text_block:
                return text_block
            return ""
        return data.get("completion", "")

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = 0.1) -> str:
        """
        Generic chat-completion style call using a list of messages.
        Handles retries and common errors.
        min_p is applied only if model_type is 'test' and min_p is not None.
        """
        if not self.api_key:
             raise ValueError(f"Cannot make API call for '{self.model_type}'. API Key is missing.")

        for attempt in range(self.max_retries):
            response = None # Initialize response to None for error checking
            try:
                # Build payload based on provider
                if self.provider == "anthropic":
                    system_prompt, user_messages = self._prepare_anthropic_messages(messages)
                    payload = {
                        "model": model,
                        "messages": user_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if model == "claude-opus-4-7":
                        del payload["temperature"] # temp not supported on claude opus 4.7
                    if system_prompt:
                        payload["system"] = system_prompt
                    # Disable reasoning/thinking for Anthropic models
                    payload["thinking"] = {"type": "disabled"}
                else:
                    # OpenAI format
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    # Apply min_p only for the test model if provided
                    if self.model_type == "test" and min_p is not None:
                        payload['min_p'] = min_p
                        logging.debug(f"Applying min_p={min_p} for test model call.")
                    elif self.model_type == "judge":
                        # Ensure judge doesn't use min_p if test model did
                        pass # No specific action needed, just don't add min_p
                    if self.base_url == 'https://api.openai.com/v1/chat/completions':
                        if 'min_p' in payload:
                            del payload['min_p']
                        if model == 'o3':
                            # o3 has special reqs via the openai api
                            del payload['max_tokens']
                            payload['max_completion_tokens'] = max_tokens
                            payload['temperature'] = 1
                        if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                            payload['reasoning_effort']="minimal"
                            del payload['max_tokens']
                            payload['max_completion_tokens'] = max_tokens
                            payload['temperature'] = 1

                        if model in ['gpt-5-chat-latest']:
                            del payload['max_tokens']
                            payload['max_completion_tokens'] = max_tokens
                            payload['temperature'] = 1
                    if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                        if 'qwen3' in model.lower():
                            # optionally disable thinking for qwen3 models
                            system_msg = [{"role": "system", "content": "/no_think"}]
                            payload['messages'] = system_msg + messages

                        # Disable reasoning for Anthropic models via OpenRouter
                        if model.startswith('anthropic'):
                            payload["reasoning"] = {"max_tokens": 0}

                    if model == 'openai/o3':
                        print('!! o3 low thinking')
                        payload["reasoning"] = {
                            "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                            "exclude": True #Set to true to exclude reasoning tokens from response
                        }
                    if model == 'openai/gpt-5.1':
                        print('!! gpt-5.1 low thinking')
                        payload["reasoning"] = {
                            "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                            "exclude": True #Set to true to exclude reasoning tokens from response
                        }

                    if model == 'openai/gpt-5.2':
                        print('!! gpt-5.2 none thinking')
                        payload["reasoning"] = {
                            "effort": "none", # Can be "high", "medium", or "low" (OpenAI-style)
                            "exclude": True #Set to true to exclude reasoning tokens from response
                        }

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                # Extract content based on provider
                if self.provider == "anthropic":
                    content = self._extract_anthropic_content(data)
                else:
                    if not data.get("choices") or not data["choices"][0].get("message") or "content" not in data["choices"][0]["message"]:
                         logging.warning(f"Unexpected API response structure on attempt {attempt+1}: {data}")
                         raise ValueError("Invalid response structure received from API")
                    content = data["choices"][0]["message"]["content"]

                # Optional: Strip <think> blocks if models tend to add them
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:].strip()
                if '<thinking>' in content and "</thinking>" in content:
                    post_think = content.find('</thinking>') + len("</thinking>")
                    content = content[post_think:].strip()
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_reasoning = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_reasoning:].strip()

                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries} for model {model}")
            except requests.exceptions.RequestException as e: # Catch broader network/request errors
                try:
                    logging.error(response.text)
                except:
                    pass
                logging.error(f"Request failed on attempt {attempt+1}/{self.max_retries} for model {model}: {e}")
                if response is not None:
                    logging.error(f"Response status code: {response.status_code}")
                    try:
                        logging.error(f"Response body: {response.text}")
                    except Exception:
                        logging.error("Could not read response body.")
                # Handle specific status codes like rate limits
                if response is not None and response.status_code == 429:
                    logging.warning("Rate limit exceeded. Backing off...")
                    # Implement exponential backoff or use Retry-After header if available
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue # Continue to next attempt
                elif response is not None and response.status_code >= 500:
                     logging.warning(f"Server error ({response.status_code}). Retrying...")
                else:
                    logging.warning(f"API error. Retrying...")

            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries} for model {model}.")
                 if response is not None:
                     logging.error(f"Raw response text: {response.text}")
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error during API call attempt {attempt+1}/{self.max_retries} for model {model}: {e}", exc_info=True)

            # Wait before retrying (if not a non-retryable error)
            if attempt < self.max_retries - 1:
                 time.sleep(self.retry_delay * (attempt + 1))

        # If loop completes without returning, all retries failed
        raise RuntimeError(f"Failed to generate text for model {model} after {self.max_retries} attempts")
