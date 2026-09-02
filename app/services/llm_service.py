"""
SynapseOS — services/llm_service.py
Universal Asynchronous LLM Client supporting Groq and OpenRouter.
Provides genuine medical reasoning, structured clinical JSON parsing, and automatic failover.
"""

import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


async def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    json_mode: bool = False,
    timeout: float = 25.0
) -> Optional[str]:
    """
    Asynchronously invokes Groq or OpenRouter chat completion API.
    Tries OpenRouter / Groq depending on available API keys.
    """
    # 1. Try OpenRouter if key is present
    if settings.OPENROUTER_API_KEY:
        target_model = model or settings.OPENROUTER_MODEL
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://synapseos.internal",
            "X-Title": "SynapseOS"
        }
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(OPENROUTER_ENDPOINT, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
                else:
                    logger.warning(f"OpenRouter API returned {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"OpenRouter call failed: {e}")

    # 2. Try Groq if key is present
    if settings.GROQ_API_KEY:
        target_model = model or settings.GROQ_MODEL
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(GROQ_ENDPOINT, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
                else:
                    logger.warning(f"Groq API returned {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")

    logger.debug("No external LLM API key configured or all API requests failed.")
    return None


async def call_llm_json(
    messages: List[Dict[str, str]],
    fallback_dict: Dict[str, Any],
    model: Optional[str] = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Executes an LLM request and guarantees a structured JSON dictionary output.
    """
    raw = await call_llm(messages=messages, model=model, temperature=temperature, json_mode=True)
    if not raw:
        return fallback_dict
        
    try:
        # Strip potential markdown formatting if returned
        clean_text = raw.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning(f"Error parsing LLM response as JSON: {e}")

    return fallback_dict
