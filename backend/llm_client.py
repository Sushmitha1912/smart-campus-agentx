import json
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_CHAT_MODEL

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _ensure_client():
    if _client is None:
        raise RuntimeError(
            "Groq client not initialized. Set GROQ_API_KEY in your .env file."
        )
    return _client


def _chat(prompt: str, system_instruction, temperature: float, json_mode: bool) -> str:
    client = _ensure_client()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def generate(prompt: str, system_instruction=None, temperature: float = 0.4) -> str:
    return _chat(prompt, system_instruction, temperature, json_mode=False)


def generate_json(prompt: str, system_instruction=None, temperature: float = 0.2) -> dict:
    json_prompt = prompt if "json" in prompt.lower() else f"{prompt}\n\nRespond with valid JSON only."
    raw = _chat(json_prompt, system_instruction, temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}