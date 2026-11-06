"""
Reusable Gemini chat model factory.

The underlying ChatGoogleGenerativeAI client (and its HTTP connection) is
cached per (api_key, model, temperature) so that repeated chat requests reuse
the same client instead of constructing a new one on every call/node execution.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI


class ChatModelError(Exception):
    """Raised when the Gemini chat model cannot be initialized."""


@lru_cache(maxsize=8)
def _cached_chat_model(api_key: str, model: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)


def get_chat_model(api_key: str | None, model: str, temperature: float) -> ChatGoogleGenerativeAI:
    if not api_key:
        raise ChatModelError("GOOGLE_API_KEY is not configured. Set it in your .env file.")
    return _cached_chat_model(api_key, model, temperature)
