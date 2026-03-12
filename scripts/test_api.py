import os

import requests

DEFAULT_BASE_URL = "https://9985678.xyz"
DEFAULT_MODEL = "gpt-5.4"


def _api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")


def test_model() -> int:
    api_key = _api_key()
    if not api_key:
        print("Missing ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
        return 2

    base_url = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    url = f"{base_url}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": model,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Reply with OK only"}],
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return 0 if response.ok else 1
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(test_model())
