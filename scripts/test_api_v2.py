import requests

url = "https://api.66688777.xyz/v1/messages"
api_key = "REDACTED_API_KEY"

def test_model():
    print(f"\n--- Testing Anthropic API format ---")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello! Reply with OK"}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

test_model()
