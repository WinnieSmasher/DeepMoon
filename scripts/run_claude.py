import os
import subprocess
import sys

DEFAULT_BASE_URL = "https://9985678.xyz"


def _resolve_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")


def main() -> int:
    env = os.environ.copy()
    api_key = _resolve_api_key()

    if not api_key:
        print(
            "Missing ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in environment.",
            file=sys.stderr,
        )
        return 2

    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    env.setdefault("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)

    try:
        return subprocess.run(["claude"], env=env, check=False).returncode
    except FileNotFoundError:
        print(
            "claude command not found. Install @anthropic-ai/claude-code and ensure it is on PATH.",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
