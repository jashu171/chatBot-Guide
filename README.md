# Autogen Chat Agent (Python) — Gemini Edition

A beginner-friendly, step-by-step guide (with copy-paste blocks) to build a **local chat agent** using **Autogen (AgentChat)** with **Google Gemini** via the OpenAI-compatible endpoint. Includes `.env`, `requirements.txt`, and a fully commented `main.py`.

---

## Goal

Build and run a **single-agent chat assistant** powered by **Gemini** that you can use in:
- **Interactive mode** (type messages in a loop).
- **One-shot mode** (pass a prompt once and exit).

---

## Prerequisites (Detailed)

1. **Python**
   - Version: **3.10 – 3.12** recommended.
   - Check your version:
     ```bash
     python --version
     ```

2. **Pip + Virtual Environment**
   - You should be able to create and activate a virtual environment.
   - On Windows, use `py -m venv venv`; on macOS/Linux, use `python -m venv venv`.

3. **Google AI Studio (Gemini) API Key**
   - Create an API key in Google AI Studio.
   - You will set it as `GOOGLE_API_KEY` in your `.env` file.

4. **Network Access**
   - The script calls Google’s **OpenAI-compatible** endpoint:
     ```
     https://generativelanguage.googleapis.com/v1beta/openai
     ```
   - Ensure outbound HTTPS is allowed.

---

## Project Structure

```
autogen-gemini-chat/
├─ .env
├─ requirements.txt
└─ main.py
```

---

## Quick Start — Step by Step

### 1) Create a project folder
```bash
mkdir autogen-gemini-chat && cd autogen-gemini-chat
```

### 2) Create and activate a virtual environment

**macOS/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -m venv venv
.env\Scripts\Activate.ps1
```

### 3) Create `.env` (copy-paste and fill your key)

> The `GOOGLE_OPENAI_COMPAT_BASE_URL` points to Google’s OpenAI-compatible API so Autogen’s OpenAI client can talk to Gemini.

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
GOOGLE_MODEL=gemini-2.0-flash
GOOGLE_OPENAI_COMPAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
SYSTEM_PROMPT=You are a concise, helpful chatbot for BotCampus.ai. Answer briefly, use plain English, and include code blocks when helpful.
TEMPERATURE=0.3
MAX_TOKENS=1024
```

### 4) Create `requirements.txt` and install

```txt
python-dotenv>=1.0.1
requests>=2.32.3
autogen-agentchat>=0.2.0
autogen-ext>=0.2.0
```

Install them:
```bash
pip install -r requirements.txt
```

### 5) Create `main.py` (fully commented, two modes: interactive & one-shot)

```python
"""
Autogen (AgentChat) + Gemini via Google's OpenAI-compatible endpoint.

Modes:
  1) Interactive chat loop:
       python main.py --interactive
  2) One-shot prompt:
       python main.py --prompt "Explain RAG in simple terms."

What this file shows:
  - Loads config from .env
  - Sets up Autogen's OpenAI-compatible model client pointed at Google Gemini
  - Creates a single AssistantAgent
  - Provides clean I/O for beginners
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# --- Load environment variables from .env ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODEL_NAME = (os.getenv("GOOGLE_MODEL") or "gemini-2.0-flash").strip()
BASE_URL = (os.getenv("GOOGLE_OPENAI_COMPAT_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (
    "You are a concise, helpful chatbot for BotCampus.ai. "
    "Answer briefly, use plain English, and include code blocks when helpful."
)

# Defaults are conservative; adjust if needed
try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
except ValueError:
    TEMPERATURE = 0.3

try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
except ValueError:
    MAX_TOKENS = 1024

# --- Basic validation for beginners ---
if not GOOGLE_API_KEY:
    raise SystemExit("ERROR: Missing GOOGLE_API_KEY in .env")

# --- Autogen imports (Agent + Model client) ---
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except Exception as e:
    raise SystemExit(
        "Autogen packages not found. Did you run 'pip install -r requirements.txt'?\n"
        f"Import error: {e}"
    )

def build_model_client():
    """
    Create an OpenAI-compatible client for Autogen but point it at Google's
    OpenAI-compatible base URL so we can use Gemini models.
    """
    return OpenAIChatCompletionClient(
        api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        base_url=BASE_URL,
    )

def build_assistant():
    """
    Create a single AssistantAgent with our chosen model client and system prompt.
    """
    model_client = build_model_client()
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
    )
    return assistant

def one_shot(prompt: str) -> str:
    """Runs one request against the assistant and returns the reply text."""
    assistant = build_assistant()
    result = assistant.run(prompt)
    return getattr(result, "content", str(result))

def interactive_loop():
    """Simple REPL. Type 'exit' or 'quit' to leave."""
    assistant = build_assistant()
    print("Interactive chat started. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = assistant.run(user_input)
        reply = getattr(result, "content", str(result))
        print(f"\nAssistant: {reply}")

def main():
    parser = argparse.ArgumentParser(
        description="Autogen + Gemini Chat Agent (OpenAI-compatible endpoint)."
    )
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive chat loop.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Run a one-shot prompt and exit.")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop()
    elif args.prompt:
        print(one_shot(args.prompt))
    else:
        print("No mode selected.\n")
        print("Examples:")
        print("  python main.py --interactive")
        print('  python main.py --prompt "Explain RAG in simple terms."')
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Clear Operation (How it works)

1. **Configuration Load** — reads `.env`.
2. **Model Client Setup** — OpenAI-compatible client pointing to Google’s endpoint.
3. **Agent Construction** — a single `AssistantAgent` with your `SYSTEM_PROMPT`.
4. **Chat Execution** — interactive loop or one-shot run.

---

## Clear Modes

**Interactive**
```bash
python main.py --interactive
```

**One-shot**
```bash
python main.py --prompt "Write a 3-point summary of Retrieval-Augmented Generation."
```

---

## Notes

- If packages are missing, run:
  ```bash
  pip install -r requirements.txt
  ```
- If auth fails, recheck `.env` values (API key, base URL, model).
- Keep `TEMPERATURE` low for precision; raise for creativity.
