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
