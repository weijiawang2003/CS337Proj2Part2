# llm_chat.py

from dotenv import load_dotenv
import os
from google import genai
from scrape import scrape_recipe
from prompt import SYSTEM_PROMPT
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")
client = genai.Client(api_key=api_key)


def ask_llm(recipe_text: str, user_message: str, history: list[str] = None) -> str:
    if history is None:
        history = []

    conversation_text = ""
    for turn in history:
        conversation_text += turn + "\n\n"

    conversation_text += (
        f"Recipe context (JSON):\n{recipe_text}\n\n"
        f"User: {user_message}\nAssistant:"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"{SYSTEM_PROMPT}\n\n{conversation_text}",
    )
    return response.text


def main():
    url = input("Paste a recipe URL: ").strip()
    recipe_text = scrape_recipe(url)

    history = []
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break

        reply = ask_llm(recipe_text, user_message, history)
        print(f"Assistant: {reply}\n")

        history.append(f"User: {user_message}")
        history.append(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
