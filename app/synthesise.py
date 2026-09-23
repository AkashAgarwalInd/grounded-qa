"""
app/synthesise.py

Generates grounded answers from retrieved RAG passages using Gemini 2.5 Flash (Free Tier).
Maintains stable system instruction prefix for prompt caching compatibility.
"""

import os
import pathlib
from google import genai
from google.genai import types

# Path to versioned system prompt
PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "synthesis_v1.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text().strip()

# Initialize Google GenAI Client (Uses free tier GEMINI_API_KEY from AI Studio)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def build_user_prompt(question: str, passages: list[dict]) -> str:
    """
    Constructs the dynamic prompt with XML-structured passages.
    System prompt remains static in client config to preserve prefix caching.
    """
    passage_blocks = []
    for p in passages:
        p_id = p.get("chunk_id", p.get("id", "unknown"))
        section = p.get("section", "N/A")
        text = p.get("text", "")
        passage_blocks.append(
            f"<passage id='{p_id}' section='{section}'>\n{text}\n</passage>"
        )

    formatted_passages = "\n".join(passage_blocks)

    return (
        f"<passages>\n{formatted_passages}\n</passages>\n\n"
        f"<question>{question}</question>"
    )

def synthesise(question: str, passages: list[dict], model: str = "gemini-3.5-flash-lite") -> str:
    """
    Calls Gemini 2.5 Flash to generate a grounded answer or return INSUFFICIENT_CONTEXT.
    """
    user_content = build_user_prompt(question, passages)

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )

    return response.text.strip()