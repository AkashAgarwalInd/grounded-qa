"""
app/synthesise.py

Generates grounded answers from retrieved RAG passages using Gemini Flash
with Pydantic-based structured outputs.
"""

import os
import pathlib
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Path to versioned system prompt
PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "synthesis_v1.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text().strip()

# Initialize Google GenAI Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# Define structured response schemas using Pydantic
class Citation(BaseModel):
    doc_id: str = Field(description="Document ID (e.g., dpdp_act_2023)")
    section: str = Field(description="Section identifier (e.g., Sec. 1)")
    quote: str = Field(description="Verbatim quote supporting the claim")


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Synthesized grounded answer to the user question")
    citations: list[Citation] = Field(description="List of supporting citations")
    sufficient_context: bool = Field(
        description="False if the provided context did not contain enough info to answer"
    )


def build_user_prompt(question: str, passages: list[dict]) -> str:
    passage_blocks = []
    for p in passages:
        p_id = p.get("chunk_id", p.get("id", "unknown"))
        section = p.get("section", "N/A")
        doc_id = p.get("doc_id", "N/A")
        text = p.get("text", "")
        passage_blocks.append(
            f"<passage id='{p_id}' doc_id='{doc_id}' section='{section}'>\n{text}\n</passage>"
        )

    formatted_passages = "\n".join(passage_blocks)

    return (
        f"<passages>\n{formatted_passages}\n</passages>\n\n"
        f"<question>{question}</question>"
    )


def synthesise(
    question: str, passages: list[dict], model: str = "gemini-3.5-flash-lite"
) -> GroundedAnswer:
    """
    Calls Gemini Flash with structured output validation returning a typed GroundedAnswer.
    """
    user_content = build_user_prompt(question, passages)

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
            tools=[],
        ),
    )

    # Automatically validated against GroundedAnswer schema by google-genai
    typed_output: GroundedAnswer = response.parsed
    return typed_output