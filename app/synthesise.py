"""
app/synthesise.py
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


def validate_grounding(answer: GroundedAnswer, passages: list[dict]) -> list[str]:
    """
    Schema-valid is not the same as true.
    
    Verifies that every cited source exists in the input passages and 
    that the extracted quote exists verbatim in the retrieved text.
    """
    errors = []
    # Build lookup table using passage metadata keys
    by_id = {(p.get("doc_id"), p.get("section")): p.get("text", "") for p in passages}

    for cit in answer.citations:
        src = by_id.get((cit.doc_id, cit.section))
        if src is None:
            errors.append(f"Cited non-existent source: {cit.doc_id} {cit.section}")
        elif cit.quote.strip() not in src:
            errors.append(
                f"Quote not found verbatim in {cit.doc_id} {cit.section}: '{cit.quote}'"
            )

    return errors


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
) -> tuple[GroundedAnswer, list[str]]:
    """
    Calls Gemini Flash to generate a typed answer and validates grounding.
    Returns a tuple of (GroundedAnswer, list_of_grounding_errors).
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

    typed_output: GroundedAnswer = response.parsed
    
    # Deterministic hallucination check
    grounding_errors = validate_grounding(typed_output, passages)

    return typed_output, grounding_errors