"""
app/synthesise.py

Generates grounded answers from retrieved RAG passages using Gemini Flash
with Pydantic-based structured outputs and deterministic grounding validation.
"""

import os
import pathlib
import re
from difflib import SequenceMatcher
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Path to versioned system prompt
PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "synthesis_v1.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text().strip()

# Initialize Google GenAI Client - with graceful fallback if no API key
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None
    import loguru
    loguru.logger.warning("No GEMINI_API_KEY set - synthesis will be disabled")


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


def clean_text(text: str) -> str:
    """Removes section numbers, punctuation, and normalizes whitespace."""
    text = re.sub(r'^\s*(\(\d+\)|Sec\.\s*\d+|Section\s*\d+)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).lower().strip()


def validate_grounding(answer: GroundedAnswer, passages: list[dict]) -> list[str]:
    """
    Schema-valid is not the same as true.
    Verifies that cited sources exist in passages and quotes are found verbatim.
    """
    errors = []
    
    # Map (doc_id, section) to a LIST of text chunks to avoid overwriting multi-chunk sections
    by_id: dict[tuple[str, str], list[str]] = {}
    for p in passages:
        key = (p.get("doc_id", "N/A"), p.get("section", "N/A"))
        by_id.setdefault(key, []).append(p.get("text", ""))

    for cit in answer.citations:
        sources = by_id.get((cit.doc_id, cit.section))
        if not sources:
            errors.append(f"Cited non-existent source: {cit.doc_id} {cit.section}")
            continue

        clean_quote = clean_text(cit.quote)
        found = False

        for src in sources:
            clean_src = clean_text(src)

            # 1. Direct normalized substring match
            if clean_quote in clean_src:
                found = True
                break

            # 2. Fuzzy fallback match (>80% overlap ratio)
            matcher = SequenceMatcher(None, clean_quote, clean_src)
            match = matcher.find_longest_match(0, len(clean_quote), 0, len(clean_src))
            if match.size >= len(clean_quote) * 0.8:
                found = True
                break

        if not found:
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
    if client is None:
        # No API key available - return unanswerable result
        from app.synthesise import GroundedAnswer
        return (
            GroundedAnswer(
                answer="",
                citations=[],
                sufficient_context=False,
            ),
            ["No API key available - LLM synthesis disabled"],
        )

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
    grounding_errors = validate_grounding(typed_output, passages)

    return typed_output, grounding_errors