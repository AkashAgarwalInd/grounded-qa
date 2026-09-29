"""Model routing for the Grounded QA RAG pipeline.

Classifies each query and routes to the appropriate Gemini model:
  SIMPLE      → gemini-2.5-flash-lite (Haiku-equivalent, cheap)
  MULTIHOP    → gemini-2.5-flash (Opal-equivalent, precise)
  UNANSWERABLE→ gemini-2.5-flash-lite (cheap, reject early)

Cost savings are reported alongside faithfulness delta vs all-Opus baseline.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

sys.path.insert(0, '/Users/purvigoel/repos/grounded-qa')

from app.config import settings
from app.retrieve import retrieve_passages
from app.synthesise import synthesise, GroundedAnswer
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "Classify this question as exactly one of: "
    "SIMPLE (single fact in one section), "
    "MULTIHOP (needs two or more sections), "
    "UNANSWERABLE (not covered by data-protection law). "
    "Reply with the label only."
)


async def classify_query(q: str) -> str:
    """Classify a query using Gemini Haiku (cheap)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "UNANSWERABLE"  # safe default

    client = genai.Client(api_key=api_key)

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[{"role": "user", "content": CLASSIFY_PROMPT + "\n\n" + q}],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8,
        ),
    )

    label = response.content.strip().upper()
    # Validate label
    if label in ("SIMPLE", "MULTIHOP", "UNANSWERABLE"):
        return label
    # Fallback
    return "UNANSWERABLE"


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------

def pick_model(label: str, retrieval_hits: List[Dict[str, Any]], 
               all_opus_baseline: bool = False) -> str:
    """Pick the model for generation based on classification and retrieval quality.

    Args:
        label: Classification label (SIMPLE, MULTIHOP, UNANSWERABLE)
        retrieval_hits: Top retrieved passages with scores
        all_opus_baseline: If True, always use opus regardless of label

    Returns:
        Model name string
    """
    if all_opus_baseline:
        return "gemini-2.5-flash"

    weak = not retrieval_hits or retrieval_hits[0].get("score", 1.0) < 0.45

    if label == "MULTIHOP" or weak:
        return "gemini-2.5-flash"  # precise model for hard queries
    return "gemini-2.5-flash-lite"  # cheap model for simple queries


# ---------------------------------------------------------------------------
# Routing table generator
# ---------------------------------------------------------------------------

PRICE = {
    "gemini-2.5-flash-lite": (0.75, 2.25),   # $/M input, $/M output (approx)
    "gemini-2.5-flash":   (1.50, 6.00),     # $/M input, $/M output (approx)
}


def cost(u, model: str) -> float:
    """Compute cost of a usage object for a given model.

    Args:
        u: Usage object with input_tokens, output_tokens, cache_read_input_tokens
        model: Gemini model name

    Returns:
        Cost in dollars
    """
    inp_price, out_price = PRICE[model]
    # Cache reads bill at ~10% of input token price
    cache_price = inp_price * 0.1

    return (
        u.input_tokens * inp_price
        + u.cache_read_input_tokens * cache_price
        + u.output_tokens * out_price
    ) / 1_000_000


async def routing_table(golden_test: List[Dict[str, Any]], 
                        all_opus_baseline: bool = False) -> None:
    """Print cost per 1K queries for routed vs all-Opus baseline.

    Also reports average faithfulness for each strategy.
    """
    total_routed = 0.0
    total_opus = 0.0
    faith_routed = []
    faith_opus = []

    for case in golden_test:
        question = case["question"]

        # --- Routed strategy ---
        classification = await classify_query(question)
        hits = await retrieve_passages(
            question, top_k=settings.top_k, use_bm25=settings.use_bm25,
            use_rerank=settings.use_rerank, use_rewrite=settings.use_rewrite,
        )

        model_name = pick_model(classification, hits,
                                all_opus_baseline=all_opus_baseline)
        passages = hits  # already retrieved

        # Generate answer
        ans, usage, gen_model = await synthesize_with_model(
            question, passages, model_name)

        # Cost
        c = cost(usage, model_name)
        total_routed += c

        # Faithfulness
        # Simple proxy: check if answer has citations
        faith = 1.0 if ans.citations else 0.0
        faith_routed.append(faith)

        # --- All-Opus baseline ---
        ans_opus, usage_opus, _ = await synthesize_with_model(
            question, passages, "gemini-2.5-flash")

        c_opus = cost(usage_opus, "gemini-2.5-flash")
        total_opus += c_opus

        faith_opus_val = 1.0 if ans_opus.citations else 0.0
        faith_opus.append(faith_opus_val)

    n = len(golden_test)
    print(f"\n{'Routed':9}  ${total_routed/n*1000:6.2f}/1K queries  "
          f"faithfulness={sum(faith_routed)/n:.3f}")
    print(f"{'All-Opus':9}  ${total_opus/n*1000:6.2f}/1K queries  "
          f"faithfulness={sum(faith_opus)/n:.3f}")
    print(f"\nSavings: {100*(1 - total_routed/total_opus):.1f}% cost reduction"
          f" (vs all-Opus)")
    print(f"Faithfulness delta: {sum(faith_routed)/n - sum(faith_opus)/n:+.3f}")


async def synthesize_with_model(question: str, passages: list,
                                model: str) -> tuple:
    """Synthesize an answer using the specified model, returning (answer, usage, model_name)."""
    from app.synthesise import build_user_prompt, GroundedAnswer, validate_grounding

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        from app.synthesise import GroundedAnswer
        empty = GroundedAnswer(answer="", citations=[], sufficient_context=False)
        return empty, type('obj', (object,), {'input_tokens': 0, 'output_tokens': 0,
                         'cache_read_input_tokens': 0})(), model

    client = genai.Client(api_key=api_key)

    system_prompt_path = __import__('pathlib').Path(
        '/Users/purvigoel/repos/grounded-qa/app/prompts/synthesis_v1.txt')
    system_prompt = system_prompt_path.read_text().strip()

    user_content = build_user_prompt(question, passages)

    response = await client.aio.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
        ),
    )

    typed_output: GroundedAnswer = response.parsed
    grounding_errors = validate_grounding(typed_output, passages)

    # Build a simple usage object from the response
    usage_obj = type('obj', (object,), {
        'input_tokens': response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        'output_tokens': response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        'cache_read_input_tokens': getattr(response.usage_metadata, 'cached_content_token_count', 0) if response.usage_metadata else 0,
    })()

    return typed_output, usage_obj, model


# ---------------------------------------------------------------------------
# CLI entry point for the routing table
# ---------------------------------------------------------------------------

async def main():
    """Generate routing table from golden test set."""

    # Load golden test cases from test_synthesis.py
    import pathlib
    test_path = pathlib.Path('/Users/purvigoel/repos/grounded-qa/test/test_synthesis.py')

    # Derive simple golden cases from the test file's derived cases
    # In a full implementation, these would be imported or loaded from a data file
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_synthesis", test_path)
    # Don't actually import to avoid side effects; derive sample cases instead

    # For now, use a minimal set of representative queries
    sample_cases = [
        {"question": "What must a Data Fiduciary do on a breach?"},
        {"question": "What are the penalties for non-compliance?"},
        {"question": "When is consent NOT required for processing?"},
        {"question": "What does Section 43A require?"},
        {"question": "What is the penalty for failing to notify a personal data breach?"},
    ]

    # Run both strategies
    print("=" * 70)
    print("MODEL ROUTING TABLE (Gemini)")
    print("=" * 70)

    print("\n--- Routed strategy (classification-based) ---")
    await routing_table(sample_cases, all_opus_baseline=False)

    print("\n--- All-Opus baseline ---")
    await routing_table(sample_cases, all_opus_baseline=True)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())