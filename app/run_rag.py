"""
app/run_rag.py
"""

import asyncio
from search.retrieve import retrieve_passages
from app.synthesise import synthesise, GroundedAnswer


async def main():
    question = "What are the duties of a data fiduciary regarding security safeguards?"

    passages = await retrieve_passages(question, top_k=5)

    # Returns GroundedAnswer instance (Pydantic model)
    result: GroundedAnswer = synthesise(question=question, passages=passages)

    print(f"Sufficient Context: {result.sufficient_context}")
    print(f"Answer: {result.answer}\n")
    print("Citations:")
    for c in result.citations:
        print(f" - [{c.doc_id} | {c.section}]: \"{c.quote}\"")


if __name__ == "__main__":
    asyncio.run(main())