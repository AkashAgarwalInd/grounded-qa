"""
app/run_rag.py
"""

import asyncio
from search.retrieve import retrieve_passages
from app.synthesise import synthesise


async def main():
    question = "What are the duties of a data fiduciary regarding security safeguards?"

    # 1. Await the async retrieval function
    passages = await retrieve_passages(question, top_k=5)

    # 2. Call synthesis with the resolved list of passages
    answer = synthesise(question=question, passages=passages)

    print(f"Question: {question}\n")
    print(f"Grounded Answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())