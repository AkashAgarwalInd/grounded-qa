"""
scripts/test_synthesis.py

Tests app/synthesise.py across 5 test cases including 1 unanswerable question.
"""

from app.synthesise import synthesise

# Dummy retrieved passages from DPDP Act 2023 / GDPR corpus
MOCK_PASSAGES = [
    {
        "chunk_id": "dpdp_sec8_1",
        "section": "Sec. 8(1)",
        "text": "A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act."
    },
    {
        "chunk_id": "dpdp_sec8_5",
        "section": "Sec. 8(5)",
        "text": "A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards to prevent personal data breach."
    },
    {
        "chunk_id": "dpdp_sec9_1",
        "section": "Sec. 9(1)",
        "text": "The Data Fiduciary shall, before processing any personal data of a child, obtain verifiable consent of the parent or lawful guardian."
    },
    {
        "chunk_id": "dpdp_sec6_1",
        "section": "Sec. 6(1)",
        "text": "The Data Principal shall have the right to obtain from the Data Fiduciary a summary of personal data being processed."
    }
]

TEST_QUESTIONS = [
    # Q1: Directly answerable
    "What security safeguards must a Data Fiduciary take?",
    # Q2: Directly answerable
    "What is required before processing personal data of a child?",
    # Q3: Directly answerable
    "What right does a Data Principal have regarding a summary of their data?",
    # Q4: Partially answerable
    "What measures must a Data Fiduciary implement for compliance?",
    # Q5: UNANSWERABLE (Not in provided passages) -> MUST return INSUFFICIENT_CONTEXT
    "What is the maximum fine in USD for a data breach under HIPAA in the United States?"
]

def run_tests():
    print("=" * 70)
    print("RUNNING RAG SYNTHESIS (v1) TESTS")
    print("=" * 70)

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Test {idx}] Question: {q}")
        print("-" * 50)
        
        # Test 5 uses empty or non-relevant passages
        passages = MOCK_PASSAGES if idx < 5 else [MOCK_PASSAGES[0]]
        
        answer = synthesise(question=q, passages=passages)
        print(f"Response:\n{answer}\n")

if __name__ == "__main__":
    run_tests()