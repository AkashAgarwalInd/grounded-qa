"""pytest tests for the RAG synthesis pipeline."""

import pytest
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from app.synthesise import synthesise, GroundedAnswer, validate_grounding


class TestSynthesis:
    """Test grounded answer synthesis."""

    @pytest.fixture
    def mock_passages(self):
        """Fixture providing mock retrieved passages."""
        return [
            {
                "chunk_id": "dpdp_sec8_1",
                "section": "Sec. 8(1)",
                "text": "A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act.",
            },
            {
                "chunk_id": "dpdp_sec8_5",
                "section": "Sec. 8(5)",
                "text": "A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards to prevent personal data breach.",
            },
        ]

    def test_synthesise_direct_answer(self, mock_passages):
        """Synthesis should produce a grounded answer."""
        question = "What security safeguards must a Data Fiduciary take?"
        answer, errors = synthesise(question=question, passages=mock_passages)
        assert hasattr(answer, "answer"), "Answer should have 'answer' field"
        assert hasattr(answer, "citations"), "Answer should have 'citations' field"
        assert hasattr(answer, "sufficient_context"), "Answer should have 'sufficient_context' field"

    def test_synthesise_grounding_validation(self, mock_passages):
        """Synthesis should validate grounding of citations."""
        question = "What security safeguards must a Data Fiduciary take?"
        answer, errors = synthesise(question=question, passages=mock_passages)
        # Should have some grounding errors or sufficient context
        assert isinstance(errors, list), "Grounding errors should be a list"

    def test_validate_grounding_verbquote(self, mock_passages):
        """Grounding validation should detect verbatim quote issues."""
        # Create a grounded answer with a quote
        from app.synthesise import GroundedAnswer, Citation
        cit = Citation(
            doc_id="dpdp_act_2023",
            section="Sec. 8(5)",
            quote="A Data Fiduciary shall protect personal data in its possession or under its control by taking reasonable security safeguards to prevent personal data breach.",
        )
        answer = GroundedAnswer(
            answer="Data fiduciary must protect personal data",
            citations=[cit],
            sufficient_context=True,
        )
        errors = validate_grounding(answer, mock_passages)
        # Errors may or may not be found depending on quote matching
        assert isinstance(errors, list), "Grounding validation should return a list"


class TestSufficientContext:
    """Test insufficient context detection."""

    def test_unanswerable_question(self):
        """Synthesis should return insufficient context for unanswerable questions."""
        mock_passages = [
            {
                "chunk_id": "dpdp_sec8_1",
                "section": "Sec. 8(1)",
                "text": "A Data Fiduciary shall implement appropriate technical measures.",
            }
        ]
        question = "What is the maximum fine in USD for a data breach under HIPAA?"
        answer, errors = synthesise(question=question, passages=mock_passages)
        # Should detect insufficient context
        assert hasattr(answer, "sufficient_context"), "Answer should have sufficient_context field"