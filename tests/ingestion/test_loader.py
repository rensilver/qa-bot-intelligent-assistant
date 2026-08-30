from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from qa_bot_intelligent_assistant.ingestion.loader import load_directory, load_document

def test_load_document_dispatches_pdf_to_pdf_loader():
    fake_docs = [Document(page_content="content")]

    with patch("qa_bot_intelligent_assistant.ingestion.loader.load_pdf") as mock_load_pdf:
        mock_load_pdf.return_value = fake_docs

        result = load_document("report.PDF")

        mock_load_pdf.assert_called_once_with("report.PDF")
        assert result == fake_docs

def test_load_document_raises_for_unsupported_extension():
    with pytest.raises(ValueError, match=r"\.txt"):
        load_document("notes.txt")

def test_load_directory_delegates_to_pdf_loader():
    fake_docs = [Document(page_content="content")]

    with patch(
        "qa_bot_intelligent_assistant.ingestion.loader.load_pdfs_from_dir"
    ) as mock_load_pdfs_from_dir:
        mock_load_pdfs_from_dir.return_value = fake_docs

        result = load_directory("some/dir")

        mock_load_pdfs_from_dir.assert_called_once_with("some/dir")
        assert result == fake_docs
