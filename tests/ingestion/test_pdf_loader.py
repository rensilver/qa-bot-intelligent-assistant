from unittest.mock import patch

from langchain_core.documents import Document

from qa_bot_intelligent_assistant.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir

def test_load_pdf_uses_pypdf_loader():
    fake_docs = [Document(page_content="page 1", metadata={"page": 0})]

    with patch("qa_bot_intelligent_assistant.ingestion.pdf_loader.PyPDFLoader") as mock_loader_cls, \
         patch("qa_bot_intelligent_assistant.ingestion.pdf_loader.compute_file_hash", return_value="fakehash"):
        mock_loader_cls.return_value.load.return_value = fake_docs

        result = load_pdf("some/file.pdf")

        mock_loader_cls.assert_called_once_with("some/file.pdf")
        assert result == fake_docs

def test_load_pdfs_from_dir_loads_all_pdfs_in_sorted_order(tmp_path):
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "not_a_pdf.txt").touch()

    with patch("qa_bot_intelligent_assistant.ingestion.pdf_loader.load_pdf") as mock_load_pdf:
        mock_load_pdf.side_effect = lambda path: [Document(page_content=path)]

        result = load_pdfs_from_dir(tmp_path)

        called_paths = [call.args[0] for call in mock_load_pdf.call_args_list]
        assert called_paths == [str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")]
        assert len(result) == 2

def test_load_pdfs_from_dir_with_no_pdfs_returns_empty_list(tmp_path):
    result = load_pdfs_from_dir(tmp_path)

    assert result == []
