from pathlib import Path

from backend.parsers.docx_parser import DocxParser
from backend.parsers.pdf_parser import PdfContractParser


class ContractDocumentParser:

    def __init__(self):
        self.docx_parser = DocxParser()
        self.pdf_parser = PdfContractParser()

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)

        suffix = path.suffix.lower()

        if suffix == ".docx":
            return self.docx_parser.extract_text(file_path)

        if suffix == ".pdf":
            return self.pdf_parser.extract_text(file_path)

        raise ValueError("Поддерживаются только DOCX и PDF файлы")