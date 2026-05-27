from pathlib import Path

from fastapi import UploadFile

from backend.parsers.docx_parser import DocxParser
from backend.parsers.pdf_parser import PdfContractParser


class ContractDocumentParser:
    """Определяет тип загруженного документа и извлекает текст из памяти."""

    def __init__(self):
        self.docx_parser = DocxParser()
        self.pdf_parser = PdfContractParser()

    async def extract_text_from_upload(
        self,
        file: UploadFile,
    ) -> str:
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()

        content = await file.read()

        if suffix == ".docx":
            return self.docx_parser.extract_text_from_bytes(content)

        if suffix == ".pdf":
            return self.pdf_parser.extract_text_from_bytes(content)

        raise ValueError("Поддерживаются только DOCX и PDF файлы")