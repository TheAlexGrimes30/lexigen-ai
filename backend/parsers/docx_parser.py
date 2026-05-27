import re

from docx import Document


class DocxParser:

    def extract_text(
        self,
        file_path: str
    ) -> str:

        document = Document(file_path)

        parts = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                parts.append(text)

        text = "\n".join(parts)

        return self.clean_text(text)

    def clean_text(
        self,
        text: str
    ) -> str:

        text = re.sub(
            r"5\. РЕКВИЗИТЫ И ПОДПИСИ СТОРОН.*",
            "",
            text,
            flags=re.DOTALL
        )

        text = text.replace("\xa0", " ")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()