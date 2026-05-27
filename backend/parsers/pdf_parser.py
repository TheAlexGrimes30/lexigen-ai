import re
from io import BytesIO

import fitz


class PdfContractParser:
    """Извлекает текст из PDF-файла без сохранения документа на диск."""

    def extract_text_from_bytes(
        self,
        content: bytes,
    ) -> str:
        document = fitz.open(
            stream=BytesIO(content),
            filetype="pdf",
        )

        parts: list[str] = []

        for page in document:
            text = page.get_text("text")

            if text:
                parts.append(text)

        document.close()

        return self.clean_text("\n".join(parts))

    def clean_text(
        self,
        text: str,
    ) -> str:
        text = text.replace("\xa0", " ")
        text = text.replace("\r", "\n")

        text = re.sub(
            r"5\.\s*РЕКВИЗИТЫ И ПОДПИСИ СТОРОН.*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        lines = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(r"[ \t]+", " ", line)
            lines.append(line)

        normalized_lines = []

        for line in lines:
            if self._is_section_title(line):
                normalized_lines.append(line)
                continue

            if self._is_clause_start(line):
                normalized_lines.append(line)
                continue

            if not normalized_lines:
                normalized_lines.append(line)
                continue

            previous = normalized_lines[-1]

            if self._should_merge(previous, line):
                normalized_lines[-1] = previous + " " + line
            else:
                normalized_lines.append(line)

        return "\n".join(normalized_lines).strip()

    def _is_section_title(
        self,
        line: str,
    ) -> bool:
        return bool(
            re.match(r"^\d+\.\s+[А-ЯЁ\s]+$", line)
        )

    def _is_clause_start(
        self,
        line: str,
    ) -> bool:
        return bool(
            re.match(r"^\d+\.\d+\.", line)
        )

    def _should_merge(
        self,
        previous: str,
        current: str,
    ) -> bool:
        if self._is_section_title(current):
            return False

        if self._is_clause_start(current):
            return False

        if previous.endswith((".", ":", ";")):
            return False

        if previous.endswith(","):
            return True

        if current.startswith(("и ", "с ", "в ", "на ", "по ", "о ", "до ")):
            return True

        return True