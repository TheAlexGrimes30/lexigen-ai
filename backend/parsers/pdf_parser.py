import re
from pathlib import Path

import fitz


class PdfContractParser:

    def extract_text(self, file_path: str) -> str:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        document = fitz.open(path)

        parts: list[str] = []

        for page in document:
            text = page.get_text("text")

            if text:
                parts.append(text)

        document.close()

        raw_text = "\n".join(parts)

        return self.clean_text(raw_text)

    def clean_text(self, text: str) -> str:

        text = text.replace("\xa0", " ")
        text = text.replace("\r", "\n")

        text = re.sub(
            r"5\.\s*РЕКВИЗИТЫ И ПОДПИСИ СТОРОН.*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
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

            if re.match(r"^\d+\.\s+[А-ЯЁ\s]+$", line):
                normalized_lines.append(line)
                continue

            if re.match(r"^\d+\.\d+\.", line):
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

    def _should_merge(
        self,
        previous: str,
        current: str
    ) -> bool:

        if re.match(r"^\d+\.\s+[А-ЯЁ\s]+$", current):
            return False

        if re.match(r"^\d+\.\d+\.", current):
            return False

        if previous.endswith((".", ":", ";")):
            return False

        if previous.endswith(","):
            return True

        if current.startswith(("и ", "с ", "в ", "на ", "по ", "о ")):
            return True

        return True