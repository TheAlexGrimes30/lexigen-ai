import re


class Sectioner:

    def extract_sections(self, text: str) -> list[dict]:
        sections = []
        current = {"header": None, "level": 0, "content": []}

        for line in text.split("\n"):
            line = line.rstrip()

            if not line.strip():
                continue

            match = re.match(r'^(#{1,6})\s+(.+)', line)

            if match:
                if current["content"]:
                    sections.append(current)

                current = {
                    "header": match.group(2).strip(),
                    "level": len(match.group(1)),
                    "content": []
                }
            else:
                current["content"].append(line)

        if current["content"]:
            sections.append(current)

        return sections


class ContextInjector:

    def inject(self, header: str, text: str) -> str:
        if not header:
            return text

        return f"[{header}] {text}"


class ChunkValidator:

    def __init__(self, min_chars: int = 150, min_words: int = 25):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        text = text.strip()

        if not text:
            return False

        if len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.25:
            return False

        return True