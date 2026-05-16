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