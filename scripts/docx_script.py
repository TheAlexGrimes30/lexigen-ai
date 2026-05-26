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

        return "\n".join(parts)


if __name__ == "__main__":

    parser = DocxParser()

    text = parser.extract_text(
        "filled_credit_financing_contract.docx"
    )

    print("\n" + "=" * 100)
    print("TEXT FROM DOCX")
    print("=" * 100 + "\n")

    print(text)

    print("\n" + "=" * 100)