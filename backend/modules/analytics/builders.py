from io import BytesIO

from docx import Document


class DocxAnalysisReportBuilder(BaseAnalysisReportBuilder):
    """Генератор DOCX-отчёта результата анализа."""

    def build(
        self,
        text: str,
    ) -> BytesIO:
        """Создаёт DOCX-файл из текста результата анализа."""
        buffer = BytesIO()

        document = Document()

        document.add_heading(
            "Результат анализа договора",
            level=1,
        )

        for line in text.splitlines():
            clean_line = line.strip()

            if clean_line:
                document.add_paragraph(clean_line)

        document.save(buffer)

        buffer.seek(0)

        return buffer
