from io import BytesIO

from docx import Document
from docx.text.paragraph import Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


class AnalysisReportService:
    """Генерирует DOCX/PDF отчёты из текста результата анализа без хранения файла на диске."""

    def build_docx(self, text: str) -> BytesIO:
        buffer = BytesIO()
        document = Document()
        document.add_heading("Результат анализа договора", level=1)
        for line in text.splitlines():
            clean_line = line.strip()
            if clean_line:
                document.add_paragraph(clean_line)
        document.save(buffer)
        buffer.seek(0)
        return buffer

    def build_pdf(self, text: str) -> BytesIO:
        buffer = BytesIO()
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        pdf = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        title_style.fontName = "HYSMyeongJo-Medium"
        body_style = styles["BodyText"]
        body_style.fontName = "HYSMyeongJo-Medium"
        body_style.fontSize = 11
        body_style.leading = 16
        story = [Paragraph("Результат анализа договора", title_style), Spacer(1, 12)]
        for line in text.splitlines():
            clean_line = line.strip()
            if clean_line:
                safe = clean_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))
                story.append(Spacer(1, 6))
        pdf.build(story)
        buffer.seek(0)
        return buffer


analysis_report_service = AnalysisReportService()