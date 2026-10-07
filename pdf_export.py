"""Turn a NeuroGPT markdown answer into a PDF (headings, paragraphs, bullets, numbered lists, tables).

    from pdf_export import markdown_to_pdf
    markdown_to_pdf(markdown_text, "demo_output/answer.pdf", "Question title")
"""
from __future__ import annotations

import os
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, ACCENT = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), colors.HexColor("#0b5cab")
FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")


def _fonts() -> tuple[str, str, str]:
    """Arial (has the symbols models like to use) when available, else the built-in Helvetica."""
    try:
        for name, file in (("Body", "arial.ttf"), ("Body-Bold", "arialbd.ttf"), ("Body-Italic", "ariali.ttf")):
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, file)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
        return "Body", "Body-Bold", "Body-Italic"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


REPLACEMENTS = {"\u2011": "-", "\u202f": " ", "\u00a0": " ", "✅": "Yes", "✔": "Yes", "❌": "No", "✖": "No", "⚠️": "(!)", "⚠": "(!)",
                "\\*": "*", "\u2192": "->", "\u2265": ">=", "\u2264": "<="}


def clean(text: str) -> str:
    text = re.sub("[✅✔]\s*(Yes)", r"\1", text)  # emoji + the word it duplicates -> just the word
    text = re.sub("[❌✖]\s*(No)", r"\1", text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"【[^】]*】", "", text)
    return re.sub("[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]", "", text)  # symbols/emoji the font cannot draw


def markdown_to_pdf(markdown: str, dst: str, title: str) -> str:
    body, bold, italic = _fonts()
    base = ParagraphStyle("base", fontName=body, fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
    styles = {
        "h1": ParagraphStyle("h1", parent=base, fontName=bold, fontSize=16, leading=20, textColor=ACCENT, spaceBefore=4, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base, fontName=bold, fontSize=13.5, leading=17, textColor=ACCENT, spaceBefore=14, spaceAfter=2, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base, fontName=bold, fontSize=11, leading=14, textColor=INK, spaceBefore=9, spaceAfter=3, keepWithNext=1),
        "h4": ParagraphStyle("h4", parent=base, fontName=bold, fontSize=9.8, leading=13, textColor=MUTED, spaceBefore=6, spaceAfter=2, keepWithNext=1),
        "summary": ParagraphStyle("summary", parent=base, backColor=colors.HexColor("#eef4fb"), borderPadding=(5, 6, 5, 6), leftIndent=6, rightIndent=6, spaceBefore=3, spaceAfter=9),
        "sub": ParagraphStyle("sub", parent=base, fontSize=8.5, textColor=MUTED, spaceAfter=3),
        "ref": ParagraphStyle("ref", parent=base, fontSize=8.2, leading=10.4, spaceAfter=2, leftIndent=16, bulletIndent=0),
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=3, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.6, leading=9.6, spaceAfter=0),
        "cellh": ParagraphStyle("cellh", parent=base, fontName=bold, fontSize=7.8, leading=9.8, spaceAfter=0, textColor=colors.white),
    }

    def inline(text: str) -> str:
        text = escape(clean(text))
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" color="#0b5cab"><u>\1</u></a>', text)      # [text](url) -> a clickable link
        return re.sub(r'(?<![">])(https?://[^\s<)]+)', r'<a href="\1" color="#0b5cab"><u>\1</u></a>', text)      # a bare address -> a clickable link

    def table(rows: list[list[str]], width: float):
        ncols = max(len(r) for r in rows)
        rows = [r + [""] * (ncols - len(r)) for r in rows]
        weights = [min(max(max(len(clean(r[c])) for r in rows), 16 if c == 0 else 15), 38) for c in range(ncols)]
        data = [[Paragraph(inline(c), styles["cellh" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[width * w / sum(weights) for w in weights], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5d0db")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    doc = SimpleDocTemplate(dst, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch,
                            bottomMargin=0.7 * inch, title=clean(title), author="NeuroGPT")
    width = letter[0] - 1.4 * inch
    lines, i = markdown.splitlines(), 0
    story = [Paragraph("NeuroGPT clinical research report", styles["sub"]), Paragraph(escape(clean(title)), styles["h1"]),
             HRFlowable(width="100%", thickness=0.8, color=ACCENT, spaceAfter=8)]
    in_summary = False
    in_refs = False
    if lines and lines[0].startswith("# "):
        i = 1
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif line.strip() in ("---", "***"):
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#c5d0db"), spaceBefore=4, spaceAfter=6))
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block if not re.match(r"^\|[\s:\-|]+\|?$", r)]
            story += [table(rows, width), Spacer(1, 8)]
        elif line.startswith(">"):       # warning banner: a shaded, bordered paragraph at the position it has in the report
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            story += [Paragraph(inline(" ".join(q for q in quote if q)), ParagraphStyle("quote", parent=base, backColor=colors.HexColor("#fffbeb"), borderColor=colors.HexColor("#b45309"),
                                                                                         borderWidth=0.8, borderPadding=6, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=10)), Spacer(1, 4)]
        elif re.match(r"^#{1,6} ", line):
            level = len(line) - len(line.lstrip("#"))
            text_h = line.lstrip("#").strip()
            story.append(CondPageBreak((1.7 if level <= 2 else 1.2) * inch))
            in_refs = text_h.lower() == "references"
            if level <= 2:
                in_summary = text_h.lower() == "clinical summary"
                story += [Paragraph(inline(text_h), styles["h2"]), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9fb6cf"), spaceAfter=4)]
            else:
                story.append(Paragraph(inline(text_h), styles["h3" if level == 3 else "h4"]))
            i += 1
        elif re.match(r"^\s*[-*] ", line):
            story.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", line)), styles["bullet"], bulletText="•"))
            i += 1
        elif re.match(r"^\d+\. ", line):
            num = re.match(r"^(\d+)\. ", line).group(1)
            story.append(Paragraph(inline(re.sub(r"^\d+\. ", "", line)), styles["ref"] if in_refs else ParagraphStyle("num", parent=styles["bullet"], leftIndent=18), bulletText=f"{num}."))
            i += 1
        else:
            para = [line.strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6} |\||>|---|\s*[-*] |\d+\. )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(para)), styles["summary"] if in_summary else base))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(body, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.7 * inch, 0.45 * inch, "NeuroGPT clinical research output - research support, not a substitute for clinical judgement.")
        canvas.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return dst
