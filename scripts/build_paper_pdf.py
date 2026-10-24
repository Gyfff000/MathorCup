"""AI-assisted PDF layout: OpenAI Codex / GPT-6, 2026-10-24.
Run with the bundled document runtime, not the numerical research environment.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, Image, LongTable, TableStyle, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
CONTENT = json.loads((ROOT / "paper/paper_content.json").read_text(encoding="utf-8"))
pdfmetrics.registerFont(TTFont("Song", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Hei", "C:/Windows/Fonts/simhei.ttf"))
pdfmetrics.registerFontFamily("Song", normal="Song", bold="Hei", italic="Song", boldItalic="Hei")

styles = {
    "Title": ParagraphStyle("Title", fontName="Hei", fontSize=17, leading=25, alignment=TA_CENTER, spaceAfter=15, textColor=colors.black),
    "Front": ParagraphStyle("Front", fontName="Hei", fontSize=13, leading=20, alignment=TA_CENTER, spaceBefore=6, spaceAfter=9),
    "Body": ParagraphStyle("Body", fontName="Song", fontSize=10.5, leading=17, firstLineIndent=21, alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=6),
    "Heading1": ParagraphStyle("Heading1", fontName="Hei", fontSize=13, leading=20, spaceBefore=12, spaceAfter=8, keepWithNext=True),
    "Heading2": ParagraphStyle("Heading2", fontName="Hei", fontSize=11.5, leading=18, spaceBefore=8, spaceAfter=5, keepWithNext=True),
    "Equation": ParagraphStyle("Equation", fontName="Song", fontSize=10, leading=17, alignment=TA_CENTER, wordWrap="CJK", spaceBefore=5, spaceAfter=8),
    "Caption": ParagraphStyle("Caption", fontName="Song", fontSize=9, leading=14, alignment=TA_CENTER, wordWrap="CJK", spaceBefore=5, spaceAfter=9),
    "Cell": ParagraphStyle("Cell", fontName="Song", fontSize=8.5, leading=12, wordWrap="CJK", alignment=TA_LEFT),
    "CellHeader": ParagraphStyle("CellHeader", fontName="Hei", fontSize=9, leading=13, wordWrap="CJK", alignment=TA_CENTER),
    "Reference": ParagraphStyle("Reference", fontName="Song", fontSize=9, leading=14, wordWrap="CJK", spaceAfter=6),
}


def text(s):
    return escape(str(s)).replace("B̂", "B_hat").replace("D̂", "D_hat").replace("\u0302", "_hat").replace("−", "-").replace("ŷ", "y_hat").replace("ᵀ", "<super>T</super>").replace("²", "<super>2</super>")


class PaperDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ["Heading1", "Heading2"]:
            level = 0 if flowable.style.name == "Heading1" else 1
            key = flowable.bookmark_key
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=False)
            if level == 0:
                self.notify("TOCEntry", (level, flowable.getPlainText(), self.page - 2, key))


def footer(canvas, doc):
    canvas.setAuthor("")
    canvas.setTitle(CONTENT["title"])
    canvas.setSubject("啤酒化合物感官预测与风格迁移参考研究")
    if doc.page >= 3:
        canvas.setFont("Song", 9)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page - 2))


output = ROOT / "paper/论文参考稿.pdf"
doc = PaperDoc(str(output), pagesize=A4, leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2*cm, bottomMargin=2*cm, pageCompression=1)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=footer)])
story = [Spacer(1, .4 * cm), Paragraph(text(CONTENT["title"]), styles["Title"]), Paragraph("摘 要", styles["Front"]), Paragraph(text(CONTENT["abstract"]), styles["Body"]), Spacer(1,.3*cm), Paragraph("<b>关键词：</b>" + text(CONTENT["keywords"]), styles["Body"]), PageBreak(), Paragraph("目 录", styles["Front"])]
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("TOC1", fontName="Song", fontSize=10.5, leading=16, leftIndent=0, firstLineIndent=0, spaceBefore=3), ParagraphStyle("TOC2", fontName="Song", fontSize=9.5, leading=14, leftIndent=18, firstLineIndent=0, spaceBefore=1)]
story += [toc, PageBreak()]
in_references = False
for i, block in enumerate(CONTENT["blocks"]):
    kind = block["type"]
    if kind == "heading":
        in_references = block["text"] == "参考文献"
        para = Paragraph(text(block["text"]), styles["Heading1" if block["level"] == 1 else "Heading2"])
        para.bookmark_key = "heading_%d" % i
        story.append(para)
    elif kind == "paragraph":
        story.append(Paragraph(text(block["text"]), styles["Reference" if in_references else "Body"]))
    elif kind == "equation":
        equation_path = ROOT / "paper/equations" / ("eq_%02d.png" % int(block["number"]))
        with PILImage.open(equation_path) as im:
            width, height = im.size
        ratio = min(72 / 250, (doc.width - 44) / width)
        equation_image = Image(str(equation_path), width=width*ratio, height=height*ratio)
        equation_image.hAlign = "CENTER"
        equation_table = LongTable([[equation_image, Paragraph("（%s）" % block["number"], styles["Equation"])]], colWidths=[doc.width-38,38])
        equation_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,0),(0,0),"CENTER"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
        story.append(equation_table)
    elif kind == "table":
        rows = [[Paragraph(text(c), styles["CellHeader"]) for c in block["headers"]]]
        rows += [[Paragraph(text(c), styles["Cell"]) for c in row] for row in block["rows"]]
        ratios = block["widths"] or [1/len(rows[0])] * len(rows[0])
        table = LongTable(rows, colWidths=[doc.width*x for x in ratios], repeatRows=1, hAlign="CENTER")
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.45,colors.HexColor("#D9D9D9")),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E8EEF2")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([Spacer(1,5),table,Spacer(1,9)])
    elif kind == "figure":
        path = ROOT / block["path"]
        with PILImage.open(path) as im: width,height=im.size
        max_height = (10.5 if "style_counts" in str(path) else 7) * cm
        ratio=min(doc.width/width,max_height/height)
        image = Image(str(path),width=width*ratio,height=height*ratio)
        image.hAlign="CENTER"
        story.append(KeepTogether([Spacer(1,4),image,Paragraph(text(block["caption"]), styles["Caption"])]))
doc.multiBuild(story)
print(json.dumps({"pdf":str(output),"bytes":output.stat().st_size},ensure_ascii=False))
