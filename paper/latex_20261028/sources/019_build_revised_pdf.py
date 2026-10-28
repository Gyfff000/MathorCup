"""Anonymous PDF with math figures, bookmarks and complete source appendix.
AI-assisted layout, OpenAI Codex, 2026-10-25. Run with document runtime.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import hashlib
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, Image, LongTable, TableStyle, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--content',required=True); parser.add_argument('--output',required=True); args=parser.parse_args()
content=json.loads(Path(args.content).read_text(encoding='utf-8')); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
for name,path in [('Song','C:/Windows/Fonts/simsun.ttc'),('Hei','C:/Windows/Fonts/simhei.ttf'),('Code','C:/Windows/Fonts/consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,path,subfontIndex=0))
pdfmetrics.registerFontFamily('Song',normal='Song',bold='Hei',italic='Song',boldItalic='Hei')
styles={
 'Title':ParagraphStyle('Title',fontName='Hei',fontSize=17,leading=25,alignment=TA_CENTER,spaceAfter=15),
 'Front':ParagraphStyle('Front',fontName='Hei',fontSize=13,leading=20,alignment=TA_CENTER,spaceAfter=12),
 'Body':ParagraphStyle('Body',fontName='Song',fontSize=10.5,leading=17,firstLineIndent=21,alignment=TA_JUSTIFY,wordWrap='CJK',spaceAfter=6),
 'Heading1':ParagraphStyle('Heading1',fontName='Hei',fontSize=13,leading=20,spaceBefore=12,spaceAfter=8,keepWithNext=True),
 'Heading2':ParagraphStyle('Heading2',fontName='Hei',fontSize=11.5,leading=18,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'Caption':ParagraphStyle('Caption',fontName='Song',fontSize=9,leading=14,alignment=TA_CENTER,wordWrap='CJK',spaceAfter=9),
 'Cell':ParagraphStyle('Cell',fontName='Song',fontSize=8.2,leading=12,wordWrap='CJK'),
 'Header':ParagraphStyle('Header',fontName='Hei',fontSize=8.5,leading=12,wordWrap='CJK',alignment=TA_CENTER),
 'Reference':ParagraphStyle('Reference',fontName='Song',fontSize=9,leading=14,wordWrap='CJK',spaceAfter=5),
 'Code':ParagraphStyle('Code',fontName='Code',fontSize=6.8,leading=8.8,wordWrap='CJK',spaceAfter=0),
}
def txt(s):
    s=str(s).replace('−','-').replace('̂','_hat').replace('ŷ','y_hat').replace('ᵀ','^T').replace('–','-').replace('—','-')
    return escape(s).replace('²','<super>2</super>')

class Doc(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ['Heading1','Heading2']:
            level=0 if flowable.style.name=='Heading1' else 1; key=flowable.bookmark_key
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(flowable.getPlainText(),key,level=level,closed=False)
            if level==0: self.notify('TOCEntry',(0,flowable.getPlainText(),self.page-2,key))

def footer(canvas,doc):
    canvas.setAuthor(''); canvas.setTitle(content['title']); canvas.setSubject(content.get('subject','数学建模参考研究'))
    if doc.page>=3: canvas.setFont('Song',9); canvas.drawCentredString(A4[0]/2,1.2*cm,str(doc.page-2))

doc=Doc(str(output),pagesize=A4,leftMargin=2.5*cm,rightMargin=2.5*cm,topMargin=2*cm,bottomMargin=2*cm,pageCompression=1)
frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
doc.addPageTemplates([PageTemplate(id='paper',frames=[frame],onPage=footer)])
story=[Spacer(1,.4*cm),Paragraph(txt(content['title']),styles['Title']),Paragraph(content.get('front_label','摘 要'),styles['Front']),Paragraph(txt(content['abstract']),styles['Body']),Spacer(1,10),Paragraph('<b>'+txt(content.get('keyword_label','关键词'))+'：</b>'+txt(content['keywords']),styles['Body']),PageBreak(),Paragraph('目 录',styles['Front'])]
toc=TableOfContents(); toc.levelStyles=[ParagraphStyle('TOC',fontName='Song',fontSize=10,leading=15,spaceBefore=2)]; story.extend([toc,PageBreak()])

def heading(s,level,key):
    p=Paragraph(txt(s),styles['Heading1' if level==1 else 'Heading2']); p.bookmark_key=key; return p

def code_markup(s):
    # Switch fonts only to preserve characters; no source text is redacted.
    runs=[]; old=None; run=''
    for char in s:
        font='Code' if ord(char) in pdfmetrics.getFont('Code').face.charToGlyph else 'Song'
        if ord(char) not in pdfmetrics.getFont(font).face.charToGlyph:
            char='\\u%04x'%ord(char); font='Code'
        if old is not None and font!=old:
            runs.append(f'<font name="{old}">'+escape(run).replace(' ','&#160;')+'</font>'); run=''
        old=font; run+=char
    if run: runs.append(f'<font name="{old}">'+escape(run).replace(' ','&#160;')+'</font>')
    return ''.join(runs)

in_refs=False; source_index=[]
for index,b in enumerate(content['blocks']):
    kind=b['type']
    if kind=='heading':
        in_refs=b['text']=='参考文献'; story.append(heading(b['text'],b['level'],f'h_{index}'))
    elif kind=='paragraph': story.append(Paragraph(txt(b['text']),styles['Reference' if in_refs else 'Body']))
    elif kind=='equation':
        path=ROOT/content.get('equation_dir','paper/equations')/f"eq_{b.get('asset_number',b['number']):02d}.png"
        with PILImage.open(path) as im: w,h=im.size
        scale=min(72/250,(doc.width-44)/w); img=Image(str(path),width=w*scale,height=h*scale)
        table=LongTable([[img,Paragraph(f"（{b['number']}）",styles['Caption'])]],colWidths=[doc.width-38,38]); table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),8)])); story.append(table)
    elif kind=='table':
        rows=[[Paragraph(txt(c),styles['Header']) for c in b['headers']]]+[[Paragraph(txt(c),styles['Cell']) for c in row] for row in b['rows']]
        widths=b.get('widths') or [1/len(rows[0])]*len(rows[0]); table=LongTable(rows,colWidths=[doc.width*w for w in widths],repeatRows=1)
        table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D5DCE1')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('NOSPLIT',(0,0),(-1,min(2,len(rows)-1))),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story.extend([Spacer(1,4),table,Spacer(1,8)])
    elif kind=='figure':
        path=ROOT/b['path']
        with PILImage.open(path) as im: w,h=im.size
        scale=min(doc.width/w,b.get('height_cm',7)*cm/h); img=Image(str(path),width=w*scale,height=h*scale)
        story.append(KeepTogether([img,Paragraph(txt(b['caption']),styles['Caption'])]))
    elif kind=='pagebreak': story.append(PageBreak())
    elif kind=='source_files':
        for number,relative in enumerate(b['paths'],1):
            path=ROOT/relative; raw=path.read_bytes(); source=raw.decode('utf-8-sig'); source_index.append({'file':relative,'sha256':hashlib.sha256(raw).hexdigest(),'lines':len(source.splitlines())})
            story.append(heading(f'源程序 {number}　{relative}',2,f'src_{number}'))
            for lineno,line in enumerate(source.splitlines(),1):
                source_line=line.expandtabs(4)
                story.append(Paragraph(f'<font color="#777777">{lineno:04d}&#160;</font>'+code_markup(source_line or ' '),styles['Code']))
            story.append(Spacer(1,10))
doc.multiBuild(story)
if source_index: (output.parent/'source_appendix_manifest.json').write_text(json.dumps(source_index,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(output),'bytes':output.stat().st_size,'source_files':len(source_index)},ensure_ascii=False))
