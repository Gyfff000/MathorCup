"""Check actual revised PDFs, anonymous metadata, pagination and source appendix."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
import pdfplumber
ROOT=Path(__file__).resolve().parents[1]; PAPER=ROOT/'paper/revised_20261025'
reports=[]
for name in ['论文修订稿.pdf','AI工具使用详情说明_阶段版.pdf']:
    path=PAPER/name; reader=PdfReader(path); texts=[p.extract_text() for p in reader.pages]
    assert reader.metadata.get('/Author','')=='' and '目 录' in texts[1]
    assert texts[2].lstrip().startswith('1\n1 ')
    assert all('\x00' not in t for t in texts)
    assert not any(re.search(r'[A-Za-z0-9_.+-]+@qq\.com',t) for t in texts)
    assert all(abs(float(p.mediabox.width)-595.2756)<.1 for p in reader.pages)
    headings=[p for p in reader.outline if not isinstance(p,list)]
    appendix=next((reader.get_destination_page_number(p)+1 for p in headings if p.title.startswith('附录A')),None)
    overflow=[]
    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages):
            # CJK hanging punctuation includes a wide side-bearing in its advance box.
            bad=[c for c in page.chars if c.get('text','').strip() and (c['x0'] < 67 or c['x1'] > float(page.width)-(60 if c.get('text') in '。，、；：' else 67) or c['top']<43 or c['bottom']>float(page.height)-25)]
            if bad: overflow.append({'page':i+1,'count':len(bad),'sample':''.join(c['text'] for c in bad[:60])})
    assert not overflow,overflow
    reports.append({'file':name,'physical_pages':len(reader.pages),'body_starts_page':3,'appendix_A_starts_physical_page':appendix,'missing_glyph_nulls':0,'text_overflow_pages':0,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
content=json.loads((PAPER/'paper_content.json').read_text(encoding='utf-8')); numbers=[b['number'] for b in content['blocks'] if b['type']=='equation']
assert numbers==list(range(1,len(numbers)+1))
manifest=json.loads((PAPER/'source_appendix_manifest.json').read_text(encoding='utf-8'))
paths=next(b['paths'] for b in content['blocks'] if b['type']=='source_files')
assert {p['file'] for p in manifest}==set(paths)
for row in manifest: assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
report={'client_date':'2026-10-25','documents':reports,'numbered_equations':len(numbers),'source_files':len(manifest),'source_lines':sum(r['lines'] for r in manifest),
        'source_appendix_matches_current_files':True,'visual_review':'rendered_pages_must_be_inspected_before_delivery'}
(PAPER/'pdf_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
