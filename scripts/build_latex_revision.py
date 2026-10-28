"""Convert the reviewed manuscript to an editable LaTeX project.

The Markdown is a text source; after generation, main.tex is independently editable.
"""
from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'paper/latex_20261028'
body = (ROOT / 'paper/revised_20261028/正文.md').read_text(encoding='utf-8')
old_md = (ROOT / 'paper/revised_absolute_20261025/论文完善稿.md').read_text(encoding='utf-8')
old_json = json.loads((ROOT / 'paper/revised_absolute_20261025/paper_content.json').read_text(encoding='utf-8'))

# Reference numbers follow their first appearance in the revised text.
mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 16: 8, 10: 9, 11: 10, 12: 11, 15: 12, 13: 13, 14: 14}
main_text, refs_text = body.split('## 参考文献', 1)
main_text = re.sub(r'\[(\d+)-(\d+)\]', lambda m: ''.join(f'[{i}]' for i in range(int(m.group(1)), int(m.group(2)) + 1)), main_text)
main_text = re.sub(r'\[(\d+)\]', lambda m: f'[{mapping[int(m.group(1))]}]', main_text)
references = {}
for line in refs_text.splitlines():
    m = re.match(r'\[(\d+)\]\s*(.*)', line)
    if m and int(m.group(1)) in mapping:
        references[mapping[int(m.group(1))]] = m.group(2)

def escape_text(text):
    conversion = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(conversion.get(ch, ch) for ch in text)

def inline(text):
    # Preserve math expressions, escape text, and remove Markdown emphasis/code marks.
    chunks = re.split(r'(\$[^$]+\$)', text)
    for i, chunk in enumerate(chunks):
        if not (chunk.startswith('$') and chunk.endswith('$')):
            chunk = chunk.replace('**', '').replace('`', '')
            # Preserve superscripts and Greek symbols in fonts with limited Unicode coverage.
            chunk = re.sub(r'10⁻¹⁶', r'$10^{-16}$', chunk)
            chunk = re.sub(r'10⁻⁸', r'$10^{-8}$', chunk)
            chunk = chunk.replace('μ', r'$\mu$').replace('λ', r'$\lambda$')
            parts = re.split(r'(\$[^$]+\$)', chunk)
            chunks[i] = ''.join(p if p.startswith('$') and p.endswith('$') else escape_text(p) for p in parts)
    return ''.join(chunks)

title = main_text.splitlines()[0][2:]
abstract = main_text.split('## 摘要\n', 1)[1].split('**关键词：**', 1)[0].strip()
keywords = main_text.split('**关键词：**', 1)[1].split('\n', 1)[0].strip()
section_text = main_text.split('## 1 问题分析', 1)[1]
section_text = '## 1 问题分析' + section_text

header = r'''\documentclass[UTF8,a4paper,twocolumn,10pt]{ctexart}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,geometry,hyperref,listings,xcolor,multicol}
\geometry{left=2cm,right=2cm,top=2cm,bottom=2cm}
\setlength{\columnsep}{0.7cm}
\setlength{\columnseprule}{0pt}
\setlength{\tabcolsep}{2pt}
\IfFontExistsTF{LXGW WenKai GB Lite}{\setmonofont{LXGW WenKai GB Lite}}{\setmonofont{Arial Unicode MS}}
\pagestyle{plain}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.3em}
\setlength{\emergencystretch}{3em}
\hypersetup{hidelinks}
\lstset{basicstyle=\ttfamily\tiny,breaklines=true,breakatwhitespace=false,breakautoindent=false,breakindent=0pt,columns=fixed,keepspaces=true,numbers=left,numberstyle=\tiny,frame=single,literate={_}{{\char95\allowbreak}}1}
\begin{document}
\twocolumn[{\begin{center}{\LARGE\bfseries TITLE}\end{center}\vspace{0.5em}
\begin{center}\textbf{摘要}\end{center}\begin{multicols}{2}
ABSTRACT

\noindent\textbf{关键词：} KEYWORDS
\end{multicols}\vspace{0.5em}}]
\thispagestyle{empty}
\clearpage
\tableofcontents
\thispagestyle{empty}
\clearpage
\setcounter{page}{1}
'''.replace('TITLE', inline(title)).replace('ABSTRACT', inline(abstract)).replace('KEYWORDS', inline(keywords))

def render_table(lines):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r'[:\- ]+', cell or ' ') for cell in r)]
    n = len(rows[0]); width = 0.85 / n
    # Small chunks permit tables to break naturally between the two columns.
    blocks = []
    for first in range(1, len(rows), 4):
        out = [r'\par\noindent{\scriptsize', '\\begin{tabular}{' + ''.join(f'>{{\\raggedright\\arraybackslash}}p{{{width:.3f}\\columnwidth}}' for _ in range(n)) + '}', r'\toprule']
        out.append(' & '.join(inline(c) for c in rows[0]) + r' \\')
        out.append(r'\midrule')
        for row in rows[first:first+4]:
            out.append(' & '.join(inline(c).replace(r'\_', r'\_\allowbreak{}').replace('.', r'.\allowbreak{}') for c in row) + r' \\')
        blocks.append('\n'.join(out + [r'\bottomrule', r'\end{tabular}}\par\smallskip']))
    return '\n'.join(blocks)

def render(markdown):
    lines = markdown.splitlines(); result = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            result.append(''); i += 1; continue
        if line.startswith('$$'):
            formula = line[2:]
            while '$$' not in formula:
                i += 1; formula += '\n' + lines[i]
            formula = formula.split('$$', 1)[0]
            formula = re.sub(r'\\tag\{\d+\}', '', formula)
            if r'\hat g_{\mathrm{mode}}' in formula:
                formula = r'\begin{aligned}' + formula.replace(r',\qquad', r',\\') + r'\end{aligned}'
            if r'\hat B=\arg\min_B' in formula:
                formula = formula.replace(r'\left\{', r'\{').replace(r'\right\}', r'\}')
            result.append('\\begin{equation}\n' + formula + '\n\\end{equation}')
        elif line.startswith('|'):
            table = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table.append(lines[i]); i += 1
            result.append(render_table(table)); continue
        elif line.startswith('### '):
            result.append('\\subsection{' + inline(re.sub(r'^\d+(?:\.\d+)*\s*', '', line[4:])) + '}')
        elif line.startswith('## '):
            result.append('\\section{' + inline(re.sub(r'^\d+(?:\.\d+)*\s*', '', line[3:])) + '}')
        else:
            result.append(inline(line) + '\n')
        i += 1
    return '\n'.join(result)

appendix = old_md.split('## 附录A 全部50项Top-3、结果索引与运行命令', 1)[1].split('## 附录B', 1)[0]
refs = '\n'.join(r'\noindent [' + str(k) + '] ' + inline(references[k]) + r'\par' for k in sorted(references))
sources = next(b['paths'] for b in old_json['blocks'] if b['type'] == 'source_files')
for path in ('scripts/review_experiments.py', 'src/review_diagnostics.py', 'scripts/build_latex_revision.py'):
    if path not in sources: sources.append(path)
source_dir = PROJECT / 'sources'; source_dir.mkdir(exist_ok=True)
listings = []
for number, relative in enumerate(sources, 1):
    src = ROOT / relative
    if not src.is_file(): raise FileNotFoundError(src)
    dst = source_dir / f'{number:03d}_{src.name}'
    shutil.copyfile(src, dst)
    listings.append('\\subsection*{' + inline(relative) + '}\n\\lstinputlisting{' + dst.relative_to(PROJECT).as_posix() + '}')

content = (header + render(section_text) + '\n\\section*{参考文献}\n\\addcontentsline{toc}{section}{参考文献}\n' + refs
           + '\n\\appendix\n\\section{全部50项Top-3与结果索引}\n' + render(appendix)
           + '\n\\section{完整源程序}\n' + '\n'.join(listings) + '\n\\end{document}\n')
figures = PROJECT / 'figures'; figures.mkdir(exist_ok=True)
for filename in ('ordinal_review.png', 'mixture_certificate.png'):
    shutil.copyfile(ROOT / 'outputs/review_20261028/figures' / filename, figures / filename)
content = content.replace('\\subsection{面板均分预测}', r'''\begin{figure*}[t]
\centering\includegraphics[width=0.95\textwidth]{figures/ordinal_review.png}
\caption{等级分布预测的累计距离与非零众数识别比较。}
\end{figure*}
\subsection{面板均分预测}''')
content = content.replace('\\section{验证范围与应用建议}', r'''\begin{figure*}[t]
\centering\includegraphics[width=0.95\textwidth]{figures/mixture_certificate.png}
\caption{严格混配约束的不可行证书及单项约束消融。}
\end{figure*}
\section{验证范围与应用建议}''')
(PROJECT / 'main.tex').write_text(content, encoding='utf-8')
print(f'Wrote {PROJECT / "main.tex"}; included {len(sources)} source files')
