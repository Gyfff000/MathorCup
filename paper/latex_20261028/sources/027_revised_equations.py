"""Typeset new equations and retain the verified historical equations."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'paper/revised_20261025/equations'; out.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'paper/equations').glob('*.png'): shutil.copyfile(p,out/p.name)
expressions={15:r'w_j=\operatorname{clip}\!\left(\frac{\operatorname{mean}_i(d_{ij}e_{ij})+\eta w_0}{\operatorname{mean}_i(d_{ij}^2)+\eta},0,1\right)',
16:r'g(y)=\sum_{k=1}^{6}\mathbf{1}(y>t_k),\quad\mathcal{G}(x)=\{g(L_y(x)),\ldots,g(U_y(x))\}',
17:r'\frac{\Vert Z_{\mathrm{candidate}}-Z_{\mathrm{source}}\Vert_2}{\sqrt{p}}\leq r,\quad r\in\{0.25,0.5,1,2\}'}
for number,expression in expressions.items(): math_to_image('$'+expression+'$',out/f'eq_{number:02d}.png',dpi=250,format='png',color='black')
print('Prepared 17 math figures')
