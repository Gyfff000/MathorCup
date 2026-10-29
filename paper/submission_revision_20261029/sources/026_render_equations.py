"""Typeset reference-paper equations; AI-assisted, OpenAI Codex, 2026-10-24."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import math_to_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/equations"
OUT.mkdir(parents=True, exist_ok=True)
EQUATIONS = [
    r"z_{ij} = \frac{\operatorname{clip}(x_{ij};a_j,b_j)-m_j}{s_j}",
    r"z_{it}=\frac{s_{it}-\mu_t}{\sigma_t},\qquad s_{it}=\mu_t+\sigma_t z_{it}",
    r"\widehat{B}=\arg\min_B\left\{\Vert Y_c-ZB\Vert_F^2+\alpha\Vert B\Vert_F^2\right\}",
    r"\operatorname{MAE}=\frac{1}{nq}\sum_{i=1}^{n}\sum_{k=1}^{q}|y_{ik}-\widehat{y}_{ik}|",
    r"r_{ik}=|y_{ik}-f_k(x_i)|,\quad r_k^*=r_{(\lceil(m+1)(1-\alpha)\rceil),k}",
    r"C_k(x)=\left[f_k(x)-r_k^*,\ f_k(x)+r_k^*\right]",
    r"W_{ij}=r_{ij}\,\mathbf{1}(|r_{ij}|\geq\tau),\quad L=\operatorname{diag}\left(\sum_j|W_{ij}|\right)-W",
    r"\widehat{B}=\arg\min_B\left\{\Vert Y_c-ZB\Vert_F^2+\alpha\Vert B\Vert_F^2+\lambda\operatorname{tr}(B^\top LB)\right\}",
    r"\pi_{jk}=\frac{1}{R}\sum_{b=1}^{R}\mathbf{1}\!\left(j\in\operatorname{Top3}_k^{(b)}\right)",
    r"\Delta_{jk}=\operatorname{MAE}_k(\widehat{Y}^{\operatorname{perm}(j)})-\operatorname{MAE}_k(\widehat{Y})",
    r"h\mid c\sim\mathcal{N}\!\left(\mu_c,w_c\Sigma_c+(1-w_c)\Sigma_{\mathrm{pool}}\right),\quad w_c=\frac{n_c}{n_c+10}",
    r"x_{\mathrm{new}}=u x_a+(1-u)x_b,\quad u\sim\operatorname{Uniform}(0.1,0.9),\quad a,b\in c",
    r"x_{\mathrm{candidate},d}\leq x_{\mathrm{source},d},\qquad \forall d\in D",
    r"f_{\mathrm{stale}}(x_{\mathrm{candidate}})\leq f_{\mathrm{stale}}(x_{\mathrm{source}})",
]
for number, expression in enumerate(EQUATIONS, 1):
    math_to_image("$" + expression + "$", OUT / f"eq_{number:02d}.png", dpi=250, format="png", color="black")
print(f"Typeset {len(EQUATIONS)} equations")
