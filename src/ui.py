"""Shared look and feel: gradient hero, glass cards and theme-aware chart styling.

All custom CSS is written to work in BOTH light and dark mode: colours are derived from the active text
colour (currentColor) or are semi-transparent, so the built-in Light/Dark switch never breaks the design.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

# Mid-tone colours that read well on both light and dark backgrounds.
BLUE, ORANGE, TEAL, VIOLET, RED, GREEN, SLATE = ("#3b82f6", "#f97316", "#14b8a6", "#8b5cf6", "#ef4444",
                                                 "#22c55e", "#64748b")
AXIS = "rgba(127,127,127,0.45)"

_CSS = """
<style>
.stApp {
  background-image:
    radial-gradient(900px 500px at -5% -10%, rgba(59,130,246,0.16), transparent 60%),
    radial-gradient(800px 460px at 105% -5%, rgba(139,92,246,0.14), transparent 60%),
    radial-gradient(900px 500px at 50% 115%, rgba(20,184,166,0.08), transparent 60%);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1500px; padding-top: 2.2rem; }
[class*="st-key-card"] {
  background: color-mix(in srgb, currentColor 4%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 11%, transparent);
  border-radius: 16px; padding: 1.1rem 1.3rem 1.2rem 1.3rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 10px 30px rgba(0,0,0,0.05);
  backdrop-filter: blur(6px);
}
.eyebrow { opacity: .65; font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  margin-bottom: .25rem; }
.muted { opacity: .72; font-size: .9rem; }
.kpi-label { opacity: .72; font-size: .9rem; }
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.2; margin: .15rem 0; }
.kpi-note { opacity: .72; font-size: .85rem; }
.good { color: #16a34a; font-weight: 600; }
.chips span { display: inline-block; background: rgba(59,130,246,0.14); color: #3b82f6; border-radius: 999px;
  padding: .15rem .7rem; margin: 0 .35rem .35rem 0; font-size: .82rem; font-weight: 500; }
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 2rem 2.2rem 1.7rem 2.2rem;
  color: #fff; margin-bottom: 1.1rem;
  background: linear-gradient(118deg, #0b1220 0%, #172554 38%, #3730a3 72%, #0891b2 118%);
  box-shadow: 0 14px 40px rgba(30,58,138,0.30); }
.hero::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(600px 260px at 85% 20%, rgba(255,255,255,0.10), transparent 70%); }
.hero .hero-eyebrow { color: rgba(255,255,255,.72); font-size: .78rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; }
.hero .hero-title { color: #fff; font-size: 2.35rem; font-weight: 750; line-height: 1.15; margin: .35rem 0 .5rem 0;
  letter-spacing: -.01em; max-width: 58%; }
.hero .hero-sub { color: rgba(255,255,255,.82); font-size: 1.02rem; max-width: 56%; line-height: 1.5; }
.hero .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem; margin-top: 1.25rem;
  position: relative; z-index: 1; max-width: 60%; }
.hero .stat { background: rgba(255,255,255,0.09); border: 1px solid rgba(255,255,255,0.18); border-radius: 14px;
  padding: .6rem .95rem; min-width: 0; backdrop-filter: blur(4px); }
.hero .stat b { display: block; font-size: 1.4rem; color: #fff; font-weight: 700; }
.hero .stat span { font-size: .72rem; color: rgba(255,255,255,.75); text-transform: uppercase; letter-spacing: .06em; }
.hero svg.art { position: absolute; right: 1.6rem; top: 50%; transform: translateY(-50%); width: 37%;
  max-width: 540px; }
@media (max-width: 1100px) { .hero svg.art { display: none; } .hero .hero-title, .hero .hero-sub,
  .hero .stats { max-width: 100%; } }
.step-num { display: inline-flex; align-items: center; justify-content: center; width: 1.8rem; height: 1.8rem;
  border-radius: 999px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: #fff; font-weight: 700;
  font-size: .9rem; margin-bottom: .4rem; }
.footer { opacity: .6; font-size: .8rem; text-align: center; margin-top: 2.2rem; }
.footer a { color: inherit; }
</style>
"""


def inject_css(extra: str = "") -> None:
    st.markdown(_CSS + (f"<style>{extra}</style>" if extra else ""), unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str, stats: list[tuple[str, str]], art: str) -> None:
    tiles = "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for v, k in stats)
    st.markdown(f"<div class='hero'>{art}<div class='hero-eyebrow'>{eyebrow}</div>"
                f"<div class='hero-title'>{title}</div><div class='hero-sub'>{subtitle}</div>"
                f"<div class='stats'>{tiles}</div></div>", unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", key: str = "") -> None:
    with st.container(key=f"card_kpi_{key}", height="stretch"):
        st.markdown(f"<div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
                    f"<div class='kpi-note'>{note}</div>", unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f"<div class='footer'>{text} · <a href='https://github.com/data-kush' target='_blank'>"
                "github.com/data-kush</a></div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340, y_title: str = "") -> go.Figure:
    """Layout only. Fonts, grid and background come from the active Streamlit theme (light or dark)."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(title=y_title, rangemode="tozero")
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def steps(items: list[tuple[str, str, str]], key: str = "s") -> None:
    """A row of numbered process steps: (material icon, title, text)."""
    cols = st.columns(len(items))
    for i, (col, (icon, title, text)) in enumerate(zip(cols, items), 1):
        with col, st.container(key=f"card_step_{key}_{i}", height="stretch"):
            st.markdown(f"<div class='step-num'>{i}</div>", unsafe_allow_html=True)
            st.markdown(f"**{icon} {title}**")
            st.caption(text)


# ---------------- hero illustrations (inline SVG, white-on-gradient) ----------------
def forecast_art() -> str:
    """Bars of past sales, a trend line, then a dashed forecast with a shaded likely range."""
    w, h, n_hist, n_fut = 520, 210, 14, 7
    base = [70, 82, 76, 95, 88, 104, 98, 116, 110, 128, 121, 138, 133, 150]
    fut = [156, 162, 158, 170, 176, 172, 184]
    step = w / (n_hist + n_fut + 1)
    y = lambda v: h - 20 - v * 0.9  # noqa: E731
    bars = "".join(f"<rect x='{step * (i + .6):.1f}' y='{y(v):.1f}' width='{step * .55:.1f}' "
                   f"height='{h - 20 - y(v):.1f}' rx='4' fill='white' fill-opacity='0.16'/>"
                   for i, v in enumerate(base))
    hist_pts = " ".join(f"{step * (i + .875):.1f},{y(v) - 6:.1f}" for i, v in enumerate(base))
    fut_x = [step * (n_hist + i + .875) for i in range(n_fut)]
    fut_pts = " ".join(f"{x:.1f},{y(v) - 6:.1f}" for x, v in zip(fut_x, fut))
    up = [f"{x:.1f},{y(v + 6 + 4 * i) - 6:.1f}" for i, (x, v) in enumerate(zip(fut_x, fut))]
    lo = [f"{x:.1f},{y(v - 6 - 4 * i) - 6:.1f}" for i, (x, v) in enumerate(zip(fut_x, fut))]
    last = f"{step * (n_hist - 1 + .875):.1f},{y(base[-1]) - 6:.1f}"
    band = " ".join([last] + up + lo[::-1])
    tx = step * (n_hist + .35)
    grid = "".join(f"<line x1='0' x2='{w}' y1='{g}' y2='{g}' stroke='white' stroke-opacity='0.08'/>"
                   for g in range(30, h - 10, 40))
    return (f"<svg class='art' viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>{grid}{bars}"
            f"<polygon points='{band}' fill='#67e8f9' fill-opacity='0.22'/>"
            f"<polyline points='{hist_pts}' fill='none' stroke='white' stroke-width='3' stroke-linejoin='round'/>"
            f"<polyline points='{last} {fut_pts}' fill='none' stroke='#67e8f9' stroke-width='3' "
            f"stroke-dasharray='7 6' stroke-linejoin='round'/>"
            f"<line x1='{tx:.1f}' x2='{tx:.1f}' y1='12' y2='{h - 20}' stroke='white' stroke-opacity='0.45' "
            f"stroke-dasharray='3 5'/><text x='{tx + 6:.1f}' y='22' fill='white' fill-opacity='0.75' "
            f"font-size='12' font-family='sans-serif'>forecast</text>"
            f"<circle cx='{fut_x[-1]:.1f}' cy='{y(fut[-1]) - 6:.1f}' r='5' fill='#67e8f9'/></svg>")


def filing_art() -> str:
    """Two overlapping report pages with highlighted added (green) and removed (red) lines, and a lens."""
    def page(x: float, y0: float, rows: list[tuple[float, str]], opacity: float) -> str:
        out = (f"<rect x='{x}' y='{y0}' width='210' height='190' rx='14' fill='white' fill-opacity='{opacity}' "
               f"stroke='white' stroke-opacity='0.35'/>"
               f"<rect x='{x + 18}' y='{y0 + 18}' width='90' height='10' rx='5' fill='white' fill-opacity='0.75'/>")
        for i, (frac, kind) in enumerate(rows):
            yy = y0 + 44 + i * 17
            color = {"g": "#34d399", "r": "#f87171"}.get(kind, "white")
            alpha = 0.85 if kind in "gr" else 0.35
            out += (f"<rect x='{x + 18}' y='{yy}' width='{174 * frac:.0f}' height='7' rx='3.5' "
                    f"fill='{color}' fill-opacity='{alpha}'/>")
            if kind == "r":
                out += (f"<line x1='{x + 14}' x2='{x + 22 + 174 * frac:.0f}' y1='{yy + 3.5}' y2='{yy + 3.5}' "
                        f"stroke='#fecaca' stroke-width='1.6'/>")
        return out
    back = page(60, 6, [(0.9, ""), (0.75, ""), (0.95, ""), (0.6, ""), (0.85, ""), (0.7, ""), (0.9, ""),
                        (0.5, "")], 0.07)
    front = page(250, 22, [(0.85, ""), (0.95, "g"), (0.7, "g"), (0.9, ""), (0.65, "r"), (0.9, ""),
                           (0.8, "g"), (0.55, "")], 0.12)
    arrow = ""
    lens = ("<circle cx='430' cy='150' r='34' fill='white' fill-opacity='0.08' stroke='white' "
            "stroke-width='5' stroke-opacity='0.9'/><line x1='455' y1='175' x2='488' y2='206' stroke='white' "
            "stroke-width='9' stroke-linecap='round' stroke-opacity='0.9'/>")
    spark = "".join(f"<circle cx='{500 - 14 * math.cos(a):.1f}' cy='{30 + 14 * math.sin(a):.1f}' r='2.2' "
                    f"fill='#67e8f9'/>" for a in [0.3, 1.2, 2.1, 3.0, 3.9, 4.8, 5.7])
    return (f"<svg class='art' viewBox='0 0 530 230' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
            f"{back}{arrow}{front}{lens}{spark}</svg>")
