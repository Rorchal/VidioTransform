"""Plot context size per step for one trajectory under several policies.
Usage: python -m ctxcache.plot <traj file> <budget> <out.png>
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .cache import Replay
from .model import load

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
C = {"raw": "#2a78d6", "compact": "#eb6834", "slim": "#1baf7a", "ours": "#4a3aa7", "tail10": "#eda100"}
L = {"raw": "什么都不做", "compact": "到 90% 就做一次摘要（Claude Code 式）", "slim": "只做进门瘦身",
     "ours": "瘦身 + 过期 + 按活跃度淘汰（90/70）", "tail10": "只留最近 10 步"}


def main():
    f, budget, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    t = load(f)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    for p in ("raw", "tail10", "compact", "slim", "ours"):
        r = Replay(p, budget, t, hi=0.9, lo=0.7).run()
        ax.plot(range(1, len(r["sizes"]) + 1), [s / 1000 for s in r["sizes"]], color=C[p], lw=1.8,
                label=f"{L[p]}  — 累计发送 {r['sent']/1000:.0f}k，重读 {r['misses']} 次")
    ax.axhline(budget / 1000, color="#52514e", lw=1, ls="--")
    ax.annotate("预算", (1, budget / 1000 + 0.3), fontsize=8, color="#52514e")
    ax.set_xlabel("步数", fontsize=9, color="#52514e")
    ax.set_ylabel("这一步发给模型的上下文（k token）", fontsize=9, color="#52514e")
    ax.set_title(f"{t.id}：{len(t.steps)} 步，预算 {int(budget/1000)}k", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
