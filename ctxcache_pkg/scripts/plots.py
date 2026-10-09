"""Render the two summary charts. Usage: python -I plots.py <agent_curves.npy> <locomo10.json> <out_dir>"""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"raw": "#2a78d6", "safe": "#eb6834", "path": "#1baf7a", "tail10": "#eda100", "dirty": "#4a3aa7"}
LBL = {
    "raw": "原样保留全部工具输出",
    "safe": "只删可证明过期的（同命令重跑 / 文件已被改）",
    "path": "按文件路径只留最新一次",
    "tail10": "只留最近 10 步",
    "dirty": "脏页：agent 自己的消息",
}
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def agent_plot(curves_npy, out):
    curves = np.load(curves_npy, allow_pickle=True).item()
    x = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for p in ("raw", "safe", "path", "tail10", "dirty"):
        m = curves[p].mean(axis=0)
        ax.plot(x, m, color=C[p], lw=2, label=LBL[p])
        dy = {"tail10": -7, "dirty": 7}.get(p, 0)
        ax.annotate(f"{m[-1]/1000:.1f}k", (50, m[-1]), xytext=(4, dy), textcoords="offset points", fontsize=8, color="#52514e", va="center")
    ax.set_xlabel("步数（129 条 ≥50 步的 SWE-bench 轨迹，均值）", fontsize=9, color="#52514e")
    ax.set_ylabel("保留在上下文中的 token（≈字符/4）", fontsize=9, color="#52514e")
    ax.set_title("agent 侧：不同保留策略下上下文随步数怎么涨", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(1, 55)
    fig.tight_layout()
    fig.savefig(out)


def chat_plot(locomo, out):
    d = json.load(open(locomo))
    rel = []
    for conv in d:
        c = conv["conversation"]
        sess = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        turns = [t for k in sess for t in c[k]]
        id2i = {t["dia_id"]: i for i, t in enumerate(turns)}
        cum = np.cumsum([len(t["text"]) / 4 for t in turns])
        for q in conv["qa"]:
            if q.get("category") == 5:
                continue
            for e in q.get("evidence", []):
                if e in id2i:
                    rel.append(cum[id2i[e]] / cum[-1])
    rel = np.array(rel)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    bins = np.linspace(0, 1, 11)
    h, _ = np.histogram(rel, bins=bins)
    ax.bar((bins[:-1] + bins[1:]) / 2, h / h.sum() * 100, width=0.09, color="#2a78d6")
    ax.axhline(10, color="#52514e", lw=1, ls="--")
    ax.annotate("均匀分布 = 每格 10%", (0.02, 10.4), fontsize=8, color="#52514e")
    ax.set_xlabel("证据所在位置（0 = 对话开头，1 = 对话结尾，按 token 计）", fontsize=9, color="#52514e")
    ax.set_ylabel("占全部证据轮次的 %", fontsize=9, color="#52514e")
    ax.set_title(f"聊天侧：LoCoMo 1531 个问题的 {len(rel)} 条证据落在对话哪里", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    agent_plot(sys.argv[1], sys.argv[3] + "/agent_growth.png")
    chat_plot(sys.argv[2], sys.argv[3] + "/chat_evidence_position.png")
