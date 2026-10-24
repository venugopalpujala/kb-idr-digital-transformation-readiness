import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

INK, INK2, MUTED = "#0b0b0b", "#444441", "#898781"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
plt.rcParams.update({"font.family": "Liberation Sans"})
fig = plt.figure(figsize=(13.28, 5.31), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, fc="#ffffff", ec=MUTED, lw=1.2, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2))

ax.text(2, 37.4, "When does calibration change an AI decision? Risk-calibrated, cost-aware LLM cascade with human review",
        fontsize=14.2, weight="bold", color=INK, va="center")

# 1 query + cascade
box(2, 12, 24, 21); ax.text(14, 30.8, "1  Cascade of three models", ha="center", fontsize=11.2, weight="bold", color=INK)
for i, (t, c) in enumerate([("S  1.5B", "#cde2fb"), ("M  7B", "#9cc4f3"), ("L  32B", "#5b9be6")]):
    ax.add_patch(Rectangle((4.2 + i * 7.0, 19.0), 6.2, 7.6, fc=c, ec="white", lw=1))
    ax.text(7.3 + i * 7.0, 22.8, t, ha="center", va="center", fontsize=9.6, color=INK)
for i in range(2):
    arrow(10.5 + i * 7.0, 22.8, 11.3 + i * 7.0, 22.8)
ax.add_patch(Rectangle((4.2, 13.4), 20.2, 3.6, fc="#fdeee8", ec="none"))
ax.text(14.3, 15.2, "then a human reviewer", ha="center", va="center", fontsize=9.6, color=INK2)
ax.text(14, 18.0, "accept, escalate or hand over", ha="center", fontsize=8.8, color=MUTED)
arrow(26.3, 22.5, 30.2, 22.5)

# 2 risk signals + calibration
box(30.5, 12, 24, 21); ax.text(42.5, 30.8, "2  Risk signals, calibrated", ha="center", fontsize=11.2, weight="bold", color=INK)
for i, t in enumerate(["token probability", "stated confidence", "self-consistency (5 samples)"]):
    ax.text(42.5, 27.2 - i * 2.9, t, ha="center", fontsize=9.4, color=INK2)
box(33, 14.4, 19, 5.4, fc="#eaf2fc", ec=BLUE)
ax.text(42.5, 17.6, "Platt / isotonic / combined", ha="center", fontsize=9.6, weight="bold", color=BLUE)
ax.text(42.5, 15.5, "raw confidence: ECE 0.25 → 0.04", ha="center", fontsize=8.8, color=INK2)
arrow(54.8, 22.5, 58.7, 22.5)

# 3 decision
box(59, 12, 18.5, 21); ax.text(68.2, 30.8, "3  Expected-cost rule", ha="center", fontsize=11.2, weight="bold", color=INK)
ax.text(68.2, 26.4, "accept if  (1 − p) × cost of error\nis the cheapest option", ha="center", va="center", fontsize=9.2, color=INK2, linespacing=1.35)
ax.text(68.2, 20.6, "cost of error 25\ncost of review 5", ha="center", va="center", fontsize=9.2, color=INK, weight="bold", linespacing=1.35)
ax.text(68.2, 15.3, "re-derived for free when\ncosts change", ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.3)
arrow(77.8, 22.5, 81.7, 22.5)

# 4 guarantee
box(82, 12, 16.2, 21); ax.text(90.1, 30.8, "4  Risk control", ha="center", fontsize=11.2, weight="bold", color=INK)
ax.text(90.1, 26.0, "conformal risk control\ntarget: 5% uncaught\nerrors", ha="center", va="center", fontsize=9.2, color=INK2, linespacing=1.35)
ax.text(90.1, 19.2, "holds on average,\nonly for the workload\nit was calibrated on", ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.3)

# findings strip
ax.add_patch(Rectangle((2, 2.0), 96.2, 8.2, fc="#f6f5f2", ec="none"))
ax.text(4, 8.3, "Findings on 2,400 real queries (200 repeated splits)", fontsize=10.2, weight="bold", color=INK, va="center")
items = [("Calibrated 5.09 vs raw 7.79", "cost per query; raw confidence was worse than sending all to a human (5.00)", ORANGE),
         ("Tuned thresholds 4.98", "calibration adds no savings; stale thresholds cost more in 10 of 12 settings", BLUE),
         ("Guarantee: 4.8% (target 5%)", "27% and 47% uncaught errors on two of three workloads calibrated elsewhere", AQUA)]
for i, (a, b, col) in enumerate(items):
    x = 4 + i * 31.5
    ax.add_patch(Rectangle((x, 3.0), 0.5, 3.8, fc=col, ec="none"))
    ax.text(x + 1.2, 5.9, a, fontsize=9.8, weight="bold", color=INK, va="center")
    ax.text(x + 1.2, 3.9, b, fontsize=8.0, color=INK2, va="center")
fig.savefig("/mnt/user-data/outputs/RCDI/graphical_abstract/graphical_abstract.png", dpi=300, facecolor="white")
fig.savefig("/mnt/user-data/outputs/RCDI/graphical_abstract/graphical_abstract.pdf", facecolor="white")
from PIL import Image
im = Image.open("/mnt/user-data/outputs/RCDI/graphical_abstract/graphical_abstract.png"); print(im.size, im.size[0] / im.size[1])
im.thumbnail((1800, 1800)); im.save("/tmp/rend/ga_prev.png")
