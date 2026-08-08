"""steepness_precision_results.json을 팔레트 규칙에 맞춰 그래프로 렌더링.

5개 데이터셋을 한 축에 다 겹치면(10개 라인) 가독성이 떨어져서, small multiples(패널당
데이터셋 1개)로 분리. 비교 가능하도록 전체 패널이 같은 y축 범위(30~100%)를 공유한다.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("steepness_precision_results.json") as f:
    data = json.load(f)

results = data["results"]
depth = data["depth"]

# dataviz skill 팔레트의 categorical slot 1~5 (blue/aqua/yellow/green/violet) 순서 그대로 사용
PALETTE = {
    "iris": "#2a78d6",
    "wine": "#1baf7a",
    "breast_cancer": "#eda100",
    "digits": "#008300",
    "diabetes": "#4a3aa7",
}
LABELS = {
    "iris": "iris",
    "wine": "wine",
    "breast_cancer": "breast cancer",
    "digits": "digits",
    "diabetes": "diabetes (pima)",
}
ORDER = ["iris", "wine", "breast_cancer", "digits", "diabetes"]

fig = plt.figure(figsize=(15, 7.5), dpi=170)
fig.patch.set_facecolor("#fcfcfb")
axes = fig.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.82, bottom=0.09, left=0.05, right=0.98)
axes[1, 2].axis("off")  # 6번째 칸은 데이터셋이 5개뿐이라 비움

panel_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

for ax, name in zip(panel_axes, ORDER):
    entry = results[name]
    xs = entry["steepness_values"]
    ys = entry["soft_accuracies"]
    color = PALETTE[name]
    entry_depth = entry.get("depth", depth)
    entry_test_size = entry.get("test_size", 30)

    ax.set_facecolor("#fcfcfb")
    ax.plot(
        xs, ys,
        color=color, linewidth=2, marker="o", markersize=3,
        solid_capstyle="round", zorder=3,
    )
    hard = entry["hard_accuracy"]
    ax.axhline(
        hard, color=color, linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.6, zorder=2,
    )
    ax.axvline(8, color="#c3c2b7", linewidth=1, linestyle=":", zorder=1)

    # 직접 라벨: 범례 없이 hard tree 값과 steepness=8 값을 패널 안에 바로 표기
    ax.text(
        0.97, 0.06, f"hard tree {hard:.0%}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=color, fontweight="bold",
    )
    idx8 = xs.index(8) if 8 in xs else None
    if idx8 is not None:
        ax.annotate(
            f"{ys[idx8]:.0%}",
            xy=(8, ys[idx8]), xytext=(8.6, min(ys[idx8] + 0.06, 0.97)),
            fontsize=8, color="#52514e",
        )

    suffix_bits = []
    if entry_depth != depth:
        suffix_bits.append(f"depth={entry_depth}")
    if entry_test_size != 30:
        suffix_bits.append(f"test_n={entry_test_size}")
    suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
    ax.set_title(
        f"{LABELS[name]}{suffix}",
        fontsize=11, fontweight="bold", color="#0b0b0b", loc="left",
    )

    ax.set_xlim(1, 20)
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(range(2, 21, 4))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=8, labelbottom=True)

for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
    ax.tick_params(labelleft=True)

fig.text(0.51, 0.035, "sigmoid steepness", ha="center", fontsize=10.5, color="#52514e")
fig.text(
    0.015, 0.5, "test accuracy (plaintext, exact sigmoid)",
    va="center", rotation="vertical", fontsize=10.5, color="#52514e",
)

fig.suptitle(
    "Steepness vs accuracy: soft decision tree (exact sigmoid) vs hard decision tree",
    color="#0b0b0b", fontsize=13.5, fontweight="bold", x=0.05, y=0.975, ha="left",
)
fig.text(
    0.05, 0.925,
    f"depth={depth} (digits: depth={data.get('depth_overrides', {}).get('digits', depth)}), "
    "candidates=3/feature, features scaled to [-1, 1], test_size=30 unless noted per panel, "
    "random_state=42. dashed line = hard tree baseline, dotted vertical = steepness=8 (production default).",
    color="#898781", fontsize=9, ha="left",
)

fig.savefig("steepness_precision_plot.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved plot")
