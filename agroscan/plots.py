"""Спільний стиль графіків (палітра перевірена на розрізнення при дальтонізмі)."""
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)


def confusion_plot(cm, labels_true, labels_pred, title, path, normalize=True):
    """Матриця помилок: однотонна послідовна шкала (синій), підписи — частка рядка у %."""
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(1, keepdims=True)
    val = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) if normalize else cm
    h = max(5.5, 0.42 * len(labels_true) + 2)
    w = max(6.5, 0.42 * len(labels_pred) + 3.5)
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    im = ax.imshow(val, cmap="Blues", vmin=0, vmax=1 if normalize else val.max())
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            if cm[i, j] > 0:
                pct = val[i, j] * 100
                txt = (f"{pct:.0f}" if pct >= 0.5 else "<1") if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="#ffffff" if val[i, j] > 0.55 * (1 if normalize else val.max()) else INK)
    ax.set_xticks(range(len(labels_pred)), labels_pred, rotation=60, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(len(labels_true)), labels_true, fontsize=8, color=INK2)
    ax.set_xlabel("Передбачений клас", color=INK2)
    ax.set_ylabel("Справжній клас", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% від справжнього класу" if normalize else "кількість", color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    if normalize:
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25", "50", "75", "100"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
