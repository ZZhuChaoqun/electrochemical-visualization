import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================================
# Nernst Equation Interactive Visualization
# - Sidebar in Chinese (UI), Plot in English (avoid tofu blocks)
# - Explanations BELOW the plot
# - Fix "curve doesn't change" feeling: add axis-lock (default ON)
# - Comparison curves clarified and styled with dashed lines
# ==========================================================

# --------------------------
# Constants
# --------------------------
R = 8.314462618     # J/(mol·K)
F = 96485.33212     # C/mol

# --------------------------
# Defaults
# --------------------------
INIT = {
    "E0": 0.200,
    "T": 298.15,
    "n": 1,
    "xmin": -3.0,
    "xmax": 3.0,

    # teaching aids
    "compare_mode": True,
    "dT": 30.0,
    "x_ref": 0.0,

    # inverse mapping
    "E_query": 0.250,

    # axis control
    "lock_axes": True,
    "ylim_margin": 0.10,  # extra headroom when locking
}

BOUNDS = {
    "E0":   (-1.0,  1.5,  0.001),
    "T":    (250.0, 400.0, 0.5),
    "n":    (1,     4,     1),
    "xmin": (-6.0,  -0.5,  0.1),
    "xmax": (0.5,    6.0,  0.1),

    "dT":   (5.0,   80.0,  1.0),
    "x_ref": (-5.0,  5.0,  0.1),
    "E_query": (-1.0, 1.5, 0.001),
    "ylim_margin": (0.02, 0.30, 0.01),
}

# --------------------------
# Model
# --------------------------
def slope_v_per_dec(T: float, n: int) -> float:
    """Nernst slope (V/decade): 2.303 RT / (nF)"""
    return 2.303 * R * T / (n * F)

def nernst_E(E0: float, T: float, n: int, x: np.ndarray) -> np.ndarray:
    """E = E0 + slope * x, where x = log10(aOx/aRed)"""
    return E0 + slope_v_per_dec(T, n) * x

def nernst_x_from_E(E0: float, T: float, n: int, E: float) -> float:
    """Inverse mapping: x = (E - E0) / slope"""
    s = slope_v_per_dec(T, n)
    if abs(s) < 1e-15:
        return float("nan")
    return (E - E0) / s

# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Nernst Equation Interactive", layout="wide")
st.title("Nernst 方程交互可视化")

with st.sidebar:
    st.header("参数设置")

    E0 = st.slider("标准电位 E0 (V)", BOUNDS["E0"][0], BOUNDS["E0"][1], value=INIT["E0"], step=BOUNDS["E0"][2])
    T  = st.slider("温度 T (K)",      BOUNDS["T"][0],  BOUNDS["T"][1],  value=INIT["T"],  step=BOUNDS["T"][2])
    n  = st.slider("电子转移数 n",     int(BOUNDS["n"][0]), int(BOUNDS["n"][1]), value=INIT["n"], step=BOUNDS["n"][2])

    xmin = st.number_input(
        "横轴下限 xmin（log10(aOx/aRed)）",
        value=float(INIT["xmin"]),
        step=float(BOUNDS["xmin"][2]),
        min_value=float(BOUNDS["xmin"][0]),
        max_value=float(BOUNDS["xmin"][1]),
    )
    xmax = st.number_input(
        "横轴上限 xmax（log10(aOx/aRed)）",
        value=float(INIT["xmax"]),
        step=float(BOUNDS["xmax"][2]),
        min_value=float(BOUNDS["xmax"][0]),
        max_value=float(BOUNDS["xmax"][1]),
    )

    if xmax <= xmin + 0.2:
        st.warning("xmax 必须大于 xmin（至少 +0.2），已自动修正。")
        xmax = xmin + 0.2

    st.markdown("---")
    st.subheader("教学增强")

    compare_mode = st.checkbox("显示对比曲线（用于理解斜率变化）", value=INIT["compare_mode"])
    dT = st.slider("对比温度增量 ΔT (K)", BOUNDS["dT"][0], BOUNDS["dT"][1], value=INIT["dT"], step=BOUNDS["dT"][2])

    x_ref = st.slider("1 decade 标尺起点 x_ref", BOUNDS["x_ref"][0], BOUNDS["x_ref"][1], value=INIT["x_ref"], step=BOUNDS["x_ref"][2])

    st.markdown("---")
    st.subheader("坐标轴控制")
    lock_axes = st.checkbox("锁定纵轴范围（推荐开启）", value=INIT["lock_axes"])
    ylim_margin = st.slider("锁定纵轴留白（V）", BOUNDS["ylim_margin"][0], BOUNDS["ylim_margin"][1],
                            value=INIT["ylim_margin"], step=BOUNDS["ylim_margin"][2])

    st.markdown("---")
    st.subheader("反算（理解“状态函数”）")
    E_query = st.slider("给定电位 E（V）→ 反算 log10(aOx/aRed)", BOUNDS["E_query"][0], BOUNDS["E_query"][1],
                        value=INIT["E_query"], step=BOUNDS["E_query"][2])

# ==========================================================
# Compute curves
# ==========================================================
x = np.linspace(float(xmin), float(xmax), 600)
s = slope_v_per_dec(float(T), int(n))
E_base = nernst_E(float(E0), float(T), int(n), x)

# Comparison curves: change n and T (only if enabled)
curves = []
curves.append(("Base (current)", x, E_base, {"ls": "-", "lw": 2.6}))

if compare_mode:
    n_alt = int(n) + 1 if int(n) < int(BOUNDS["n"][1]) else int(n)
    T_alt = min(float(T) + float(dT), float(BOUNDS["T"][1]))

    E_n = nernst_E(float(E0), float(T), int(n_alt), x)
    E_T = nernst_E(float(E0), float(T_alt), int(n), x)

    # 用虚线/点线强化区别（你喜欢的风格）
    curves.append((f"n -> {n_alt}", x, E_n, {"ls": "--", "lw": 2.2}))
    curves.append((f"T -> {T_alt:.0f} K", x, E_T, {"ls": ":", "lw": 2.2}))

# 1-decade marker at x_ref and x_ref+1
x1 = float(x_ref)
x2 = float(x_ref) + 1.0
E1 = float(E0) + s * x1
E2 = float(E0) + s * x2
dE_dec = E2 - E1  # equals slope

# Inverse mapping: given E, solve x
x_from_E = nernst_x_from_E(float(E0), float(T), int(n), float(E_query))

# ==========================================================
# Plot (English only inside axes)
# ==========================================================
fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=140)

for label, xx, EE, style in curves:
    ax.plot(xx, EE, label=label, **style)

# E0 dashed reference line (keep it — visually helpful)
ax.axhline(float(E0), linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
ax.text(x.min(), float(E0) + 0.005, "E0 (reference)", fontsize=9, color="gray")

# 1-decade marker (mV/dec meaning)
ax.scatter([x1, x2], [E1, E2], color="black", zorder=5, s=22)
ax.annotate("", xy=(x2, E2), xytext=(x1, E1),
            arrowprops=dict(arrowstyle="<->", lw=1.2, color="black"))
ax.text((x1 + x2) / 2.0, (E1 + E2) / 2.0 + 0.03,
        f"Δx = 1 decade\nΔE = {dE_dec*1000:.1f} mV",
        ha="center", va="bottom", fontsize=10)

# Inverse query marker: show where the solved x lands at the given E
if np.isfinite(x_from_E):
    xq = np.clip(x_from_E, x.min(), x.max())
    ax.scatter([xq], [float(E_query)], color="red", zorder=6, s=30)
    ax.annotate("Given E -> solve x",
                xy=(xq, float(E_query)),
                xytext=(xq + 0.4, float(E_query) + 0.05),
                arrowprops=dict(arrowstyle="->", lw=1, color="red"),
                fontsize=10, color="red")

# Labels (English)
ax.set_title("Nernst behavior: E vs log10(aOx/aRed)", fontsize=13, pad=10)
ax.set_xlabel("log10(aOx/aRed)", fontsize=12)
ax.set_ylabel("E (V)", fontsize=12)
ax.grid(True, linestyle=":", linewidth=0.8)
ax.legend(loc="upper left", fontsize=9)

ax.set_xlim(x.min(), x.max())

# ---- key fix: axis locking so curves "visibly move" ----
if lock_axes:
    # Use a stable reference y-range computed from a fixed baseline:
    # baseline uses current E0/T/n but with fixed xmin/xmax,
    # then add margin to avoid tight cropping.
    y_min = E_base.min() - float(ylim_margin)
    y_max = E_base.max() + float(ylim_margin)
    ax.set_ylim(y_min, y_max)
else:
    # Auto range (may make curves look "unchanged" because axes chase them)
    all_E = np.concatenate([c[2] for c in curves])
    ax.set_ylim(all_E.min() - 0.08, all_E.max() + 0.10)

st.pyplot(fig, clear_figure=True)

# ==========================================================
# BELOW the plot: formula & explanation
# ==========================================================
st.markdown("---")
st.subheader("公式与读数")

colA, colB, colC = st.columns([1.15, 0.95, 1.3])

with colA:
    st.latex(r"E = E^0 + \frac{2.303RT}{nF}\log_{10}\left(\frac{a_{Ox}}{a_{Red}}\right)")

with colB:
    st.write(f"- E0 = **{E0:.3f} V**")
    st.write(f"- T  = **{T:.2f} K**")
    st.write(f"- n  = **{int(n)}**")
    st.write(f"- slope = **{s*1000:.2f} mV/dec**")

with colC:
    st.markdown("**图例解释**")
    st.write("• **Base (current)**：当前参数下的 Nernst 直线")
    st.write("• **n -> n+1（虚线）**：只改 n，用来观察斜率变小/变大")
    st.write("• **T -> T+ΔT（点线）**：只改 T，用来观察斜率随温度变化")
    st.write("说明：如果你只想看一条线，关闭“显示对比曲线”。")

with st.expander("补充说明（可折叠）", expanded=False):
    st.markdown(
        "- 横轴是 `log10(aOx/aRed)`；稀溶液常用 `a≈C`，可近似为 `log10(COx/CRed)`。\n"
        "- Nernst 方程描述的是**平衡电位（状态函数）**，不提供电流大小信息。\n"
        "- 若你觉得“曲线变化不明显”，请保持“锁定纵轴范围”开启。"
    )

# Optional: show inverse result clearly
if np.isfinite(x_from_E):
    st.info(f"反算结果：当 E = {E_query:.3f} V 时，log10(aOx/aRed) = {x_from_E:.3f}（在当前 E0/T/n 下）")
