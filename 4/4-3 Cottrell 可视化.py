import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import erfc

# 法拉第常数（电化学常数：每摩尔电子携带的电荷）
F = 96485.3329  # C/mol


def cottrell_i(t, n, A_m2, D, C_star_molm3):
    """
    计算 Cottrell 方程的电流 i(t)
    适用前提：扩散控制、平面电极、半无限扩散、阶跃电位、表面浓度近似瞬间耗尽
    公式：i(t) = n F A C* sqrt(D / (pi t))
    """
    return n * F * A_m2 * C_star_molm3 * np.sqrt(D / (np.pi * t))


def conc_profile_erfc(x_m, t, D, C_star_molm3):
    """
    计算半无限扩散在阶跃边界条件下的解析浓度剖面 C(x,t)
    适用前提：C(0,t)≈0（表面浓度耗尽），C(∞,t)=C*（体相浓度恒定）
    解析解：C(x,t)=C* erfc( x / (2 sqrt(D t)) )
    """
    z = x_m / (2.0 * np.sqrt(D * t))
    return C_star_molm3 * np.vectorize(erfc)(z)


# 配置 Streamlit 页面：标题与宽屏布局（两张并排图会更美观）
st.set_page_config(page_title="Cottrell Equation", layout="wide")

# 页面标题与导读：告诉读者三张图分别解释什么
st.title("Cottrell 方程交互可视化")
st.markdown(
    """
    本应用演示扩散控制条件下的 **Cottrell 方程**，并从“时间响应”和“空间浓度分布”两方面解释
    为什么电流随时间呈现 $t^{-1/2}$ 衰减。

    三部分图像分别对应：
    - **图 1**：电流 $i(t)$ 随时间的衰减曲线（现象）
    - **图 2**：$i$ 对 $t^{-1/2}$ 的线性判据（验证 Cottrell 行为）
    - **图 3**：解析浓度剖面 $C(x,t)$ 的演化（解释扩散层增厚、梯度变缓）
    """
)

# 侧边栏：所有可调参数集中放在这里，方便交互体验
sb = st.sidebar
sb.header("参数设置")

# 电子转移数：影响电流比例系数（整体放大/缩小）
n = sb.slider("电子转移数 n", 1, 6, 1)

# 电极面积：用 cm² 输入更贴合电化学实验习惯，后面统一换算到 m²
A_cm2 = sb.slider("电极有效面积 A (cm²)", 0.01, 10.0, 1.0)

# 扩散系数 D：用 log10(D) 滑块是为了避免 1e-12~1e-8 量级浮点 slider 难拖动的问题
logD = sb.slider("log10(D)，D 单位 m²/s", -12.0, -8.0, -10.0, 0.1)
D = 10 ** logD
sb.caption(f"当前 D = {D:.2e} m²/s")

# 体相浓度：用 mol/L 输入更符合化学配制习惯，后面换算到 mol/m³
C_molL = sb.number_input("体相浓度 C* (mol/L)", min_value=0.0, value=1e-3, format="%.3e")

# 时间范围：避免 t=0 的奇点，因此给一个最小时间 t_min
t_min, t_max = sb.slider("时间范围 t (s)", 1e-4, 100.0, (1e-2, 10.0), format="%.2e")

# 采样点数：控制曲线平滑度（点越多越平滑，但渲染稍慢）
N = sb.slider("时间采样点数", 300, 5000, 1200, 100)

# 噪声：用于模拟实验数据波动（可选，不影响理论曲线逻辑）
add_noise = sb.checkbox("加入模拟实验噪声（可选）", value=False)
noise = sb.slider("噪声强度（相对标准差）", 0.0, 0.2, 0.03, 0.01) if add_noise else 0.0

# 纵轴固定：避免 matplotlib 自动缩放导致“调参但曲线看起来差不多”的错觉
fix_axis = sb.checkbox("固定纵坐标范围（便于比较）", value=True)
update_ref = sb.button("以当前参数更新坐标参考")

# 快速数值：用少量代表点展示电流量级（可关闭）
show_quick = sb.checkbox("显示电流量级示例", value=True)

# 单位换算：把用户输入转换为计算所需 SI 单位
A_m2 = A_cm2 * 1e-4          # cm² -> m²
C_star = C_molL * 1000.0     # mol/L -> mol/m³

# 生成时间数组（log 采样）：因为 i(t) 在对数时间尺度下更容易看清前期与后期变化
t = np.logspace(np.log10(t_min), np.log10(t_max), N)

# 计算理论电流曲线 i(t)
i = cottrell_i(t, n, A_m2, D, C_star)

# 若开启噪声：在理论电流上叠加乘性噪声，并取绝对值避免出现负电流（纯演示用）
if add_noise and noise > 0:
    rng = np.random.default_rng(42)
    i = np.abs(i * (1.0 + rng.normal(0.0, noise, i.size)))

# 构造判据变量：t^{-1/2}（Cottrell 判据图的横坐标）
x_lin = t ** (-0.5)

# 对 i 与 t^{-1/2} 做线性拟合：i = a + b * t^{-1/2}
# 理想扩散控制下应当呈线性关系；截距 a 在真实实验中可能不为 0（双电层/仪器等影响）
b, a = np.polyfit(x_lin, i, 1)

# 保存“参考纵轴上限”：用于固定 y 轴范围，让调参后的变化更直观
if "ref_ymax" not in st.session_state or update_ref:
    st.session_state.ref_ymax = float(i.max()) * 1.05

# 主页面两列布局：左图 i(t)，右图 i–t^{-1/2}
col1, col2 = st.columns(2)

with col1:
    # 图 1：i(t) vs t（x 轴取对数，便于覆盖多个数量级时间）
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(t, i, lw=2.5)
    ax.set_xscale("log")
    ax.set_xlabel("Time, t (s)")
    ax.set_ylabel("Current, i(t) (A)")
    ax.set_title("Figure 1: Current response i(t)")
    if fix_axis:
        ax.set_ylim(0, st.session_state.ref_ymax)
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
    st.pyplot(fig)

with col2:
    # 图 2：i vs t^{-1/2}（线性判据图，理论上应接近直线）
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(x_lin, i, lw=2.5)
    xfit = np.linspace(x_lin.min(), x_lin.max(), 200)
    ax.plot(xfit, a + b * xfit, lw=1.8)
    ax.set_xlabel(r"$t^{-1/2}$ (s$^{-1/2}$)")
    ax.set_ylabel("Current, i(t) (A)")
    ax.set_title("Figure 2: Cottrell criterion (i vs t$^{-1/2}$)")
    if fix_axis:
        ax.set_ylim(0, st.session_state.ref_ymax)
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)
    ax.text(
        0.05, 0.95,
        "Linear check: i = a + b·t^(-1/2)\n"
        f"b = {b:.3e} A·s^(1/2)\n"
        f"a = {a:.3e} A",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7")
    )
    st.pyplot(fig)

# 公式与解释：用 st.latex 保证公式稳定渲染，避免 markdown 中转义失败
st.subheader("Cottrell 方程及其物理含义")
st.markdown("扩散控制、平面电极、半无限扩散、阶跃电位条件下：")
st.latex(r"i(t)=\frac{n F A D^{1/2} C^*}{\pi^{1/2} t^{1/2}}")

st.markdown(
    """
    该表达式给出一个关键结论：在扩散控制下，电流随时间按 $t^{-1/2}$ 衰减。
    这种衰减来自扩散层随时间增厚，导致近电极区域浓度梯度逐渐减小。
    """
)
st.latex(r"\delta(t)\sim \sqrt{Dt}")

# 第三图：用解析浓度剖面把“扩散层增厚、梯度变缓”可视化出来
st.subheader("解析浓度剖面与扩散层演化（半无限扩散）")
st.markdown("在 $C(0,t)\\approx 0,\\; C(\\infty,t)=C^*$ 的阶跃边界条件下，解析解为：")
st.latex(r"C(x,t)=C^*\,\mathrm{erfc}\!\left(\frac{x}{2\sqrt{Dt}}\right)")

st.markdown(
    """
    如何阅读图 3：
    - 随时间增大，浓度剖面向溶液内部展开，说明“扩散影响范围”随时间扩大；
    - 近电极处曲线变得更“平缓”，对应浓度梯度减小；
    - 常用 $2\\sqrt{Dt}$ 作为代表性尺度，直观标注扩散影响范围的量级（并非严格边界）。
    """
)

# 选取三个代表时间点：起点/几何中点/终点（几何中点更适合对数时间范围）
t_mid = np.sqrt(t_min * t_max)
t_list = [t_min, t_mid, t_max]

# 实时展示特征长度：让读者看到 sqrt(Dt) 与 2sqrt(Dt) 如何随参数变化
st.markdown("**特征扩散长度（实时计算，单位 µm）**")
c1, c2, c3 = st.columns(3)
for col, tj in zip((c1, c2, c3), t_list):
    delta = np.sqrt(D * tj)
    col.markdown(
        f"t = {tj:.2e} s  \n"
        f"√(Dt) = {delta*1e6:.2f} µm  \n"
        f"2√(Dt) = {2*delta*1e6:.2f} µm"
    )

# 生成空间坐标 x：取到若干倍 sqrt(D t_max)，确保三条曲线都能看到“回到体相”的过程
x_max = 6.0 * np.sqrt(D * t_max)
x = np.linspace(0.0, x_max, 600)
x_um = x * 1e6

# 绘制浓度剖面：横坐标使用 µm，避免出现过多小数与 0
fig, ax = plt.subplots(figsize=(12.8, 4.8))
for tj in t_list:
    ax.plot(x_um, conc_profile_erfc(x, tj, D, C_star), lw=2.2, label=f"t = {tj:.2e} s")

# 标注 2*sqrt(D t_max)：作为“扩散影响范围量级”的参照线，并水平标注说明
L_um = 2.0 * np.sqrt(D * t_max) * 1e6
ax.axvline(L_um, ls="--", lw=1.2)
ax.annotate(
    r"$2\sqrt{Dt}$ (characteristic diffusion length)",
    xy=(L_um, 0.6 * C_star),
    xytext=(L_um * 1.15, 0.8 * C_star),
    arrowprops=dict(arrowstyle="->", lw=1.2),
    fontsize=11
)

ax.set_xlabel("Distance from electrode, x (µm)")
ax.set_ylabel("Concentration, C(x,t) (mol/m³)")
ax.set_title("Concentration profiles under semi-infinite diffusion")
ax.grid(True, ls="--", lw=0.6, alpha=0.6)
ax.legend()
st.pyplot(fig)

# 用一句话把三张图闭环：空间演化 -> 梯度变化 -> 通量/电流时间律
st.markdown(
    """
    由图 3 可见，扩散层随时间增厚使得浓度梯度逐渐减小；扩散通量与浓度梯度成正比，
    因此电流随时间呈 $t^{-1/2}$ 衰减，这与图 1 的时间响应和图 2 的线性判据相一致。
    """
)

# 小字号的“量级示例”：让读者对电流数量级有直觉（不使用 st.metric，避免字号过大）
if show_quick:
    st.subheader("电流量级示例（用于直观理解）")
    cols = st.columns(3)
    for c, tj in zip(cols, t_list):
        ij = cottrell_i(np.array([tj]), n, A_m2, D, C_star)[0]
        c.markdown(f"t = {tj:.2e} s  \n`i = {ij:.3e} A`")
