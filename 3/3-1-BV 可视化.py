import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 关键设置：避免 Unicode 负号（U+2212）在某些字体下缺字，导致 glyph 8722 警告
plt.rcParams["axes.unicode_minus"] = False

# 常量：气体常数与法拉第常数
R = 8.314  # J/(mol·K)
F = 96485.3329  # C/mol


def tafel_slope(alpha: float, n: int, T: float) -> float:
    """
    计算 Tafel 斜率（单位：V/dec）
    b = 2.303 RT / (alpha n F)
    """
    return (2.303 * R * T) / (alpha * n * F)


def butler_volmer(eta, i0, alpha_a, alpha_c, n, T, ilim):
    """
    Butler–Volmer 方程（含可选“阴极限流”软限制）
    约定（与代码绘图一致）：
      i_a = i0 * exp( alpha_a nF η / RT )
      i_c = i0 * exp( -alpha_c nF η / RT )     # 这里把 i_c 定义为“阴极电流幅值”（始终为正）
      i = i_a - i_c                             # 总电流（阴极方向为负）

    若启用 i_lim（仅用于演示传质限制趋势）：
      i_c_eff = i_c / (1 + i_c / i_lim)
    """
    ia = i0 * np.exp(alpha_a * n * F * eta / (R * T))
    ic = i0 * np.exp(-alpha_c * n * F * eta / (R * T))

    # 可选：用软限制模拟“极限电流/传质限制”对阴极支路的截断效应
    if np.isfinite(ilim):
        ic = ic / (1.0 + ic / ilim)

    i = ia - ic
    return i, ia, ic


def main():
    # 页面基础设置：宽屏布局，便于两张图并排
    st.set_page_config(page_title="Butler–Volmer", layout="wide")
    st.title("Butler–Volmer 方程交互可视化")

    # 页面说明：中文解释用于教学；图内文字统一英文避免中文字体方块
    st.markdown(
        """
        本应用演示 **Butler–Volmer（BV）方程**：电流密度与过电位之间的非线性关系，并展示其在
        **线性坐标**与**对数坐标**下的典型形态。

        - **图 1（线性）**：总电流与阳极/阴极分量的整体关系  
        - **图 2（对数）**：用 \\(|i|\\) 展示对数坐标下的行为（避免负值无法取对数）
        """
    )

    # 侧边栏：交互参数集中管理
    sb = st.sidebar
    sb.header("参数设置")

    # 交换电流密度 i0：决定动力学“整体尺度”，i0 越大，曲线整体电流越大
    i0 = sb.slider("交换电流密度 i0 (A/m²)", 0.01, 100.0, 1.0)

    # 温度与电子数：影响指数项敏感度与 Tafel 斜率
    T = sb.slider("温度 T (K)", 200, 1000, 300)
    n = sb.slider("电子转移数 n", 1, 5, 1)

    # 传递系数：决定正/负支路的陡峭程度与不对称性
    alpha_a = sb.slider("阳极传递系数 αa", 0.10, 1.00, 0.50)
    sb.caption(f"阳极 Tafel 斜率 b_a = {tafel_slope(alpha_a, n, T):.3f} V/dec")

    alpha_c = sb.slider("阴极传递系数 αc", 0.10, 1.00, 0.50)
    sb.caption(f"阴极 Tafel 斜率 b_c = {tafel_slope(alpha_c, n, T):.3f} V/dec")

    # 过电位范围：控制展示窗口
    eta_min, eta_max = sb.slider("过电位范围 η (V)", -1.0, 1.0, (-0.25, 0.25), 0.01)

    # 可选：限制电流密度（演示传质限制导致的平台趋势）
    use_ilim = sb.checkbox("启用限制电流密度 i_lim（可选）", value=False)
    ilim = sb.slider("i_lim (A/m²)", 0.01, 10.0, 1.0) if use_ilim else np.inf

    # 固定坐标轴：避免自动缩放让“调参变化”不直观
    sb.divider()
    fix_axis = sb.checkbox("固定坐标轴范围（便于比较）", value=True)
    update_ref = sb.button("以当前参数更新坐标参考")

    # 计算：生成 η 采样点并计算 BV 电流
    eta = np.linspace(eta_min, eta_max, 1200)
    i, ia, ic = butler_volmer(eta, i0, alpha_a, alpha_c, n, T, ilim)

    # 对数图准备：log 坐标不能画负值，因此使用幅值 |i|
    abs_i = np.abs(i)
    abs_i_pos = abs_i[abs_i > 0]
    if abs_i_pos.size == 0:
        abs_i_pos = np.array([1e-30])

    # 参考坐标轴（用于固定显示范围，让调参更“看得出来”）
    if "ref_lin_xlim" not in st.session_state or update_ref:
        xlim = float(max(np.max(np.abs(ia)), np.max(np.abs(ic)), np.max(np.abs(i))) * 1.10)
        st.session_state.ref_lin_xlim = (-xlim, xlim)

    if "ref_log_xlim" not in st.session_state or update_ref:
        xmin = float(max(np.min(abs_i_pos) * 0.8, i0 / 1000.0))
        xmax = float(np.max(abs_i_pos) * 1.10)
        st.session_state.ref_log_xlim = (xmin, xmax)

    # 主体布局：两张图并排显示
    col1, col2 = st.columns(2)

    with col1:
        # 图 1：线性坐标下展示总电流与分量（阴极分量用负号显示方向）
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(i, eta, lw=2.5, label="Butler–Volmer (total)")
        ax.plot(ia, eta, lw=2.0, ls="--", label=r"$i_a$ (anodic)")
        ax.plot(-ic, eta, lw=2.0, ls="--", label=r"$i_c$ (cathodic)")

        ax.set_title("Figure 1: Linear scale")
        ax.set_xlabel("Current density, i (A/m²)")
        ax.set_ylabel("Overpotential, η (V)")
        if fix_axis:
            ax.set_xlim(st.session_state.ref_lin_xlim)
        ax.grid(True, ls="--", lw=0.6, alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    with col2:
        # 图 2：对数坐标下展示幅值 |i|，便于观察指数增长/衰减的直线化趋势
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(np.abs(i), eta, lw=2.5, label="|i| (total)")
        ax.plot(np.abs(ia), eta, lw=2.0, ls="--", label=r"$|i_a|$ (anodic)")
        ax.plot(np.abs(ic), eta, lw=2.0, ls="--", label=r"$|i_c|$ (cathodic)")

        ax.set_xscale("log")
        ax.set_title("Figure 2: Log scale (magnitude)")
        ax.set_xlabel("Current density magnitude, |i| (A/m²)")
        ax.set_ylabel("Overpotential, η (V)")
        if fix_axis:
            ax.set_xlim(st.session_state.ref_log_xlim)
        ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    # 公式与解释：用 st.latex 保证公式稳定显示（修复 \quad 与 i 连写导致的 \quadi 错误）
    st.subheader("公式与物理含义（BV）")

    st.markdown("Butler–Volmer 方程（以过电位 η 为自变量）常写为：")
    st.latex(
        r"i(\eta)=i_0\left[\exp\!\left(\frac{\alpha_a nF\eta}{RT}\right)-\exp\!\left(-\frac{\alpha_c nF\eta}{RT}\right)\right]"
    )

    st.markdown("极分量：")
    st.latex(r"i_a = i_0 \exp\!\left(\frac{\alpha_a nF\eta}{RT}\right)")
    st.latex(r"i_c = i_0 \exp\!\left(-\frac{\alpha_c nF\eta}{RT}\right)")
    st.latex(r"i = i_a - i_c")

    st.markdown(
        """
        说明与要点：
        - **i0（交换电流密度）** 表征反应动力学的“整体尺度”，i0 越大，在同一 η 下电流越大；
        - **αa、αc（传递系数）** 决定阳极/阴极支路的陡峭程度，并决定对应的 Tafel 斜率；
        - 在足够大的正/负过电位区间，BV 行为可近似进入 Tafel 区间（对数坐标下呈近似线性趋势）。
        """
    )

    st.subheader("Tafel 斜率（参考）")
    st.latex(r"b=\frac{2.303RT}{\alpha nF}\quad(\mathrm{V/dec})")
    st.markdown(
        f"当前参数：  "
        f"$b_a$ = **{tafel_slope(alpha_a, n, T):.3f} V/dec**, "
        f"$b_c$ = **{tafel_slope(alpha_c, n, T):.3f} V/dec**"
    )

    if use_ilim:
        st.subheader("限制电流密度 i_lim（可选）")
        st.markdown(
            "启用 $i_{lim}$ 后，这里对阴极分量做了软限制，用于演示传质限制导致的“平台/钝化”趋势。"
        )


if __name__ == "__main__":
    # 入口：这是 Streamlit 应用，请用 streamlit run 启动
    main()
