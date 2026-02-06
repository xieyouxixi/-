# 先导入基础库，再导入matplotlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 先设置后端，避免导入时的显示问题
import matplotlib.pyplot as plt

# -------------------------- 1. 数据准备：还原评分卡数据 --------------------------
scorecard_data = {
    "评分区间": ["300–350", "350–400", "400–450", "450–500", "500–550", "550–600",
               "600–650", "650–700", "700–750", "750–800", "800–850", "850–900",
               "900–950", "950–1000"],
    "区间中点": [325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825, 875, 925, 975],
    "区间人数": [49, 165, 441, 919, 1499, 1915, 1915, 1499, 919, 441, 165, 49, 11, 2],
    "违约人数": [49, 163, 412, 662, 470, 144, 27, 4, 0, 0, 0, 0, 0, 0]
}

# 转换为DataFrame
df = pd.DataFrame(scorecard_data)

# 计算核心指标
df["好客户数"] = df["区间人数"] - df["违约人数"]
total_bad = df["违约人数"].sum()
total_good = df["好客户数"].sum()

# 计算KS关键指标
df["BadPct"] = df["违约人数"] / total_bad
df["GoodPct"] = df["好客户数"] / total_good
df["Cum_BadPct"] = df["BadPct"].cumsum()
df["Cum_GoodPct"] = df["GoodPct"].cumsum()
df["KS"] = abs(df["Cum_BadPct"] - df["Cum_GoodPct"])

# 找到最大KS值及对应信息
max_ks_idx = df["KS"].idxmax()
max_ks_value = df.loc[max_ks_idx, "KS"]
max_ks_score = df.loc[max_ks_idx, "区间中点"]
max_ks_interval = df.loc[max_ks_idx, "评分区间"]

print(f"最大KS值：{max_ks_value:.2%}")
print(f"对应评分区间：{max_ks_interval}（区间中点：{max_ks_score}）")

# -------------------------- 2. 可视化配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 兼容更多Windows字体
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8), dpi=150)

# 定义颜色方案
color_bad = '#E74C3C'
color_good = '#27AE60'
color_ks = '#3498DB'
color_line = '#95A5A6'

# -------------------------- 3. 绘制核心图表 --------------------------
plt.plot(df["区间中点"], df["Cum_BadPct"],
         color=color_bad, marker='o', markersize=6, linewidth=2.5, label='累计坏客户占比（Bad）')
plt.plot(df["区间中点"], df["Cum_GoodPct"],
         color=color_good, marker='s', markersize=6, linewidth=2.5, label='累计好客户占比（Good）')
plt.plot(df["区间中点"], df["KS"],
         color=color_ks, marker='^', markersize=7, linewidth=3, linestyle='--', label='KS值')

# 标注最大KS值
plt.scatter(max_ks_score, max_ks_value, color='red', s=150, edgecolor='black', zorder=5)
plt.annotate(
    f'最大KS值：{max_ks_value:.2%}\n评分区间：{max_ks_interval}',
    xy=(max_ks_score, max_ks_value),
    xytext=(max_ks_score + 50, max_ks_value - 0.2),
    fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', linewidth=1.5)
)

# 辅助线
plt.axhline(y=max_ks_value, color=color_line, linestyle=':', linewidth=2, label=f'最大KS值线：{max_ks_value:.2%}')
plt.axvline(x=max_ks_score, color=color_line, linestyle=':', linewidth=2, label=f'最优分界点：{max_ks_score}分')

# -------------------------- 4. 图表美化 --------------------------
plt.xlabel('评分区间中点', fontsize=14, fontweight='bold')
plt.ylabel('占比 / KS值', fontsize=14, fontweight='bold')
plt.title('评分卡KS区分度分析图\n（贷前A卡模型：KS≥0.2为有效区分）', fontsize=16, fontweight='bold', pad=20)
plt.xlim(300, 1000)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
plt.legend(loc='upper left', fontsize=12, framealpha=0.9, shadow=True)

# 百分比格式y轴
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.gca().invert_xaxis()  # 反转x轴（风控习惯：低分在右，高分在左）
plt.tight_layout()

# 保存并显示
plt.savefig('评分卡KS分析图.png', dpi=150, bbox_inches='tight')


# -------------------------- 5. 输出详细指标表 --------------------------
print("\n=== 评分卡KS详细指标表 ===")
result_df = df[["评分区间", "区间中点", "Cum_BadPct", "Cum_GoodPct", "KS"]].copy()
result_df["Cum_BadPct"] = result_df["Cum_BadPct"].map(lambda x: f'{x:.2%}')
result_df["Cum_GoodPct"] = result_df["Cum_GoodPct"].map(lambda x: f'{x:.2%}')
result_df["KS"] = result_df["KS"].map(lambda x: f'{x:.2%}')
print(result_df.to_string(index=False))