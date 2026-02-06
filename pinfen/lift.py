import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（兼容Windows/Mac/Linux，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ---------------------- 1. 数据准备（核心修正：确保所有字段长度一致） ----------------------
# 修正后的数据：所有字段均为14个元素，补充缺失的"400–450"评分区间
data = {
    '评分区间': ['300–350', '350–400', '400–450', '450–500', '500–550', '550–600',
               '600–650', '650–700', '700–750', '750–800', '800–850', '850–900', '900–950', '950–1000'],
    '区间中点': [325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825, 875, 925, 975],
    '区间人数': [49, 165, 441, 919, 1499, 1915, 1915, 1499, 919, 441, 165, 49, 11, 2],
    '违约概率(%)': [99.78, 98.78, 93.49, 71.90, 31.35, 7.53, 1.43, 0.26, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00],
    '违约人数': [49, 163, 412, 662, 470, 144, 27, 4, 0, 0, 0, 0, 0, 0]
}

# 数据校验：确保所有字段长度一致（避免再次出现长度不匹配错误）
field_lengths = {key: len(value) for key, value in data.items()}
print("各字段长度校验：", field_lengths)
if len(set(field_lengths.values())) != 1:
    raise ValueError(f"字段长度不一致！{field_lengths}")

# 构建DataFrame
df = pd.DataFrame(data)

# ---------------------- 2. 数据预处理（确保数值型字段可计算） ----------------------
# 转换为数值型，错误值设为NaN（避免计算报错）
df['区间人数'] = pd.to_numeric(df['区间人数'], errors='coerce')
df['违约人数'] = pd.to_numeric(df['违约人数'], errors='coerce')
df['违约概率'] = df['违约概率(%)'] / 100  # 转换为小数（便于后续计算）

# 计算信贷风控核心指标
total_samples = df['区间人数'].sum()  # 总样本数
total_bad_count = df['违约人数'].sum()  # 总违约人数
total_bad_rate = total_bad_count / total_samples  # 整体坏账率
df['区间坏账率'] = df['违约人数'] / df['区间人数']  # 区间坏账率（实际违约占比）
df['Lift提升度'] = df['区间坏账率'] / total_bad_rate  # Lift=区间坏账率/整体坏账率（模型区分能力）

# 填充可能的NaN值（如区间人数为0时的除法错误）
df = df.fillna(0)

# ---------------------- 3. 多子图可视化（2x2布局，覆盖风控核心分析场景） ----------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 画布大小：16*12英寸（高清）
fig.suptitle('评分卡模型效果可视化分析', fontsize=20, fontweight='bold', y=0.95)  # 总标题

# 专业配色方案（避免视觉杂乱，符合学术/业务报告规范）
colors = {
    'main': '#2E86AB',    # 主色（蓝色：用于核心趋势）
    'accent': '#A23B72',  # 强调色（紫色：用于关键指标）
    'neutral': '#F18F01'  # 中性色（橙色：用于辅助图表）
}

# ---------------------- 子图1：违约概率分布（评分区间维度） ----------------------
ax1 = axes[0, 0]
bars1 = ax1.bar(
    df['评分区间'], df['违约概率(%)'],
    color=colors['main'], alpha=0.8,  # 透明度0.8，避免过于厚重
    edgecolor='white', linewidth=1.5  # 白色边框，提升图表清晰度
)
ax1.set_title('违约概率分布', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('评分区间', fontsize=12)
ax1.set_ylabel('违约概率(%)', fontsize=12)
ax1.tick_params(axis='x', rotation=45)  # X轴文字旋转45度，避免重叠
ax1.grid(axis='y', alpha=0.3, linestyle='--')  # 虚线网格，不干扰数据展示

# 柱子上添加数值标签（仅显示非零值，避免拥挤）
for bar in bars1:
    height = bar.get_height()
    if height > 0.1:  # 只显示>0.1%的标签，过滤极小值
        ax1.text(
            bar.get_x() + bar.get_width()/2, height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10
        )

# ---------------------- 子图2：样本人数分布（直方图） ----------------------
ax2 = axes[0, 1]
# 用直方图展示人数分布，bins=区间数，weights=区间人数（实现按人数加权）
ax2.hist(
    df['区间中点'], bins=len(df),
    weights=df['区间人数'],
    color=colors['neutral'], alpha=0.8,
    edgecolor='white', linewidth=1.5
)
ax2.set_title('样本人数分布', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('评分区间中点', fontsize=12)
ax2.set_ylabel('区间人数', fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 标注人数峰值（突出核心用户群体）
max_count_row = df.loc[df['区间人数'].idxmax()]
ax2.annotate(
    f'峰值：{max_count_row["区间人数"]}人\n{max_count_row["评分区间"]}',
    xy=(max_count_row['区间中点'], max_count_row['区间人数']),
    xytext=(max_count_row['区间中点'] + 200, max_count_row['区间人数'] + 200),
    arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=2),
    fontsize=11, color=colors['accent'], fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)  # 文本背景框，提升可读性
)

# ---------------------- 子图3：Lift提升度趋势（模型区分能力核心指标） ----------------------
ax3 = axes[1, 0]
# 筛选Lift>0的数据（避免0值干扰趋势）
lift_data = df[df['Lift提升度'] > 0]
# 绘制Lift趋势线（带标记点）
ax3.plot(
    lift_data['区间中点'], lift_data['Lift提升度'],
    marker='o', markersize=6, linewidth=3,
    color=colors['accent'], markerfacecolor='white', markeredgewidth=2
)
# 添加基准线（Lift=1：随机选择的效果，Lift>1说明模型有效）
ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='基准线（Lift=1）')
# 填充Lift>1的区域（突出模型有效区间）
ax3.fill_between(
    lift_data['区间中点'], lift_data['Lift提升度'], 1,
    alpha=0.3, color=colors['accent']
)

ax3.set_title('Lift提升度趋势', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('评分区间中点', fontsize=12)
ax3.set_ylabel('Lift提升度', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3, linestyle='--')

# 添加前5个高Lift值标签（避免图表拥挤）
for _, row in lift_data.head(5).iterrows():
    ax3.text(
        row['区间中点'], row['Lift提升度'] + 0.8,
        f'{row["Lift提升度"]:.1f}x',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

# ---------------------- 子图4：累计Lift曲线（模型整体效果评估） ----------------------
ax4 = axes[1, 1]
# 按评分升序排列（风险从高到低）
df_sorted = df.sort_values('区间中点').reset_index(drop=True)
# 计算累计指标
df_sorted['累计样本数'] = df_sorted['区间人数'].cumsum()
df_sorted['累计违约人数'] = df_sorted['违约人数'].cumsum()
df_sorted['累计坏账率'] = df_sorted['累计违约人数'] / df_sorted['累计样本数']
df_sorted['累计Lift'] = df_sorted['累计坏账率'] / total_bad_rate

# 绘制累计Lift曲线
ax4.plot(
    df_sorted['累计样本数'] / total_samples * 100,  # X轴：累计样本占比（%）
    df_sorted['累计Lift'],
    marker='s', markersize=4, linewidth=3,
    color=colors['main']
)
# 添加基准线
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='基准线（Lift=1）')

ax4.set_title('累计Lift曲线', fontsize=14, fontweight='bold', pad=20)
ax4.set_xlabel('累计样本占比(%)', fontsize=12)
ax4.set_ylabel('累计Lift提升度', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3, linestyle='--')

# ---------------------- 4. 布局优化与保存 ----------------------
plt.tight_layout()  # 自动调整子图间距，避免标签重叠
# 保存高清图片（300dpi，支持插入报告/论文）
plt.savefig(
    '评分卡模型可视化分析.png',
    dpi=300, bbox_inches='tight', facecolor='white'  # 白色背景，避免透明底问题
)
plt.show()  # 显示图片

# ---------------------- 5. 输出核心统计信息（业务决策支持） ----------------------
print("\n" + "="*60)
print("评分卡模型核心统计信息")
print("="*60)
print(f"总样本数：{total_samples:,}")
print(f"总违约人数：{total_bad_count:,}")
print(f"整体坏账率：{total_bad_rate:.4f} ({total_bad_rate*100:.2f}%)")
print(f"最高Lift提升度：{df['Lift提升度'].max():.2f}x")
print(f"Lift>3的高风险区间数：{len(df[df['Lift提升度']>3])}个")
print(f"低风险区间（评分>650分）违约率：{df[df['区间中点']>650]['违约概率(%)'].max():.2f}%")
print("="*60)