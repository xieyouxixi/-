import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------- 1. 数据准备（评分卡数据）----------------------
# 构建评分卡数据框（还原原始数据，处理格式问题）
scorecard_data = pd.DataFrame({
    '评分区间': ['300–350', '350–400', '400–450', '450–500', '500–550', '550–600',
                 '600–650', '650–700', '700–750', '750–800', '800–850', '850–900',
                 '900–950', '950–1000'],
    '区间中点': [325, 375, 425, 475, 525, 575, 625, 675, 725, 775, 825, 875, 925, 975],
    '区间人数': [49, 165, 441, 919, 1499, 1915, 1915, 1499, 919, 441, 165, 49, 11, 2],
    '违约概率(%)': [99.78, 98.78, 93.49, 71.90, 31.35, 7.53, 1.43, 0.26, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00],
    '违约人数': [49, 163, 412, 662, 470, 144, 27, 4, 0, 0, 0, 0, 0, 0]
})

# 数据预处理：违约概率转换为小数，计算非违约人数
scorecard_data['违约概率'] = scorecard_data['违约概率(%)'] / 100  # 百分比转小数
scorecard_data['非违约人数'] = scorecard_data['区间人数'] - scorecard_data['违约人数']

# ---------------------- 2. 生成模型评估所需数据（标签+预测概率+权重）----------------------
# 初始化存储列表
y_true = []  # 真实标签（1=违约，0=非违约）
y_pred_prob = []  # 预测概率（评分卡对应的违约概率）
sample_weights = []  # 样本权重（每个区间的人数，代表该区间的样本量）

# 遍历每个评分区间，生成对应的标签、预测概率和权重
for idx, row in scorecard_data.iterrows():
    # 违约样本：标签=1，预测概率=该区间违约概率，权重=违约人数
    y_true.extend([1] * int(row['违约人数']))
    y_pred_prob.extend([row['违约概率']] * int(row['违约人数']))
    sample_weights.extend([1] * int(row['违约人数']))  # 若需按人数加权，可改为[row['违约人数']]

    # 非违约样本：标签=0，预测概率=该区间违约概率，权重=非违约人数
    y_true.extend([0] * int(row['非违约人数']))
    y_pred_prob.extend([row['违约概率']] * int(row['非违约人数']))
    sample_weights.extend([1] * int(row['非违约人数']))  # 若需按人数加权，可改为[row['非违约人数']]

# 转换为numpy数组（适配sklearn接口）
y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
sample_weights = np.array(sample_weights)

# ---------------------- 3. 计算AUC值 ----------------------
# 方法1：不考虑样本权重（默认，适用于个体级数据）
auc_without_weight = roc_auc_score(y_true, y_pred_prob)

# 方法2：考虑样本权重（适配评分卡统计数据，更贴合实际业务场景）
auc_with_weight = roc_auc_score(y_true, y_pred_prob, sample_weight=sample_weights)

# 输出结果
print("=" * 50)
print("评分卡模型AUC计算结果")
print("=" * 50)
print(f"不考虑样本权重的AUC: {auc_without_weight:.4f}")
print(f"考虑样本权重的AUC: {auc_with_weight:.4f}")
print("\nAUC效果判断（参考标准）:")
print("0.50-0.70: 效果较低")
print("0.70-0.85: 效果一般")
print("0.85-0.95: 效果很好")
print("0.95-1.00: 效果非常好（罕见）")
print("=" * 50)

# ---------------------- 4. 绘制ROC曲线（可视化）----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 计算ROC曲线的假正率（FPR）、真正率（TPR）
fpr, tpr, _ = roc_curve(y_true, y_pred_prob, sample_weight=sample_weights)

# 绘制ROC曲线
plt.figure(figsize=(8, 6), dpi=100)
# ROC曲线
plt.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'ROC曲线 (AUC = {auc_with_weight:.4f})')
# 随机猜测基准线（AUC=0.5）
plt.plot([0, 1], [0, 1], color='#A23B72', linewidth=1.5, linestyle='--', label='随机猜测（AUC=0.5）')

# 图表美化
plt.xlabel('假正率（FPR）', fontsize=12)
plt.ylabel('真正率（TPR）', fontsize=12)
plt.title('评分卡模型ROC曲线', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# 保存图片（可选）
plt.savefig('评分卡ROC曲线.png', bbox_inches='tight', dpi=150)
plt.show()