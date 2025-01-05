import numpy as np
import pandas as pd
import lifelines  # 导入Lifelines
import matplotlib.pyplot as plt

# 载入数据集
df_member = pd.read_excel('文件路径')
df_member.head()  # 输出前几行数据

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# Kaplan-Meier留存曲线
kmf = lifelines.KaplanMeierFitter()  # 创建KaplanMeier模型
kmf.fit(df_member['等待时间'], event_observed=df_member['是否留存'], label='消费者预期留存线')  # 拟合模型
fig_pmt, ax_pmt = plt.subplots(figsize=(10, 6))  # 创建图像和坐标系
kmf.plot(ax=ax_pmt)  # 绘图
ax_pmt.set_title('Kaplan-Meier留存曲线-所有消费者')  # 图题
ax_pmt.set_xlabel('等待时间')  # x轴
ax_pmt.set_ylabel('留存率（%）')  # y轴
plt.show()

# 定义分类留存曲线函数
def life_by_cat(feature, t='等待时间', event='是否留存', df=df_member, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for cat in df[feature].unique():  # 遍历类别
        idx = df[feature] == cat
        kmf = lifelines.KaplanMeierFitter()  # 构建KaplanMeier模型
        kmf.fit(df.loc[idx, t], event_observed=df.loc[idx, event], label=cat)  # 拟合模型
        kmf.plot(ax=ax, label=cat)  # 绘图
    if ax is not None and not plt.get_fignums():
        plt.show()

# Cox比例风险模型
cph = lifelines.CoxPHFitter()  # 创建CoxPH模型
cph.fit(df_member, duration_col='等待时间', event_col='是否留存', show_progress=False)  # 拟合模型
cph.print_summary()  # 输出结果

# 绘制Cox模型的相关系数和置信区间
fig_coef, ax_coef = plt.subplots(figsize=(10, 6))  # 创建图像和坐标系
ax_coef.set_title('相关系数和置信区间')  # 图题
cph.plot(ax=ax_coef)
plt.show()  # 显示图形