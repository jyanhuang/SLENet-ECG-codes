import matplotlib.pyplot as plt

# 数据
# time_ms = [796, 811, 841, 834, 810, 920, 906, 807, 975, 853, 735, 736, 741]
time_ms = [796, 810, 920, 906]
# accuracy_percent = [98.41, 98.50, 98.54, 98.41, 98.59, 98.59, 98.54, 98.60, 98.60, 98.60, 98.56, 98.54, 98.37]
accuracy_percent = [98.41, 98.59, 98.59, 98.54]

# 设置图形大小
plt.figure(figsize=(10, 6))

# 画柱状图，每个柱状图用不同颜色表示
bars = plt.bar(time_ms, accuracy_percent, color=['#BF6874', '#965A97', '#8EB05A', '#F5BF4C'], width=2)

# 设置横纵坐标标签和标题
plt.xlabel('Time (ms)')
plt.ylabel('Overall Accuracy (%)')

# 设置纵坐标范围和刻度
plt.ylim(98.3, 98.65)
plt.yticks([98.3, 98.35, 98.4, 98.45, 98.5, 98.55, 98.6, 98.65])

# 设置横坐标范围和刻度
plt.xlim(790, 930)
plt.xticks(range(790, 930, 10))

# 添加图例
legend_labels = ['w/o pruning', 'pruning Conv1 (0.3)', 'pruning Conv1 (0.4)', 'pruning Conv1 (0.5)']
plt.legend(bars, legend_labels, bbox_to_anchor=(0.35, 1))

# 添加横向框线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.savefig('Conv1_pruning.png', dpi=1000, bbox_inches='tight')
plt.show()
