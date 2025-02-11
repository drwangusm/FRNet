import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 训练结果列表
results_files = [
    '/final/FRNet/runs/train/SCB3/yolov5/results.csv',
    '/final/FRNet/runs/train/SCB3/yolov6/results.csv',
    '/final/FRNet/runs/train/SCB3/yolov8/results.csv',
    '/final/FRNet/runs/train/SCB3/yolov10n/results.csv',
    '/final/FRNet/runs/train/SCB3/yolo11/results.csv',
    '/final/FRNet/runs/train/SCB3/yolo11-FPSC-RGCSPELAN/results.csv',
]

# 与results_files顺序对应
custom_labels = [
    'YOLOv5',
    'YOLOv6',
    'YOLOv8n',
    'YOLOv10n',
    'YOLOv11',
    'YOLO-RF',
]

def plot_metric_comparison_with_zoom(metric_key, metric_label, custom_labels):
    plt.figure(figsize=(12, 8))

    # 主图绘制
    for file_path, custom_label in zip(results_files, custom_labels):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # 清理列名中的多余空格

        if 'epoch' not in df.columns or metric_key not in df.columns:
            print(f"Missing required columns in {file_path}. Available columns: {df.columns}")
            continue

        # 绘制主图
        plt.plot(df['epoch'], df[metric_key], label=custom_label)

    plt.title(metric_label, fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(metric_label, fontsize=16)
    plt.xticks(fontsize=12)  # 调整 x 轴刻度字体大小
    plt.yticks(fontsize=12)  # 调整 y 轴刻度字体大小
    plt.legend(fontsize=12)  # 调整图例字体大小
    # 添加放大镜
    ax_main = plt.gca()
    ax_inset = inset_axes(ax_main, width="80%", height="80%", loc='center', 
                          bbox_to_anchor=(0.3, 0.3, 0.4, 0.4), bbox_transform=ax_main.transAxes)

    # 绘制放大图
    for file_path, custom_label in zip(results_files, custom_labels):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if 'epoch' not in df.columns or metric_key not in df.columns:
            continue

        ax_inset.plot(df['epoch'], df[metric_key], label=custom_label)

    # 定义放大区域
    zoom_xmin, zoom_xmax = 270, 280  # 设定x轴放大区域
    zoom_ymin, zoom_ymax = 0.65, 0.73  # 设定y轴放大区域（调整到放大区域）
    # 设置放大图的坐标范围（对应蓝色区域）
    ax_inset.set_xlim(zoom_xmin, zoom_xmax)
    ax_inset.set_ylim(zoom_ymin, zoom_ymax)
    ax_inset.set_xticks([280, 290, 300])
    ax_inset.set_yticks([0.68, 0.70, 0.72])
    ax_inset.tick_params(labelsize=10)
    ax_inset.set_xlabel('Epochs', fontsize=10)  # 设置放大图 x 轴标签字体
    ax_inset.set_ylabel(metric_label, fontsize=10)  # 设置放大图 y 轴标签字体

    # # 在主图上添加红色框，标记被放大的区域(添加竖线)
    # rect = Rectangle((zoom_xmin, zoom_ymin), zoom_xmax - zoom_xmin, zoom_ymax - zoom_ymin, 
    #                  linewidth=1, edgecolor='red', facecolor='none')
    # ax_main.add_patch(rect)

    # 标记放大区域
    mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="red", lw=1)

    # 保存并显示图像
    save_path = f'/final/FRNet/plot_results/scbd3/{metric_label}_zoomed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == '__main__':
    # 绘制mAP对比图并添加放大区域
    plot_metric_comparison_with_zoom('metrics/mAP50(B)', 'mAP@50', custom_labels)
    # plot_metric_comparison_with_zoom('metrics/mAP50-95(B)', 'mAP@50-95', custom_labels)