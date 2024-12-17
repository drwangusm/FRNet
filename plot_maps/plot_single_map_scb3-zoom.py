import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 训练结果列表
results_files = [
    '/final/improved-yolov11/runs/train/SCB3/yolov5/results.csv',
    '/final/improved-yolov11/runs/train/SCB3/yolov6/results.csv',
    '/final/improved-yolov11/runs/train/SCB3/yolov8/results.csv',
    '/final/improved-yolov11/runs/train/SCB3/yolov10n/results.csv',
    '/final/improved-yolov11/runs/train/SCB3/yolo11/results.csv',
    '/final/improved-yolov11/runs/train/SCB3/yolo11-FPSC-RGCSPELAN/results.csv',
]

# 与results_files顺序对应
custom_labels = [
    'yolov5',
    'yolov6',
    'yolov8',
    'yolov10n',
    'yolo11',
    'Ours',
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

    plt.title(metric_label)
    plt.xlabel('Epochs')
    plt.ylabel(metric_label)
    plt.legend()

    # 添加放大镜
    ax_main = plt.gca()
    ax_inset = inset_axes(ax_main, width="40%", height="40%", loc='center',
                          bbox_to_anchor=(0.6, 0.5, 0.4, 0.4), bbox_transform=ax_main.transAxes)

    # 绘制放大图
    for file_path, custom_label in zip(results_files, custom_labels):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if 'epoch' not in df.columns or metric_key not in df.columns:
            continue

        ax_inset.plot(df['epoch'], df[metric_key], label=custom_label)

    # 设置放大图区域范围
    ax_inset.set_xlim(280, 300)  # 这里设置放大图的x轴范围
    ax_inset.set_ylim(0.47, 0.52)  # 这里设置放大图的y轴范围
    ax_inset.set_xticks([280, 290, 300])
    ax_inset.set_yticks([0.47, 0.50, 0.52])
    ax_inset.tick_params(labelsize=8)

    # 标记放大区域
    mark_inset(ax_main, ax_inset, loc1=2, loc2=4, fc="none", ec="red", lw=1.5)

    # 保存并显示图像
    save_path = f'/final/improved-yolov11/plot_results/scbd3/{metric_label}_zoomed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == '__main__':
    # 绘制mAP对比图并添加放大区域
    # plot_metric_comparison_with_zoom('metrics/mAP50(B)', 'mAP@50', custom_labels)
    plot_metric_comparison_with_zoom('metrics/mAP50-95(B)', 'mAP@50-95', custom_labels)
