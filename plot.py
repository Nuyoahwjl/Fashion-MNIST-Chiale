import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def plot_loss_accuracy1(losses, accuracies):
    # 平滑处理
    window_size = 15
    # 将NumPy数组转换为Pandas Series
    losses_series = pd.Series(losses)
    accuracies_series = pd.Series(accuracies)
    smooth_losses = losses_series.rolling(window=window_size, center=True, min_periods=1).mean()
    smooth_accuracies = accuracies_series.rolling(window=window_size, center=True, min_periods=1).mean()
    # 创建双轴图
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    # 绘制Loss曲线（左侧轴）
    line1, = ax1.plot(losses, alpha=0.2, color='tab:blue', label='Loss (Original)')
    line2, = ax1.plot(smooth_losses, color='tab:blue', linewidth=2, label='Loss (Smoothed)')
    # 绘制Accuracy曲线（右侧轴）
    line3, = ax2.plot(accuracies, alpha=0.2, color='tab:orange', label='Accuracy (Original)')
    line4, = ax2.plot(smooth_accuracies, color='tab:orange', linewidth=2, label='Accuracy (Smoothed)')
    # 设置轴标签
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    # 设置刻度颜色
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    # 合并图例
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), fontsize='small')
    # 添加标题和网格
    plt.title('Training Loss and Accuracy')
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_accuracy2(losses, accuracies):
    window_size = 15
    losses_series = pd.Series(losses)
    accuracies_series = pd.Series(accuracies)
    smooth_losses = losses_series.rolling(window=window_size, center=True, min_periods=1).mean()
    smooth_accuracies = accuracies_series.rolling(window=window_size, center=True, min_periods=1).mean()
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(losses, alpha=0.2, color='tab:blue', label='Loss (Original)')
    plt.plot(smooth_losses, color='tab:blue', linewidth=2, label='Loss (Smoothed)')
    plt.title('Training Loss and Accuracy')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(accuracies, alpha=0.2, color='tab:orange', label='Accuracy (Original)')
    plt.plot(smooth_accuracies, color='tab:orange', linewidth=2, label='Accuracy (Smoothed)')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    log_dir = os.path.join(current_dir, 'logs')
    losses = np.load(os.path.join(log_dir, 'losses.npy'))
    accuracies = np.load(os.path.join(log_dir, 'accuracies.npy'))
    plot_loss_accuracy1(losses, accuracies)
    plot_loss_accuracy2(losses, accuracies)
    losses = losses[0:1000]
    accuracies = accuracies[0:1000] 
    plot_loss_accuracy1(losses, accuracies)
    plot_loss_accuracy2(losses, accuracies)

