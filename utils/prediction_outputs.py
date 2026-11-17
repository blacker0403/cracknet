"""Inference visualizations and model-score summaries; no ground-truth metrics."""
import csv
from pathlib import Path

import cv2
import numpy as np
from utils.visualization import save_figure
import matplotlib.pyplot as plt


def prediction_statistics(probability, threshold):
    foreground = probability >= threshold
    selected_probability = np.where(foreground, probability, 1 - probability)
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return {'width': probability.shape[1], 'height': probability.shape[0], 'threshold': threshold,
            'foreground_pixels': int(foreground.sum()),
            'foreground_percent': float(foreground.mean() * 100),
            'foreground_probability_mean': float(probability[foreground].mean()) if foreground.any() else None,
            'background_probability_mean': float((1 - probability[~foreground]).mean()) if (~foreground).any() else None,
            'predicted_class_probability_mean': float(selected_probability.mean()),
            'uncertain_percent': float((np.abs(probability - threshold) <= .1).mean() * 100),
            'binary_entropy_mean_bits': float((-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean())}


def save_prediction_visualization(image, probability, threshold, sample_id, output, edge=None):
    output = str(output)
    foreground = probability >= threshold
    stats = prediction_statistics(probability, threshold)
    overlay = image.copy()
    overlay[foreground] = (.55 * image[foreground] + .45 * np.array([255, 60, 40])).astype(np.uint8)
    cv2.imwrite(output + '.original.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output + '.overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    axes[0].imshow(image)
    axes[0].set_title('Original image')
    axes[1].imshow(foreground, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Predicted mask (threshold={threshold:g})')
    if edge is not None:
        axes[2].imshow(edge, cmap='gray', vmin=0, vmax=1)
    else:
        axes[2].text(.5, .5, 'Edge head not available', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Predicted edge probability')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(sample_id)
    score = stats['foreground_probability_mean']
    score_text = f'{score:.3f}' if score is not None else 'N/A'
    fig.text(.5, .025, f"Crack pixels: {stats['foreground_pixels']:,} ({stats['foreground_percent']:.4f}%)"
             f" | Mean foreground p: {score_text} | Near-threshold: {stats['uncertain_percent']:.4f}%",
             ha='center', fontsize=8)
    fig.text(.5, -.005, 'Model probabilities are not measured accuracy.', ha='center', fontsize=8)
    save_figure(fig, output + '.comparison.png')

    selected_probability = np.where(foreground, probability, 1 - probability)
    for suffix, values, title in [('probability', probability, 'Crack probability p'),
                                  ('confidence', selected_probability, 'Predicted-class probability')]:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        heatmap = ax.imshow(values, cmap='magma', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(heatmap, ax=ax, label='Model probability (0 to 1)', fraction=.046, pad=.04)
        fig.text(.5, .015, 'Model score; measured accuracy requires pixel-level ground truth.',
                 ha='center', fontsize=8)
        save_figure(fig, output + f'.{suffix}.png')
    return stats


def write_prediction_summary(output, rows):
    output = Path(output)
    with (output / 'prediction_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'README.md').write_text('''# 推理结果说明

每张图片保留原文件扩展名，再追加以下后缀：

| 后缀 | 内容 |
|---|---|
| .original.png | 解码后的原图，保持原尺寸 |
| .mask.png | 按冻结阈值生成的二值裂缝掩码，白色为预测裂缝 |
| .edge.png | 模型边界头的概率灰度图，非二值真值；无边界头时不生成 |
| .comparison.png | 原图、预测mask、预测edge的横向对照 |
| .overlay.png | 在原图上以红色叠加预测裂缝，红色不代表预测错误 |
| .probability.npy | 原尺寸每像素裂缝概率p的数值数组 |
| .probability.png | 裂缝概率热图，所有图统一0–1色标 |
| .confidence.png | 预测类别的概率热图：预测裂缝取p，预测背景取1-p |

## prediction_summary.csv 的字段

- id、width、height：图片相对路径及原始宽高。
- threshold：源验证集选择并冻结的分割阈值。
- foreground_pixels、foreground_percent：预测裂缝像素数及其占全图百分比。
- foreground_probability_mean：预测裂缝区域的平均裂缝概率，范围0–1；没有预测裂缝时留空。
- background_probability_mean：预测背景区域的平均背景概率(1-p)，范围0–1；没有预测背景时留空。
- predicted_class_probability_mean：全图预测类别的平均概率，范围0–1。
- uncertain_percent：裂缝概率与分割阈值相差不超过0.1的像素百分比。
- binary_entropy_mean_bits：二分类预测熵的全图平均值，约0–1；越高表示模型分布越不确定。
- inference_seconds：该图滑窗推理及拼接耗时，不含可视化导出；旧结果补图时若无逐图计时则留空。

这些概率来自模型输出，未验证概率校准，也不是预测正确率。大面积背景可以抬高全图平均置信度，应结合前景概率和原图叠加图判断。
本流程不读取真值标签，未计算真实准确率、IoU或Dice。真实分割指标需要与原图对齐的像素级真值掩码；检测框TXT不能直接替代分割掩码。
''', encoding='utf-8')
