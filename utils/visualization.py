from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def dataset_preview(image, mask, sample_id, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image)
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    fig.suptitle(sample_id)
    for ax in axes:
        ax.axis('off')
    save_figure(fig, path)


def prediction_panel(image, target, probability, threshold, sample_id, path, edge=None, baseline=None):
    pred = probability >= threshold
    error = np.zeros((*target.shape, 3), np.uint8)
    error[pred & target.astype(bool)] = [40, 180, 70]
    error[pred & ~target.astype(bool)] = [240, 60, 60]
    error[~pred & target.astype(bool)] = [40, 100, 240]
    skeleton = np.zeros_like(error)
    skeleton[skeletonize(target.astype(bool))] = [40, 100, 240]
    skeleton[skeletonize(pred)] = [240, 60, 60]
    panels = [('Image', image), ('GT', target), ('Candidate', pred), ('TP green / FP red / FN blue', error),
              ('Probability', probability), ('Skeleton GT blue / prediction red', skeleton)]
    if edge is not None:
        panels.append(('Predicted boundary', edge))
    if baseline is not None:
        panels.append(('Baseline (own val threshold)', baseline))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (title, data) in zip(axes.flat, panels):
        ax.imshow(data, cmap='magma' if title in {'Probability', 'Predicted boundary'} else 'gray',
                  vmin=0, vmax=1 if data.ndim == 2 else None)
        ax.set_title(title)
    fig.suptitle(sample_id)
    save_figure(fig, path)


def training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ['train_loss', 'val_loss']:
        axes[0].plot(history['epoch'], history[key], label=key)
    for key in ['iou', 'dice', 'cldice']:
        axes[1].plot(history['epoch'], history[key], label=key)
    for ax in axes:
        ax.legend()
        ax.set_xlabel('Epoch')
    save_figure(fig, path)
