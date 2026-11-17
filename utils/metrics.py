"""Foreground metrics at native scale. AUPRC is explicitly histogram-approximated AP."""
import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize


def ratio(num, den, empty=1.0):
    return float(num / den) if den else float(empty)


def confusion(pred, target):
    p, y = np.asarray(pred, bool), np.asarray(target, bool)
    return np.array([(p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()], dtype=np.int64)


def pixel_scores(counts, empty=1.0):
    tp, fp, fn, tn = counts.tolist()
    return {'iou': ratio(tp, tp + fp + fn, empty), 'dice': ratio(2 * tp, 2 * tp + fp + fn, empty),
            'precision': ratio(tp, tp + fp, empty if tp + fn == 0 else 0),
            'recall': ratio(tp, tp + fn, empty), 'pixel_accuracy': ratio(tp + tn, tp + fp + fn + tn, empty),
            'two_class_miou': (ratio(tp, tp + fp + fn, empty) + ratio(tn, tn + fp + fn, empty)) / 2}


def boundary(mask, kernel=3):
    mask = np.asarray(mask, np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((kernel, kernel), np.uint8),
                            borderType=cv2.BORDER_REPLICATE).astype(bool)


def topology_counts(pred, target):
    p, y = np.asarray(pred, bool), np.asarray(target, bool)
    sp, sy = skeletonize(p), skeletonize(y)
    lp, ly = label(p, connectivity=2), label(y, connectivity=2)
    # Fragmentation proxy: extra predicted components intersecting each GT skeleton component.
    overlap = sy & (lp > 0)
    pairs = np.unique(np.stack([ly[overlap], lp[overlap]], axis=1), axis=0)
    touched = np.bincount(pairs[:, 0], minlength=int(ly.max()) + 1)
    fragments = int(np.maximum(touched[1:] - 1, 0).sum())
    return {'skel_pred': int(sp.sum()), 'skel_gt': int(sy.sum()),
            'skel_pred_inside_gt': int((sp & y).sum()), 'skel_gt_inside_pred': int((sy & p).sum()),
            'pred_components': int(lp.max()), 'gt_components': int(ly.max()),
            'fragmentation_proxy': fragments}


def topology_scores(counts, empty=1.0):
    precision = ratio(counts['skel_pred_inside_gt'], counts['skel_pred'], empty if counts['skel_gt'] == 0 else 0)
    recall = ratio(counts['skel_gt_inside_pred'], counts['skel_gt'], empty)
    return {'cldice': ratio(2 * precision * recall, precision + recall, 0), 'skeleton_coverage': recall}


def boundary_counts(pred, target, tolerance=2):
    p, y = boundary(pred), boundary(target)
    yy, xx = np.ogrid[-tolerance:tolerance + 1, -tolerance:tolerance + 1]
    kernel = (xx ** 2 + yy ** 2 <= tolerance ** 2).astype(np.uint8)
    dp, dy = cv2.dilate(p.astype(np.uint8), kernel) > 0, cv2.dilate(y.astype(np.uint8), kernel) > 0
    return np.array([(p & dy).sum(), p.sum(), (y & dp).sum(), y.sum()], dtype=np.int64)


def boundary_score(counts, empty=1.0):
    mp, npred, my, ngt = counts
    precision = ratio(mp, npred, empty if ngt == 0 else 0)
    recall = ratio(my, ngt, empty if npred == 0 else 0)
    return ratio(2 * precision * recall, precision + recall, 0)


class PRHistogram:
    def __init__(self, bins=4096):
        self.bins = bins
        self.positive = np.zeros(bins, np.int64)
        self.negative = np.zeros(bins, np.int64)

    def update(self, probability, target):
        bins = np.minimum((np.clip(probability, 0, 1) * (self.bins - 1)).astype(int), self.bins - 1)
        target = np.asarray(target, bool)
        self.positive += np.bincount(bins[target], minlength=self.bins)
        self.negative += np.bincount(bins[~target], minlength=self.bins)

    def ap(self, empty=1.0):
        if not self.positive.sum():
            return float(empty)  # documented convention for all-negative GT
        tp, fp = np.cumsum(self.positive[::-1]), np.cumsum(self.negative[::-1])
        precision = tp / np.maximum(tp + fp, 1)
        return float(np.sum(precision * self.positive[::-1]) / self.positive.sum())

    def curves(self, empty=1.0):
        """Operating points at quantized pixel probabilities; same bins as AP."""
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        positives = int(self.positive.sum())
        precision = np.divide(tp, tp + fp, out=np.full(self.bins, empty if positives == 0 else 0.0), where=tp + fp > 0)
        recall = tp / positives if positives else np.full(self.bins, empty)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(self.bins), where=precision + recall > 0)
        return {'threshold': np.arange(self.bins - 1, -1, -1) / (self.bins - 1),
                'precision': precision, 'recall': recall, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': positives - tp}


class MetricAccumulator:
    def __init__(self, threshold=0.5, tolerance=2, empty=1.0, bins=4096, detailed=True):
        self.threshold, self.tolerance, self.empty, self.bins = threshold, tolerance, empty, bins
        self.detailed = detailed
        self.cm, self.bc = np.zeros(4, np.int64), np.zeros(4, np.int64)
        self.tc, self.rows, self.pr = None, [], PRHistogram(bins)

    def merge(self, other):
        self.cm += other.cm
        self.bc += other.bc
        if other.tc is not None:
            self.tc = other.tc.copy() if self.tc is None else {k: self.tc[k] + v for k, v in other.tc.items()}
        self.rows.extend(other.rows)
        self.pr.positive += other.pr.positive
        self.pr.negative += other.pr.negative

    def update(self, probability, target, sample_id=''):
        if probability.shape != target.shape or not np.isfinite(probability).all():
            raise ValueError('Metric arrays must have equal native shape and finite probabilities.')
        pred = probability >= self.threshold
        cm = confusion(pred, target)
        self.cm += cm
        extra = dict.fromkeys(['cldice', 'skeleton_coverage', 'boundary_f1', 'pred_components', 'gt_components', 'fragmentation_proxy'])
        if self.detailed:
            bc, tc = boundary_counts(pred, target, self.tolerance), topology_counts(pred, target)
            self.bc += bc
            self.tc = tc.copy() if self.tc is None else {k: self.tc[k] + v for k, v in tc.items()}
            extra = {**topology_scores(tc, self.empty), 'boundary_f1': boundary_score(bc, self.empty),
                     **{k: tc[k] for k in ['pred_components', 'gt_components', 'fragmentation_proxy']}}
        pr = PRHistogram(self.bins)
        pr.update(probability, target)
        self.pr.positive += pr.positive
        self.pr.negative += pr.negative
        scores = {**pixel_scores(cm, self.empty), **extra, 'auprc_hist_ap': pr.ap(self.empty)}
        self.rows.append({'id': sample_id, **scores})
        return scores

    def compute(self):
        if not self.rows:
            raise ValueError('Cannot evaluate an empty split.')
        extra = ({**topology_scores(self.tc, self.empty), 'boundary_f1': boundary_score(self.bc, self.empty)}
                 if self.detailed else dict.fromkeys(['cldice', 'skeleton_coverage', 'boundary_f1']))
        return {'micro': {**pixel_scores(self.cm, self.empty), **extra, 'auprc_hist_ap': self.pr.ap(self.empty)},
                'macro': {k: float(np.mean([r[k] for r in self.rows])) if self.rows[0][k] is not None else None
                          for k in self.rows[0] if k != 'id'},
                'confusion_tp_fp_fn_tn': self.cm.tolist(), 'topology_totals': self.tc,
                'n_images': len(self.rows), 'threshold': self.threshold, 'detailed_metrics_computed': self.detailed,
                'auprc_definition': f'Histogram average precision, {self.bins} bins; not official Crack500 ODS/OIS/AIU'}
