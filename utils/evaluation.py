"""Validation threshold search and evaluation share exactly the same native inference."""
import time
import numpy as np
import torch

from utils.inference import sliding_window
from utils.metrics import MetricAccumulator, confusion, pixel_scores


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, threshold=None, criterion=None, search=False,
             visualizer=None, corruption=None, curve_collector=None, detailed=True, progress=False):
    e = cfg['evaluation']
    threshold = e['threshold'] if threshold is None else threshold
    metrics = MetricAccumulator(threshold, e['boundary_tolerance'], e['empty_score'], e['auprc_bins'], detailed)
    grid = e['threshold_grid'] if search else []
    counts = {float(t): np.zeros(4, np.int64) for t in grid}
    domains, domain_counts = {}, {}
    losses, skipped, times = [], 0, []
    loss_terms = {key: [] for key in ['bce', 'dice', 'cldice', 'edge']}
    metric_seconds = 0.0
    for index, sample in enumerate(dataset):
        image, target = sample['image'], sample['mask']
        if corruption is not None:
            image = corruption(image, sample['id'])
        start = time.perf_counter()
        output = sliding_window(model, image, e['window'], e['overlap'], device, cfg['train']['amp'],
                                cfg['train'].get('amp_dtype', 'float16'), e.get('window_batch_size', 1))
        times.append(time.perf_counter() - start)
        metric_start = time.perf_counter()
        probability, y = output['probability'].numpy(), target[0].numpy()
        active = metrics
        domain = sample.get('domain') if cfg.get('multidomain') else None
        if domain is not None:
            if domain not in domains:
                domains[domain] = MetricAccumulator(threshold, e['boundary_tolerance'], e['empty_score'], e['auprc_bins'], detailed)
                domain_counts[domain] = {float(t): np.zeros(4, np.int64) for t in grid}
            active = domains[domain]
        active.update(probability, y, sample['id'])
        for t in counts:
            cm = confusion(probability >= t, y)
            counts[t] += cm
            if domain is not None:
                domain_counts[domain][t] += cm
        metric_seconds += time.perf_counter() - metric_start
        if criterion is not None:
            # Native-scale, blended probabilities -> logits for documented validation loss.
            pred = {'logits': torch.logit(output['probability'].clamp(1e-6, 1 - 1e-6))[None, None].to(device),
                    'edge_logits': None}
            if output['edge_probability'] is not None:
                pred['edge_logits'] = torch.logit(output['edge_probability'].clamp(1e-6, 1 - 1e-6))[None, None].to(device)
            loss, terms = criterion(pred, target[None].to(device))
            losses.append(float(loss))
            for key in loss_terms:
                loss_terms[key].append(terms[key])
            skipped += terms['cldice_skipped']
        if visualizer is not None:
            visualizer(sample, output, active.rows[-1])
        if progress and ((index + 1) % e.get('progress_every', 100) == 0 or index + 1 == len(dataset)):
            print(f'[validation] {index + 1}/{len(dataset)} images; CPU_metrics={metric_seconds:.1f}s; detailed={detailed}', flush=True)
    for accumulator in domains.values():
        metrics.merge(accumulator)
    result = metrics.compute()
    if domains:
        result['domains'] = {name: accumulator.compute() for name, accumulator in sorted(domains.items())}
        result['domain_mean'] = {key: float(np.mean([r['micro'][key] for r in result['domains'].values()]))
                                 if result['micro'][key] is not None else None
                                 for key in result['micro']}
        result['domain_worst_iou'] = min(r['micro']['iou'] for r in result['domains'].values())
    result.update(val_loss=float(np.mean(losses)) if losses else None, cldice_skipped=skipped,
                  loss_components={k: float(np.mean(v)) for k, v in loss_terms.items()} if losses else None,
                  native_inference_seconds_mean=float(np.mean(times)),
                  native_fps=1 / float(np.mean(times)),
                  cpu_metric_seconds=metric_seconds,
                  latency_scope='Sliding inference including patch transfer/stitching; excludes image decoding and metrics')
    if search:
        table = [{'threshold': t, **pixel_scores(cm, e['empty_score'])} for t, cm in counts.items()]
        if domains:
            for row in table:
                row['domain_mean_dice'] = float(np.mean([pixel_scores(cm[row['threshold']], e['empty_score'])['dice']
                                                         for cm in domain_counts.values()]))
        # Stable tie rule: smaller threshold wins (grid order sorted explicitly).
        objective = 'domain_mean_dice' if domains else 'dice'
        best = max(sorted(table, key=lambda r: r['threshold']), key=lambda row: row[objective])
        result['threshold_search'] = {'split': 'val', 'objective': 'source_domain_mean_dice' if domains else 'micro_f1', 'tie_rule': 'smallest threshold',
                                      'selected': best['threshold'], 'table': table}
    if curve_collector is not None:
        curve_collector.update(curves=metrics.pr.curves(e['empty_score']),
                               positive_histogram=metrics.pr.positive.copy(),
                               negative_histogram=metrics.pr.negative.copy())
    return result, metrics.rows
