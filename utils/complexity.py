import time
import statistics
import torch
from torch import nn


def complexity(model, shape=(1, 3, 384, 384), device='cpu'):
    """Conv/transpose-conv/Linear MACs; BN, activation, pooling, resizing and losses excluded."""
    macs, handles = [0], []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.ConvTranspose2d):
            macs[0] += inputs[0].numel() * (module.out_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs[0] += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {'parameters': sum(p.numel() for p in model.parameters()), 'conv_linear_macs': macs[0],
            'input_shape': list(shape), 'mac_definition': 'Conv/ConvTranspose/Linear; transpose counts input positions; multiply-add=1 MAC, ~2 FLOPs; excludes normalization/resize/etc.'}


def benchmark(model, shape, device='cpu', warmup=5, repeats=30, amp=False):
    if repeats < 1 or warmup < 0 or shape[0] != 1:
        raise ValueError('Benchmark requires B=1, repeats>=1 and warmup>=0.')
    x, times = torch.zeros(shape, device=device), []
    kind = torch.device(device).type
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for i in range(warmup + repeats):
                if kind == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast(device_type=kind, enabled=amp and kind == 'cuda'):
                    model(x)
                if kind == 'cuda':
                    torch.cuda.synchronize()
                if i >= warmup:
                    times.append(time.perf_counter() - start)
    finally:
        model.train(was_training)
    median = statistics.median(times)
    return {'latency_ms_median': median * 1000, 'fps_median': 1 / median, 'warmup': warmup,
            'repeats': repeats, 'device': str(device), 'amp': amp and kind == 'cuda',
            'cpu_threads': torch.get_num_threads(), 'scope': 'B=1 model forward; excludes I/O, transfer and sliding-window stitching'}
