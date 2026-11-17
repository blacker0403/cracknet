"""One synthetic optimizer update for a chosen model; no checkpoint is saved."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.cracktoponet import build_model
from losses.combined import CombinedLoss
from utils.config import load_config, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/tutorial/local_e8.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError('Use a new output JSON to preserve earlier probes.')
    cfg = validate(load_config(args.config))
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['train']['cpu_threads'])
    device = torch.device(cfg['train']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use the CPU smoke route first.')
    model = build_model(cfg['model'], initialize=False).to(device).train()
    criterion = CombinedLoss(cfg['loss']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    enabled = cfg['train']['amp'] and device.type == 'cuda'
    dtype = getattr(torch, cfg['train'].get('amp_dtype', 'float16'))
    scaler = torch.amp.GradScaler('cuda', enabled=enabled and dtype == torch.float16, init_scale=1024)
    batch, size, accumulation = cfg['train']['batch_size'], cfg['data']['crop_size'], cfg['train']['accumulation_steps']
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(accumulation):
        image = torch.randn(batch, 3, size, size, device=device)
        target = torch.zeros(batch, 1, size, size, device=device)
        target[:, :, size // 2:size // 2 + 2, size // 8:7 * size // 8] = 1
        with torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
            output = model(image)
            loss, terms = criterion(output, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss.')
        scaler.scale(loss / accumulation).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['clip_grad'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients; this probe did not succeed.')
    scaler.step(optimizer)
    scaler.update()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    report = {'passed': True, 'synthetic_only': True, 'optimizer_updates': 1, 'checkpoint_written': False,
              'initialization': 'random, no ImageNet download', 'scaler_initial_scale': 1024,
              'batch_size': batch, 'accumulation_steps': accumulation, 'effective_batch_size': batch * accumulation,
              'crop_size': size, 'device': str(device), 'amp': enabled, 'amp_dtype': str(dtype),
              'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm), 'seconds': time.perf_counter() - start,
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else None,
              'limitation': 'Synthetic update only; real-data distribution, long-run memory and model quality are not verified.'}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
