import torch
from torch.nn import functional as F


def positions(length, window, stride):
    if length <= window:
        return [0]
    return sorted(set(list(range(0, length - window + 1, stride)) + [length - window]))


@torch.inference_mode()
def sliding_window(model, image, window=512, overlap=128, device='cpu', amp=False, amp_dtype='float16', batch_size=1):
    """Single normalized CHW image -> native-size CPU probability maps, no labels."""
    if image.ndim != 3 or not 0 <= overlap < window or window < 32 or batch_size < 1:
        raise ValueError('Expect CHW image, window >=32 and 0 <= overlap < window.')
    device = torch.device(device)
    h, w = image.shape[-2:]
    weight = torch.hann_window(window, periodic=False).clamp_min(0.01)
    weight = weight[:, None] * weight[None, :]
    denominator = torch.zeros(h, w)
    numerator, edge_numerator = torch.zeros(h, w), None
    was_training = model.training
    model.eval()
    try:
        tiles = [(top, left) for top in positions(h, window, window - overlap)
                 for left in positions(w, window, window - overlap)]
        for start in range(0, len(tiles), batch_size):
            locations, patches, sizes = tiles[start:start + batch_size], [], []
            for top, left in locations:
                patch = image[:, top:top + window, left:left + window]
                ph, pw = patch.shape[-2:]
                patches.append(F.pad(patch, (0, window - pw, 0, window - ph), mode='replicate'))
                sizes.append((ph, pw))
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda',
                                dtype=getattr(torch, amp_dtype)):
                out = model(torch.stack(patches).to(device))
            logits = out['logits'] if isinstance(out, dict) else out
            probabilities = logits.float().sigmoid()[:, 0].cpu()
            edges = None
            if isinstance(out, dict) and out.get('edge_logits') is not None:
                edge = F.interpolate(out['edge_logits'].float(), size=(window, window), mode='bilinear', align_corners=False)
                edges = edge.sigmoid()[:, 0].cpu()
                if edge_numerator is None:
                    edge_numerator = torch.zeros(h, w)
            for i, ((top, left), (ph, pw)) in enumerate(zip(locations, sizes)):
                prob = probabilities[i, :ph, :pw]
                weights = weight[:ph, :pw]
                numerator[top:top + ph, left:left + pw] += prob * weights
                denominator[top:top + ph, left:left + pw] += weights
                if edges is not None:
                    edge_numerator[top:top + ph, left:left + pw] += edges[i, :ph, :pw] * weights
    finally:
        model.train(was_training)
    return {'probability': numerator / denominator,
            'edge_probability': edge_numerator / denominator if edge_numerator is not None else None}
