# -*- coding: utf-8 -*-
"""
COSPNet Stage1 测试 / 组件检索 / 恢复脚本（对齐最新版 Stage1 训练逻辑）

核心对齐点：
1) 与最新版 train_stage1.py 一致，生成路径改为：
      comp_codebook(comp_id) -> mapping_net(code) -> stylegan/stylegan_ema
   不再假设旧版固定风格向量 w_base。
2) 优先使用 checkpoint 中的 stylegan_ema 作为测试生成器（质量更稳定）；
   若不存在，则退回 stylegan。
3) 兼容旧 checkpoint：
   - 若存在 mapping_net：走新路径
   - 若不存在 mapping_net 但存在 w_base：自动退回旧路径
4) 保留两种模式：
   - 随机采样模式：导出组件生成结果网格
   - 检索恢复模式：读取 segment_results.json，对 query 组件做 top-k 原型检索并重建整字
"""

import os
import sys
import json
import math
import argparse
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

try:
    from component_stylegan import ComponentCodebook, ComponentStyleGAN, MappingNetwork
except Exception:
    from models.component_stylegan import ComponentCodebook, ComponentStyleGAN, MappingNetwork


def set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def dict_to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [dict_to_ns(v) for v in x]
    return x


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_cfg(path: str, checkpoint: Optional[dict] = None):
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.yaml', '.yml']:
                return dict_to_ns(yaml.safe_load(f))
            if ext == '.json':
                return dict_to_ns(json.load(f))
            raise ValueError(f'Unsupported config format: {ext}')

    if checkpoint is not None:
        for key in ['cfg', 'config', 'args', 'train_cfg']:
            if key in checkpoint:
                val = checkpoint[key]
                if isinstance(val, SimpleNamespace):
                    return val
                if isinstance(val, dict):
                    return dict_to_ns(val)

    raise ValueError('未找到配置，请传 --config。')


def flexible_state_dict_load(module, state_dict, module_name='module'):
    if module is None or state_dict is None:
        return False
    try:
        module.load_state_dict(state_dict, strict=True)
        print(f'[OK] 严格加载 {module_name}')
        return True
    except Exception as e1:
        print(f'[Warn] 严格加载 {module_name} 失败: {e1}')

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        cleaned[nk] = v

    try:
        missing, unexpected = module.load_state_dict(cleaned, strict=False)
        print(f'[OK] 非严格加载 {module_name}')
        print(f'       missing keys   = {len(missing)}')
        print(f'       unexpected keys= {len(unexpected)}')
        return True
    except Exception as e2:
        print(f'[Warn] 非严格加载 {module_name} 仍失败: {e2}')
        return False


def tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f'输出张量维度不对: {tuple(x.shape)}')
    if x.shape[1] not in [1, 3]:
        x = x[:, :1]
    if float(x.min()) < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def make_grid_pil(batch: torch.Tensor, nrow: int = 8, pad: int = 4, bg: int = 255) -> Image.Image:
    batch = tensor_to_vis(batch)
    b, c, h, w = batch.shape
    nrow = max(1, min(nrow, b))
    ncol = int(math.ceil(b / nrow))
    grid_w = nrow * w + (nrow + 1) * pad
    grid_h = ncol * h + (ncol + 1) * pad
    canvas = Image.new('L' if c == 1 else 'RGB', (grid_w, grid_h), color=bg)
    for i in range(b):
        r = i // nrow
        col = i % nrow
        x0 = pad + col * (w + pad)
        y0 = pad + r * (h + pad)
        arr = batch[i].numpy()
        if c == 1:
            pil = Image.fromarray((arr[0] * 255.0).astype(np.uint8), mode='L')
        else:
            pil = Image.fromarray((np.transpose(arr, (1, 2, 0)) * 255.0).astype(np.uint8), mode='RGB')
        canvas.paste(pil, (x0, y0))
    return canvas


def save_grid(batch: torch.Tensor, save_path: str, nrow: int = 8, title: Optional[str] = None):
    img = make_grid_pil(batch, nrow=nrow)
    if title:
        title_h = 32
        canvas = Image.new(img.mode, (img.width, img.height + title_h), color=255)
        canvas.paste(img, (0, title_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, fill=0)
        img = canvas
    img.save(save_path)
    print(f'[OK] 保存图片: {save_path}')


def read_gray_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def pil_from_gray(img: np.ndarray) -> Image.Image:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(img.astype(np.uint8), mode='L')


def auto_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float((bw > 0).mean())
    fg = 255 - bw if white_ratio > 0.5 else bw.copy()
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fg


def normalize_mask(img: np.ndarray, size: int = 64, pad: int = 8) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    fg = auto_binarize(gray)
    ys, xs = np.where(fg > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    crop = fg[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = min((size - pad) / max(1, w), (size - pad) / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def fill_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def hu_log_features(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu


def shape_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum() + 1e-6
    iou = inter / union
    ha = hu_log_features(mask_a)
    hb = hu_log_features(mask_b)
    hu_dist = np.mean(np.abs(ha - hb))
    hu_score = math.exp(-hu_dist / 2.5)
    fa = fill_ratio(mask_a)
    fb = fill_ratio(mask_b)
    fill_score = max(0.0, 1.0 - abs(fa - fb) * 3.0)
    return float(0.60 * iou + 0.25 * hu_score + 0.15 * fill_score)


def cosine_similarity_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a.astype(np.float32).reshape(-1) / 255.0)
    b = (mask_b.astype(np.float32).reshape(-1) / 255.0)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def aspect_ratio_from_bbox(bbox: Sequence[int]) -> float:
    _, _, w, h = [int(v) for v in bbox]
    return float(w) / max(1.0, float(h))


def crop_foreground(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    fg = auto_binarize(gray)
    ys, xs = np.where(fg > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return gray[y1:y2, x1:x2].copy()


def save_gray(path: str, img: np.ndarray):
    ensure_dir(os.path.dirname(path) or '.')
    cv2.imwrite(path, img)


def draw_bbox_overlay(gray: np.ndarray, components_out: List[Dict[str, Any]]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for comp in components_out:
        x, y, w, h = [int(v) for v in comp['bbox']]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 255), 2)
        top1 = comp['topk'][0] if comp.get('topk') else None
        label = f'#{comp.get("index", 0)}'
        if top1 is not None:
            label += f' id={int(top1["comp_id"])} s={float(top1["score"]):.3f}'
        cv2.putText(vis, label, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def paste_component(canvas: np.ndarray, comp_img: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return canvas
    fg_crop = crop_foreground(comp_img)
    if fg_crop is None or fg_crop.size == 0:
        return canvas
    resized = cv2.resize(fg_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    y2 = min(canvas.shape[0], y + h)
    x2 = min(canvas.shape[1], x + w)
    resized = resized[:max(0, y2 - y), :max(0, x2 - x)]
    if resized.size == 0:
        return canvas
    canvas[y:y2, x:x2] = np.maximum(canvas[y:y2, x:x2], resized)
    return canvas


def load_segmentation_results(seg_json: str) -> List[Dict]:
    with open(seg_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if 'components' in data and 'image_path' in data:
            return [data]
        if 'results' in data and isinstance(data['results'], list):
            return data['results']
        if 'items' in data and isinstance(data['items'], list):
            return data['items']
        raise ValueError('无法识别的 segment json 格式（dict）')
    if isinstance(data, list):
        return data
    raise ValueError('无法识别的 segment json 格式')


def filter_segmentation_items(items: List[Dict], image_name: Optional[str] = None, image_path: Optional[str] = None):
    out = []
    for item in items:
        ok = True
        if image_name is not None and item.get('image_name') != image_name:
            ok = False
        if image_path is not None and os.path.abspath(item.get('image_path', '')) != os.path.abspath(image_path):
            ok = False
        if ok:
            out.append(item)
    return out


def load_query_patch_and_mask(component: Dict, parent_item: Dict, query_size: int = 128) -> Tuple[np.ndarray, np.ndarray, str]:
    bbox = [int(v) for v in component['bbox']]
    x, y, w, h = bbox

    for key in ['saved_mask', 'mask_path']:
        p = component.get(key)
        if p and os.path.isfile(p):
            m = read_gray_image(p)
            if m is not None:
                patch_img = m.copy()
                query_mask = normalize_mask(m, size=64)
                patch_show = cv2.resize(normalize_mask(patch_img, size=query_size), (query_size, query_size), interpolation=cv2.INTER_NEAREST)
                return patch_show, query_mask, key

    mask_path = parent_item.get('binary_mask')
    if mask_path and os.path.isfile(mask_path):
        fg = read_gray_image(mask_path)
        if fg is not None:
            crop = fg[y:y + h, x:x + w].copy()
            if crop.size > 0 and np.any(crop > 0):
                query_mask = normalize_mask(crop, size=64)
                patch_show = cv2.resize(normalize_mask(crop, size=query_size), (query_size, query_size), interpolation=cv2.INTER_NEAREST)
                return patch_show, query_mask, 'binary_mask_crop'

    patch_path = component.get('saved_patch')
    if patch_path and os.path.isfile(patch_path):
        patch = read_gray_image(patch_path)
        if patch is not None:
            query_mask = normalize_mask(patch, size=64)
            patch_show = cv2.resize(normalize_mask(patch, size=query_size), (query_size, query_size), interpolation=cv2.INTER_NEAREST)
            return patch_show, query_mask, 'saved_patch'

    image_path = parent_item.get('image_path')
    gray = read_gray_image(image_path)
    if gray is None:
        raise ValueError(f'无法读取 query patch，也无法读取原图: {image_path}')
    patch = gray[y:y + h, x:x + w].copy()
    query_mask = normalize_mask(patch, size=64)
    patch_show = cv2.resize(normalize_mask(patch, size=query_size), (query_size, query_size), interpolation=cv2.INTER_NEAREST)
    return patch_show, query_mask, 'image_crop'


def infer_num_components_from_ckpt(ckpt: dict, cfg: Optional[SimpleNamespace] = None) -> Optional[int]:
    if cfg is not None:
        try:
            if int(cfg.model.num_components) > 1:
                return int(cfg.model.num_components)
        except Exception:
            pass
    sd = ckpt.get('comp_codebook')
    if isinstance(sd, dict):
        for _, v in sd.items():
            if torch.is_tensor(v) and v.ndim == 2:
                return int(v.shape[0])
    return None


def _strip_module_prefix(state_dict: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    if not isinstance(state_dict, dict):
        return cleaned
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned[new_key] = value
    return cleaned


def infer_generator_options_from_state_dict(state_dict: Optional[Dict[str, torch.Tensor]]) -> Dict[str, bool]:
    cleaned = _strip_module_prefix(state_dict)
    final_conv0 = cleaned.get('final_conv.0.weight')
    final_conv2 = cleaned.get('final_conv.2.weight')
    return {
        'use_noise_in_blocks': any(
            '.noise1.weight' in key or '.noise2.weight' in key
            for key in cleaned.keys()
        ),
        'legacy_output_head': (
            torch.is_tensor(final_conv0)
            and final_conv0.ndim == 4
            and int(final_conv0.shape[1]) == 1
        ) or (
            torch.is_tensor(final_conv2)
            and final_conv2.ndim == 4
            and tuple(int(v) for v in final_conv2.shape[2:]) == (3, 3)
        ),
    }


def infer_generator_options_from_ckpt(ckpt: dict, use_ema: bool = True) -> Dict[str, bool]:
    if use_ema and isinstance(ckpt.get('stylegan_ema'), dict) and len(ckpt['stylegan_ema']) > 0:
        return infer_generator_options_from_state_dict(ckpt.get('stylegan_ema'))
    if isinstance(ckpt.get('stylegan'), dict) and len(ckpt['stylegan']) > 0:
        return infer_generator_options_from_state_dict(ckpt.get('stylegan'))
    return {
        'use_noise_in_blocks': False,
        'legacy_output_head': False,
    }


def build_models_direct(
    cfg,
    device: torch.device,
    num_components: int,
    generator_options: Optional[Dict[str, bool]] = None,
):
    codebook_dim = int(cfg.model.codebook_dim)
    w_dim = int(cfg.model.w_dim)
    generator_options = generator_options or {}
    comp_codebook = ComponentCodebook(num_components, codebook_dim).to(device)
    mapping_net = MappingNetwork(codebook_dim, w_dim).to(device)
    stylegan = ComponentStyleGAN(
        codebook_dim=codebook_dim,
        w_dim=w_dim,
        use_noise_in_blocks=bool(generator_options.get('use_noise_in_blocks', False)),
        legacy_output_head=bool(generator_options.get('legacy_output_head', False)),
    ).to(device)
    stylegan_ema = ComponentStyleGAN(
        codebook_dim=codebook_dim,
        w_dim=w_dim,
        use_noise_in_blocks=bool(generator_options.get('use_noise_in_blocks', False)),
        legacy_output_head=bool(generator_options.get('legacy_output_head', False)),
    ).to(device)
    return comp_codebook, mapping_net, stylegan, stylegan_ema


def expand_w_base(w_base: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if w_base.ndim == 1:
        w_base = w_base.unsqueeze(0)
    if w_base.shape[0] == 1:
        return w_base.to(device).expand(batch, -1)
    if w_base.shape[0] >= batch:
        return w_base[:batch].to(device)
    reps = int(math.ceil(batch / max(1, w_base.shape[0])))
    return w_base.to(device).repeat(reps, 1)[:batch]


def has_mapping_path(ckpt: dict) -> bool:
    return isinstance(ckpt.get('mapping_net'), dict) and len(ckpt.get('mapping_net', {})) > 0


def has_legacy_w_base(ckpt: dict) -> bool:
    return 'w_base' in ckpt and torch.is_tensor(ckpt['w_base'])


def generator_forward(
    comp_ids: torch.Tensor,
    comp_codebook,
    generator,
    mapping_net=None,
    legacy_w_base: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    codes = comp_codebook(comp_ids)
    if mapping_net is not None:
        code_flat = codes.squeeze(-1).squeeze(-1)
        w = mapping_net(code_flat)
    elif legacy_w_base is not None:
        if device is None:
            device = comp_ids.device
        w = expand_w_base(legacy_w_base, batch=comp_ids.shape[0], device=device)
    else:
        raise RuntimeError('既没有 mapping_net，也没有 legacy w_base，无法生成组件。')
    fake = generator(codes, w)
    return fake


def build_generator_bank(
    comp_codebook,
    generator,
    mapping_net,
    legacy_w_base: Optional[torch.Tensor],
    num_components: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for st in range(0, num_components, batch_size):
            ed = min(num_components, st + batch_size)
            comp_ids = torch.arange(st, ed, device=device, dtype=torch.long)
            fake = generator_forward(
                comp_ids=comp_ids,
                comp_codebook=comp_codebook,
                generator=generator,
                mapping_net=mapping_net,
                legacy_w_base=legacy_w_base,
                device=device,
            )
            outs.append(tensor_to_vis(fake).cpu())
    bank = torch.cat(outs, dim=0)
    print(f'[Info] generator bank shape = {tuple(bank.shape)}')
    return bank


def build_bank_meta(bank: torch.Tensor, mask_size: int = 64) -> List[Dict[str, Any]]:
    bank = tensor_to_vis(bank)
    meta: List[Dict[str, Any]] = []
    for idx in range(bank.shape[0]):
        arr = (bank[idx, 0].numpy() * 255.0).astype(np.uint8)
        mask = normalize_mask(arr, size=mask_size)
        ys, xs = np.where(mask > 0)
        aspect = 1.0
        if len(xs) > 0 and len(ys) > 0:
            ww = xs.max() - xs.min() + 1
            hh = ys.max() - ys.min() + 1
            aspect = float(ww) / max(1.0, float(hh))
        meta.append({
            'comp_id': idx,
            'img': arr,
            'mask': mask,
            'fill': fill_ratio(mask),
            'aspect': aspect,
            'blank': fill_ratio(mask) < 0.003,
        })
    return meta


def score_query_to_bank(query_mask: np.ndarray, query_bbox: Sequence[int], bank_meta: List[Dict[str, Any]], image_shape: Optional[Sequence[int]] = None, topk: int = 5):
    q_fill = fill_ratio(query_mask)
    q_aspect = aspect_ratio_from_bbox(query_bbox)
    q_box_ratio = None
    if image_shape is not None and len(image_shape) >= 2:
        ih, iw = int(image_shape[0]), int(image_shape[1])
        _, _, bw, bh = [int(v) for v in query_bbox]
        q_box_ratio = float((bw * bh) / max(1.0, iw * ih))

    results = []
    for item in bank_meta:
        bmask = item['mask']
        shape_score = shape_similarity(query_mask, bmask)
        cos_score = cosine_similarity_mask(query_mask, bmask)
        aspect_score = math.exp(-abs(math.log((q_aspect + 1e-6) / (item['aspect'] + 1e-6))))
        fill_score = max(0.0, 1.0 - abs(q_fill - item['fill']) * 2.5)

        score = 0.55 * shape_score + 0.20 * cos_score + 0.15 * aspect_score + 0.10 * fill_score
        if q_box_ratio is not None:
            size_score = max(0.0, 1.0 - abs(q_box_ratio * 4.0 - item['fill']) * 0.8)
            score = 0.90 * score + 0.10 * size_score
        else:
            size_score = None

        if item['blank']:
            score *= 0.15
        results.append({
            'comp_id': int(item['comp_id']),
            'score': float(score),
            'shape_score': float(shape_score),
            'cosine_score': float(cos_score),
            'aspect_score': float(aspect_score),
            'fill_score': float(fill_score),
            'size_score': None if size_score is None else float(size_score),
            'blank': bool(item['blank']),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:topk]


def render_component_topk_board(query_patch_vis: np.ndarray, retrieved: List[Dict[str, Any]], bank_meta: List[Dict[str, Any]], save_path: str):
    cards = [pil_from_gray(query_patch_vis)]
    labels = ['query']
    for rank, item in enumerate(retrieved, start=1):
        comp_id = int(item['comp_id'])
        img = cv2.resize(bank_meta[comp_id]['img'], (128, 128), interpolation=cv2.INTER_NEAREST)
        cards.append(pil_from_gray(img))
        labels.append(f'top{rank} | id={comp_id} | {item["score"]:.3f}')

    pad = 8
    w, h = 128, 128
    total_w = len(cards) * (w + pad) + pad
    total_h = h + 48
    canvas = Image.new('L', (total_w, total_h), color=255)
    draw = ImageDraw.Draw(canvas)
    for i, (card, label) in enumerate(zip(cards, labels)):
        x = pad + i * (w + pad)
        y = 24
        canvas.paste(card, (x, y))
        draw.rectangle([x, y, x + w, y + h], outline=0, width=1)
        draw.text((x, 4), label, fill=0)
    canvas.save(save_path)


def save_reconstructed_views(item: Dict[str, Any], components_out: List[Dict[str, Any]], bank_meta: List[Dict[str, Any]], out_dir: str, max_rank: int = 5):
    gray = read_gray_image(item['image_path'])
    if gray is None:
        raise ValueError(f'无法读取原图: {item["image_path"]}')

    H, W = gray.shape[:2]
    max_rank = max(1, max_rank)
    for rank in range(1, max_rank + 1):
        recon = np.zeros((H, W), dtype=np.uint8)
        for comp in sorted(components_out, key=lambda x: (x['bbox'][3] * x['bbox'][2], x['bbox'][1], x['bbox'][0])):
            topk = comp.get('topk', [])
            if len(topk) < rank:
                continue
            comp_id = int(topk[rank - 1]['comp_id'])
            recon = paste_component(recon, bank_meta[comp_id]['img'], comp['bbox'])
        save_gray(os.path.join(out_dir, f'reconstructed_rank{rank}.png'), recon)
        if rank == 1:
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            recon_col = cv2.cvtColor(recon, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(overlay, 0.65, recon_col, 0.85, 0)
            box_vis = draw_bbox_overlay(gray, components_out)
            cv2.imwrite(os.path.join(out_dir, 'reconstructed_overlay.png'), overlay)
            cv2.imwrite(os.path.join(out_dir, 'original_with_boxes.png'), box_vis)


def collect_real_images(real_dir: str, limit: int = 32, image_size: Optional[int] = None):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = []
    for root, _, names in os.walk(real_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() in exts:
                files.append(os.path.join(root, n))
    files = sorted(files)[:limit]
    imgs = []
    for fp in files:
        try:
            img = Image.open(fp).convert('L')
            if image_size is not None:
                img = ImageOps.fit(img, (image_size, image_size), method=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).unsqueeze(0))
        except Exception as e:
            print(f'[Warn] 读取真实图片失败: {fp}, error={e}')
    if not imgs:
        return None
    return torch.stack(imgs, dim=0)


def select_generator_name(ckpt: dict, use_ema: bool = True) -> str:
    if use_ema and isinstance(ckpt.get('stylegan_ema'), dict) and len(ckpt['stylegan_ema']) > 0:
        return 'stylegan_ema'
    if isinstance(ckpt.get('stylegan'), dict) and len(ckpt['stylegan']) > 0:
        return 'stylegan'
    raise RuntimeError('checkpoint 中既没有 stylegan_ema 也没有 stylegan。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='stage1_best.pth', help='Stage1 checkpoint 路径')
    parser.add_argument('--config', type=str, default='cospnet.yaml', help='训练配置 yaml/json')
    parser.add_argument('--outdir', type=str, default='test_outputs_cospnet', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=32, help='随机采样模式：生成样本数')
    parser.add_argument('--nrow', type=int, default=8, help='网格每行数量')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_components', type=int, default=None, help='可选：手动指定组件数')
    parser.add_argument('--real_dir', type=str, default=None, help='可选：真实图片目录，用于一起保存 real grid')
    parser.add_argument('--use_ema', action='store_true', help='优先使用 stylegan_ema（推荐）')
    parser.add_argument('--no_ema', action='store_true', help='强制不用 stylegan_ema，只用 stylegan')

    # 检索 / 恢复模式
    parser.add_argument('--segment_json', type=str, default=None, help='自动分割输出的 segment_results.json')
    parser.add_argument('--image_name', type=str, default=None, help='只处理某个 image_name')
    parser.add_argument('--image_path', type=str, default=None, help='只处理某个 image_path')
    parser.add_argument('--topk', type=int, default=5, help='检索 top-k')
    parser.add_argument('--bank_batch_size', type=int, default=64)
    parser.add_argument('--save_bank_preview', action='store_true', help='保存组件原型库预览图')
    args = parser.parse_args()

    if args.no_ema:
        args.use_ema = False
    elif not args.use_ema:
        # 默认行为：如果有 ema 就用 ema
        args.use_ema = True

    ensure_dir(args.outdir)
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f'[Info] device = {device}')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError('checkpoint 不是 dict 格式，当前脚本无法解析。')
    print('[Info] checkpoint top-level keys:')
    for k in ckpt.keys():
        print('   -', k)

    cfg = load_cfg(args.config, ckpt)
    num_components = args.num_components or infer_num_components_from_ckpt(ckpt, cfg)
    if num_components is None:
        raise RuntimeError('无法推断 num_components，请手动传 --num_components。')
    print(f'[Info] num_components = {num_components}')

    generator_options = infer_generator_options_from_ckpt(ckpt, use_ema=args.use_ema)
    print(
        f'[Info] generator options = '
        f'use_noise_in_blocks={generator_options["use_noise_in_blocks"]}, '
        f'legacy_output_head={generator_options["legacy_output_head"]}'
    )
    comp_codebook, mapping_net, stylegan, stylegan_ema = build_models_direct(
        cfg,
        device,
        num_components,
        generator_options=generator_options,
    )

    ok_c = flexible_state_dict_load(comp_codebook, ckpt.get('comp_codebook'), 'comp_codebook')
    if not ok_c:
        raise RuntimeError('comp_codebook 权重加载失败。')

    # 新版：mapping_net
    use_mapping = has_mapping_path(ckpt)
    if use_mapping:
        ok_m = flexible_state_dict_load(mapping_net, ckpt.get('mapping_net'), 'mapping_net')
        if not ok_m:
            raise RuntimeError('mapping_net 权重加载失败。')
        legacy_w_base = None
        print('[Info] 生成路径: comp_codebook -> mapping_net -> stylegan')
    else:
        mapping_net = None
        if has_legacy_w_base(ckpt):
            legacy_w_base = ckpt['w_base'].detach().float().cpu()
            print(f'[Info] 生成路径: legacy w_base, shape = {tuple(legacy_w_base.shape)}')
        else:
            raise RuntimeError('checkpoint 中既没有 mapping_net，也没有 w_base，无法测试。')

    generator_name = select_generator_name(ckpt, use_ema=args.use_ema)
    if generator_name == 'stylegan_ema':
        ok_g = flexible_state_dict_load(stylegan_ema, ckpt.get('stylegan_ema'), 'stylegan_ema')
        if not ok_g:
            raise RuntimeError('stylegan_ema 权重加载失败。')
        generator = stylegan_ema
    else:
        ok_g = flexible_state_dict_load(stylegan, ckpt.get('stylegan'), 'stylegan')
        if not ok_g:
            raise RuntimeError('stylegan 权重加载失败。')
        generator = stylegan
    print(f'[Info] 当前测试生成器 = {generator_name}')

    comp_codebook.eval().to(device)
    if mapping_net is not None:
        mapping_net.eval().to(device)
    stylegan.eval().to(device)
    stylegan_ema.eval().to(device)
    generator.eval().to(device)

    # 模式 A：随机采样
    if args.segment_json is None:
        comp_ids = torch.arange(args.num_samples, device=device, dtype=torch.long) % num_components
        with torch.no_grad():
            fake = generator_forward(
                comp_ids=comp_ids,
                comp_codebook=comp_codebook,
                generator=generator,
                mapping_net=mapping_net,
                legacy_w_base=legacy_w_base if 'legacy_w_base' in locals() else None,
                device=device,
            )
            fake = tensor_to_vis(fake)

        print(f'[Info] fake shape = {tuple(fake.shape)}')
        print(f'[Info] fake min/max/mean/std = {float(fake.min()):.4f} / {float(fake.max()):.4f} / {float(fake.mean()):.4f} / {float(fake.std()):.4f}')
        save_grid(
            fake,
            os.path.join(args.outdir, 'fake_samples.png'),
            nrow=args.nrow,
            title=f'Generated samples from {os.path.basename(args.ckpt)} ({generator_name})'
        )

        if args.real_dir:
            real_batch = collect_real_images(args.real_dir, limit=args.num_samples, image_size=int(fake.shape[-1]))
            if real_batch is not None:
                save_grid(real_batch, os.path.join(args.outdir, 'real_samples.png'), nrow=args.nrow, title='Real samples')

        print('\n================= 测试完成（随机采样模式） =================')
        print(f'生成结果已保存到: {args.outdir}')
        return

    # 模式 B：检索 + 恢复
    bank = build_generator_bank(
        comp_codebook=comp_codebook,
        generator=generator,
        mapping_net=mapping_net,
        legacy_w_base=legacy_w_base if 'legacy_w_base' in locals() else None,
        num_components=num_components,
        device=device,
        batch_size=args.bank_batch_size,
    )
    bank_meta = build_bank_meta(bank, mask_size=64)

    if args.save_bank_preview:
        preview_count = min(64, bank.shape[0])
        save_grid(
            bank[:preview_count],
            os.path.join(args.outdir, 'prototype_bank_preview.png'),
            nrow=8,
            title=f'Prototype bank preview ({generator_name})'
        )

    seg_items = load_segmentation_results(args.segment_json)
    seg_items = filter_segmentation_items(seg_items, image_name=args.image_name, image_path=args.image_path)
    if not seg_items:
        raise RuntimeError('segment_json 中没有找到匹配的图像项。')

    all_results = []
    for item in seg_items:
        image_name = item.get('image_name', os.path.splitext(os.path.basename(item.get('image_path', 'image')))[0])
        image_dir = os.path.join(args.outdir, image_name)
        ensure_dir(image_dir)
        ensure_dir(os.path.join(image_dir, 'components'))

        comps_out = []
        image_shape = item.get('image_shape')
        for comp in item.get('components', []):
            idx = int(comp.get('index', len(comps_out)))
            bbox = [int(v) for v in comp['bbox']]
            query_patch_vis, query_mask, query_source = load_query_patch_and_mask(comp, item, query_size=128)
            topk_items = score_query_to_bank(query_mask, bbox, bank_meta, image_shape=image_shape, topk=args.topk)

            comp_dir = os.path.join(image_dir, 'components', f'comp_{idx:03d}')
            ensure_dir(comp_dir)
            save_gray(os.path.join(comp_dir, 'query_patch.png'), query_patch_vis)
            save_gray(os.path.join(comp_dir, 'query_mask_norm64.png'), query_mask)
            for rank, cand in enumerate(topk_items, start=1):
                cid = int(cand['comp_id'])
                save_gray(os.path.join(comp_dir, f'top{rank}_id_{cid:04d}.png'), bank_meta[cid]['img'])

            render_component_topk_board(query_patch_vis, topk_items, bank_meta, os.path.join(comp_dir, 'topk_board.png'))

            qmeta = {
                'bbox': bbox,
                'source': query_source,
            }
            if image_shape is not None and len(image_shape) >= 2:
                ih, iw = int(image_shape[0]), int(image_shape[1])
                x, y, w, h = bbox
                qmeta['center_norm'] = [float((x + 0.5 * w) / max(1.0, iw)), float((y + 0.5 * h) / max(1.0, ih))]
                qmeta['area_ratio'] = float((w * h) / max(1.0, iw * ih))
                qmeta['aspect'] = float(w / max(1.0, h))

            comps_out.append({
                'index': idx,
                'bbox': bbox,
                'saved_patch': comp.get('saved_patch'),
                'query': qmeta,
                'topk': topk_items,
            })

        save_reconstructed_views(item, comps_out, bank_meta, image_dir, max_rank=min(args.topk, 5))

        img_result = {
            'image_name': image_name,
            'image_path': item.get('image_path'),
            'image_shape': item.get('image_shape'),
            'num_components': len(comps_out),
            'bank_mode': generator_name,
            'uses_mapping_net': bool(mapping_net is not None),
            'components': comps_out,
        }
        all_results.append(img_result)

        with open(os.path.join(image_dir, 'retrieval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(img_result, f, indent=2, ensure_ascii=False)
        print(f'[OK] 完成检索与恢复: {image_name}, num_components={len(comps_out)}')

    with open(os.path.join(args.outdir, 'retrieval_results_all.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print('\n================= 测试完成（检索 + 恢复模式） =================')
    print(f'输出目录: {args.outdir}')
    print('包含内容：')
    print('1) 每个组件的 query patch / query mask / top-k 候选图')
    print('2) 每张图的 reconstructed_rank1~rank5.png')
    print('3) reconstructed_overlay.png 与 original_with_boxes.png')
    print('4) retrieval_results.json / retrieval_results_all.json')


if __name__ == '__main__':
    main()
