"""Component retrieval and restoration wrapper used by the unified pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import RestorationConfig
from .instance_restoration import run_instance_component_restoration
from .io_utils import ensure_dir, resolve_path, write_json
from .legacy_loader import load_retrieval_module


@dataclass
class RestorationBundle:
    module: Any
    device: torch.device
    cfg: Any
    generator_name: str
    num_components: int
    comp_codebook: Any
    mapping_net: Any
    stylegan: Any
    stylegan_ema: Any
    generator: Any
    legacy_w_base: Any
    bank: torch.Tensor
    bank_meta: list[dict[str, Any]]
    checkpoint_path: Path
    config_path: Path


def _load_bundle(
    config: RestorationConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> RestorationBundle:
    module = load_retrieval_module()
    ckpt_path = resolve_path(config.ckpt_path, search_roots=search_roots, base_dir=base_dir)
    cfg_path = resolve_path(config.config_path, search_roots=search_roots, base_dir=base_dir)

    module.set_seed(config.seed)
    device_name = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[restoration] device={device} ckpt={ckpt_path}")
    print(f"[restoration] stage1_config={cfg_path}")

    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load stage1 checkpoint: {ckpt_path}. "
            "The file may be incomplete or corrupted. "
            "If your full checkpoint lives on the remote server, point ckpt_path there."
        ) from exc
    try:
        cfg = module.load_cfg("", checkpoint)
        print("[restoration] cfg_source=checkpoint")
    except Exception:
        cfg = module.load_cfg(str(cfg_path), checkpoint)
        print(f"[restoration] cfg_source={cfg_path}")
    num_components = config.num_components or module.infer_num_components_from_ckpt(checkpoint, cfg)
    if num_components is None:
        raise RuntimeError("Could not infer num_components from the stage1 checkpoint.")

    generator_options = {}
    if hasattr(module, "infer_generator_options_from_ckpt"):
        generator_options = module.infer_generator_options_from_ckpt(
            checkpoint,
            use_ema=config.use_ema,
        )
        print(
            "[restoration] generator_options="
            f"use_noise_in_blocks={generator_options.get('use_noise_in_blocks', False)} "
            f"legacy_output_head={generator_options.get('legacy_output_head', False)}"
        )

    comp_codebook, mapping_net, stylegan, stylegan_ema = module.build_models_direct(
        cfg,
        device,
        num_components,
        generator_options=generator_options,
    )

    if not module.flexible_state_dict_load(
        comp_codebook,
        checkpoint.get("comp_codebook"),
        "comp_codebook",
    ):
        raise RuntimeError("Failed to load comp_codebook weights.")

    uses_mapping_net = module.has_mapping_path(checkpoint)
    if uses_mapping_net:
        if not module.flexible_state_dict_load(
            mapping_net,
            checkpoint.get("mapping_net"),
            "mapping_net",
        ):
            raise RuntimeError("Failed to load mapping_net weights.")
        legacy_w_base = None
    else:
        mapping_net = None
        if not module.has_legacy_w_base(checkpoint):
            raise RuntimeError("Checkpoint contains neither mapping_net nor legacy w_base.")
        legacy_w_base = checkpoint["w_base"].detach().float().cpu()

    generator_name = module.select_generator_name(checkpoint, use_ema=config.use_ema)
    if generator_name == "stylegan_ema":
        if not module.flexible_state_dict_load(
            stylegan_ema,
            checkpoint.get("stylegan_ema"),
            "stylegan_ema",
        ):
            if isinstance(checkpoint.get("stylegan"), dict) and len(checkpoint["stylegan"]) > 0:
                print("[restoration] fallback_generator=stylegan")
                if not module.flexible_state_dict_load(
                    stylegan,
                    checkpoint.get("stylegan"),
                    "stylegan",
                ):
                    raise RuntimeError("Failed to load both stylegan_ema and stylegan weights.")
                generator_name = "stylegan"
                generator = stylegan
            else:
                raise RuntimeError("Failed to load stylegan_ema weights.")
        else:
            generator = stylegan_ema
    else:
        if not module.flexible_state_dict_load(
            stylegan,
            checkpoint.get("stylegan"),
            "stylegan",
        ):
            raise RuntimeError("Failed to load stylegan weights.")
        generator = stylegan

    comp_codebook.eval().to(device)
    if mapping_net is not None:
        mapping_net.eval().to(device)
    stylegan.eval().to(device)
    stylegan_ema.eval().to(device)
    generator.eval().to(device)

    bank = module.build_generator_bank(
        comp_codebook=comp_codebook,
        generator=generator,
        mapping_net=mapping_net,
        legacy_w_base=legacy_w_base,
        num_components=num_components,
        device=device,
        batch_size=config.bank_batch_size,
    )
    bank_meta = module.build_bank_meta(bank, mask_size=64)
    print(f"[restoration] prototype_bank_ready count={len(bank_meta)}")

    return RestorationBundle(
        module=module,
        device=device,
        cfg=cfg,
        generator_name=generator_name,
        num_components=num_components,
        comp_codebook=comp_codebook,
        mapping_net=mapping_net,
        stylegan=stylegan,
        stylegan_ema=stylegan_ema,
        generator=generator,
        legacy_w_base=legacy_w_base,
        bank=bank,
        bank_meta=bank_meta,
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
    )


def run_component_restoration(
    segment_json_path: str | Path,
    output_dir: str | Path,
    config: RestorationConfig,
    *,
    search_roots: list[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict:
    resolved_ckpt_path = resolve_path(
        config.ckpt_path,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    checkpoint_type = None
    try:
        checkpoint_head = torch.load(str(resolved_ckpt_path), map_location="cpu")
        if isinstance(checkpoint_head, dict):
            checkpoint_type = checkpoint_head.get("checkpoint_type")
    except Exception:
        checkpoint_type = None

    if checkpoint_type == "instance_retrieval_restorer_v1":
        return run_instance_component_restoration(
            segment_json_path,
            output_dir,
            config,
            search_roots=search_roots,
            base_dir=base_dir,
        )

    resolved_segment_json = resolve_path(
        segment_json_path,
        search_roots=search_roots,
        base_dir=base_dir,
    )
    resolved_output = ensure_dir(output_dir)
    print(f"[restoration] segment_json={resolved_segment_json}")
    bundle = _load_bundle(config, search_roots=search_roots, base_dir=base_dir)
    module = bundle.module

    if config.save_bank_preview:
        preview_count = min(64, bundle.bank.shape[0])
        module.save_grid(
            bundle.bank[:preview_count],
            str(resolved_output / "prototype_bank_preview.png"),
            nrow=8,
            title=f"Prototype bank preview ({bundle.generator_name})",
        )

    segmentation_items = module.load_segmentation_results(str(resolved_segment_json))
    segmentation_items = module.filter_segmentation_items(
        segmentation_items,
        image_name=config.image_name,
        image_path=config.image_path,
    )
    if not segmentation_items:
        raise RuntimeError("No segmentation items matched the restoration filters.")
    print(f"[restoration] total_images={len(segmentation_items)} output_dir={resolved_output}")

    all_results = []
    manifest_items = []
    for image_index, item in enumerate(segmentation_items, start=1):
        image_name = item.get("image_name")
        if not image_name:
            image_name = Path(item.get("image_path", "image")).stem
        print(f"[restoration] ({image_index}/{len(segmentation_items)}) image={image_name}")

        image_dir = ensure_dir(resolved_output / image_name)
        component_root = ensure_dir(image_dir / "components")
        components_out = []
        image_shape = item.get("image_shape")

        components = item.get("components", [])
        print(f"[restoration] image={image_name} components={len(components)}")
        for component_index, component in enumerate(components, start=1):
            idx = int(component.get("index", len(components_out)))
            bbox = [int(value) for value in component["bbox"]]
            print(
                f"[restoration] image={image_name} component=({component_index}/{len(components)}) "
                f"bbox={bbox}"
            )
            query_patch_vis, query_mask, query_source = module.load_query_patch_and_mask(
                component,
                item,
                query_size=128,
            )
            topk_items = module.score_query_to_bank(
                query_mask,
                bbox,
                bundle.bank_meta,
                image_shape=image_shape,
                topk=config.topk,
            )

            component_dir = ensure_dir(component_root / f"comp_{idx:03d}")
            module.save_gray(str(component_dir / "query_patch.png"), query_patch_vis)
            module.save_gray(str(component_dir / "query_mask_norm64.png"), query_mask)
            for rank, candidate in enumerate(topk_items, start=1):
                comp_id = int(candidate["comp_id"])
                module.save_gray(
                    str(component_dir / f"top{rank}_id_{comp_id:04d}.png"),
                    bundle.bank_meta[comp_id]["img"],
                )

            module.render_component_topk_board(
                query_patch_vis,
                topk_items,
                bundle.bank_meta,
                str(component_dir / "topk_board.png"),
            )

            query_meta = {
                "bbox": bbox,
                "source": query_source,
            }
            if image_shape is not None and len(image_shape) >= 2:
                image_h = int(image_shape[0])
                image_w = int(image_shape[1])
                x, y, w, h = bbox
                query_meta["center_norm"] = [
                    float((x + 0.5 * w) / max(1.0, image_w)),
                    float((y + 0.5 * h) / max(1.0, image_h)),
                ]
                query_meta["area_ratio"] = float((w * h) / max(1.0, image_w * image_h))
                query_meta["aspect"] = float(w / max(1.0, h))

            components_out.append(
                {
                    "index": idx,
                    "bbox": bbox,
                    "saved_patch": component.get("saved_patch"),
                    "query": query_meta,
                    "topk": topk_items,
                }
            )

        module.save_reconstructed_views(
            item,
            components_out,
            bundle.bank_meta,
            str(image_dir),
            max_rank=min(config.topk, 5),
        )

        reconstructed_rank_paths = {
            f"rank_{rank}": str((image_dir / f"reconstructed_rank{rank}.png").resolve())
            for rank in range(1, min(config.topk, 5) + 1)
            if (image_dir / f"reconstructed_rank{rank}.png").exists()
        }
        overlay_path = image_dir / "reconstructed_overlay.png"
        box_path = image_dir / "original_with_boxes.png"

        image_result = {
            "image_name": image_name,
            "image_path": item.get("image_path"),
            "image_shape": item.get("image_shape"),
            "num_components": len(components_out),
            "bank_mode": bundle.generator_name,
            "uses_mapping_net": bool(bundle.mapping_net is not None),
            "paths": {
                "image_dir": str(image_dir.resolve()),
                "retrieval_results": str((image_dir / "retrieval_results.json").resolve()),
                "reconstructed_ranks": reconstructed_rank_paths,
                "reconstructed_overlay": str(overlay_path.resolve()) if overlay_path.exists() else None,
                "original_with_boxes": str(box_path.resolve()) if box_path.exists() else None,
            },
            "components": components_out,
        }
        all_results.append(image_result)
        write_json(image_result, image_dir / "retrieval_results.json")
        manifest_items.append(
            {
                "image_name": image_name,
                "image_path": item.get("image_path"),
                "image_dir": str(image_dir.resolve()),
                "retrieval_results": str((image_dir / "retrieval_results.json").resolve()),
                "reconstructed_rank1": reconstructed_rank_paths.get("rank_1"),
            }
        )
    print(f"[restoration] completed count={len(all_results)}")

    all_results_path = write_json(all_results, resolved_output / "retrieval_results_all.json")
    manifest = {
        "stage": "restoration",
        "device": str(bundle.device),
        "segment_json": str(resolved_segment_json),
        "output_dir": str(resolved_output),
        "checkpoint_path": str(bundle.checkpoint_path.resolve()),
        "config_path": str(bundle.config_path.resolve()),
        "generator_name": bundle.generator_name,
        "num_components": bundle.num_components,
        "count": len(all_results),
        "results_path": str(all_results_path),
        "items": manifest_items,
    }
    write_json(manifest, resolved_output / "manifest.json")
    return manifest
