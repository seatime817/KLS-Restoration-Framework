"""Command line entrypoint for the unified pipeline."""

from __future__ import annotations

import argparse
import json

from .config import load_pipeline_config
from .learned_refinement_training import run_learned_refinement_training
from .pipeline import KhitanRestorationPipeline
from .refinement import run_refinement
from .restoration import run_component_restoration
from .instance_restoration_training import run_instance_restoration_training
from .segmentation import run_segmentation
from .stage1_training import run_stage1_training
from .super_resolution import run_super_resolution


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified blurry Khitan text restoration pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline.")
    pipeline_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    pipeline_parser.add_argument("--input", type=str, default=None, help="Override input path.")
    pipeline_parser.add_argument("--output", type=str, default=None, help="Override output root.")

    super_parser = subparsers.add_parser("super-resolve", help="Run SwinIR only.")
    super_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    super_parser.add_argument("--input", type=str, required=True, help="Input image or directory.")
    super_parser.add_argument("--output", type=str, required=True, help="Output directory.")
    super_parser.add_argument("--model-path", type=str, default=None, help="Optional SwinIR model path override.")

    segment_parser = subparsers.add_parser("segment", help="Run segmentation only.")
    segment_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    segment_parser.add_argument("--input", type=str, required=True, help="Input image or directory.")
    segment_parser.add_argument("--output", type=str, required=True, help="Output directory.")

    restore_parser = subparsers.add_parser(
        "restore-components",
        help="Run component retrieval and reconstruction only.",
    )
    restore_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    restore_parser.add_argument("--segment-json", type=str, required=True, help="segment_results.json path.")
    restore_parser.add_argument("--output", type=str, required=True, help="Output directory.")
    restore_parser.add_argument("--ckpt", type=str, default=None, help="Stage1 checkpoint override.")
    restore_parser.add_argument("--stage1-config", type=str, default=None, help="Stage1 YAML/JSON override.")
    restore_parser.add_argument("--topk", type=int, default=None, help="Override top-k candidates.")

    refine_parser = subparsers.add_parser(
        "refine",
        help="Fuse SwinIR SR output with GAN-generated priors into the final text image.",
    )
    refine_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    refine_parser.add_argument(
        "--restoration-input",
        type=str,
        required=True,
        help="Restoration output directory or retrieval_results_all.json path.",
    )
    refine_parser.add_argument("--output", type=str, required=True, help="Output directory.")

    train_stage1_parser = subparsers.add_parser(
        "train-stage1",
        help="Train Stage1 component codebook + StyleGAN checkpoint.",
    )
    train_stage1_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    train_stage1_parser.add_argument("--output", type=str, default=None, help="Override output root.")
    train_stage1_parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")

    train_instance_parser = subparsers.add_parser(
        "train-instance-restorer",
        help="Train instance retrieval + conditional restoration checkpoint.",
    )
    train_instance_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    train_instance_parser.add_argument("--output", type=str, default=None, help="Override output root.")
    train_instance_parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")

    train_learned_refine_parser = subparsers.add_parser(
        "train-learned-refinement",
        help="Train learned ROI-based SR/prior fusion checkpoint.",
    )
    train_learned_refine_parser.add_argument("--config", type=str, default=None, help="YAML or JSON config path.")
    train_learned_refine_parser.add_argument("--output", type=str, default=None, help="Override output root.")
    train_learned_refine_parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train-stage1":
        result = run_stage1_training(
            args.config,
            output_root=args.output,
            resume_path=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "train-instance-restorer":
        result = run_instance_restoration_training(
            args.config,
            output_root=args.output,
            resume_path=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "train-learned-refinement":
        result = run_learned_refinement_training(
            args.config,
            output_root=args.output,
            resume_path=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    config = load_pipeline_config(args.config)

    if args.command == "pipeline":
        pipeline = KhitanRestorationPipeline(config)
        result = pipeline.run(input_path=args.input, output_root=args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    search_roots = list(config.search_roots)
    base_dir = config.config_base_dir
    if base_dir:
        search_roots.insert(0, base_dir)

    if args.command == "super-resolve":
        if args.model_path:
            config.super_resolution.model_source = "custom"
            config.super_resolution.custom_model_path = args.model_path
            config.super_resolution.model_path = args.model_path
        result = run_super_resolution(
            args.input,
            args.output,
            config.super_resolution,
            search_roots=search_roots,
            base_dir=base_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "segment":
        result = run_segmentation(
            args.input,
            args.output,
            config.segmentation,
            search_roots=search_roots,
            base_dir=base_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "restore-components":
        if args.ckpt:
            config.restoration.ckpt_path = args.ckpt
        if args.stage1_config:
            config.restoration.config_path = args.stage1_config
        if args.topk is not None:
            config.restoration.topk = args.topk
        result = run_component_restoration(
            args.segment_json,
            args.output,
            config.restoration,
            search_roots=search_roots,
            base_dir=base_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "refine":
        result = run_refinement(
            args.restoration_input,
            args.output,
            config.refinement,
            search_roots=search_roots,
            base_dir=base_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
