# Khitan Restoration Project

This repository now contains a clean wrapper project around the existing experimental code.

## What The Unified Project Does

The new `khitan_restore` package exposes one pipeline with four stable layers:

1. SwinIR super-resolution for blurry input images
2. Component segmentation for the super-resolved character image
3. Component retrieval against the Stage1 codebook and GAN prototype bank
4. Prior-guided refinement that fuses the SwinIR result with the GAN-generated structure prior

The original experimental scripts are still kept in place. The new package uses them as backend implementations so the current research code is preserved while the project gets a cleaner interface.

## New Entry Points

Run the full pipeline:

```bash
python -m khitan_restore.cli pipeline --config configs/pipeline.example.yaml
```

The bundled sample input is:

`sample_data/input.png`

Run one stage at a time:

```bash
python -m khitan_restore.cli super-resolve --config configs/pipeline.example.yaml --input sample_data/input.png --output runs/sr_only
python -m khitan_restore.cli segment --config configs/pipeline.example.yaml --input runs/sr_only/images --output runs/seg_only
python -m khitan_restore.cli restore-components --config configs/pipeline.example.yaml --segment-json runs/seg_only/segment_results.json --output runs/restore_only
python -m khitan_restore.cli refine --config configs/pipeline.example.yaml --restoration-input runs/restore_only --output runs/final_only
```

There is also a convenience script:

```bash
python scripts/run_pipeline.py pipeline --config configs/pipeline.example.yaml
```

## Directory Layout

```text
khitan_restore/
  cli.py
  config.py
  io_utils.py
  legacy_loader.py
  pipeline.py
  super_resolution.py
  segmentation.py
  restoration.py
configs/
  pipeline.example.yaml
scripts/
  run_pipeline.py
```

Legacy code that is still used internally:

- `main_test_swinir.py`
- `khitan_auto_component_segmenter_paper_experiment/segment_core.py`
- `test/component_stylegan.py`
- `test/test_stage1_gan_best_retrieval_modified.py`

The new final stage is inspired by the last part of MARCONet in
`Learning Generative Structure Prior for Blind Text Image Super-resolution`:

- align the GAN-generated character prior to each detected character box
- normalize the prior to the local SR patch in an AdaIN-like way
- fuse the prior and the SR result at multiple scales
- write out a refined final text image instead of stopping at the raw GAN reconstruction

## Output Layout

Running the full pipeline creates a stable output tree under `output.root_dir`:

```text
runs/example/
  resolved_config.json
  pipeline_manifest.json
  01_super_resolution/
    images/
    manifest.json
  02_segmentation/
    patches/
    visualizations/
    segment_results.json
    manifest.json
  03_restoration/
    prototype_bank_preview.png
    <image_name>/
      components/
      reconstructed_rank1.png
      retrieval_results.json
    retrieval_results_all.json
    manifest.json
  04_refinement/
    <image_name>/
      sr_input.png
      gan_prior.png
      final_refined.png
      alpha_map.png
      refinement_results.json
    manifest.json
```

## Remote Server Usage

If your local workspace only has partial data, keep the config file local and point `search_roots` to the mounted path or absolute server-side workspace path that contains the real data and checkpoints.

The example config assumes the Stage1 checkpoint is available at:

`checkpoints/stage1_best.pth`

If that file is not local, put the server workspace root in `search_roots` so the resolver can find the remote checkpoint.

The path resolver used by the new package checks:

1. the config file directory
2. the project root
3. the current working directory
4. every path listed under `search_roots`

That makes it easier to keep one project layout while switching between local debugging and remote full-data execution.

For path semantics:

- plain relative paths like `runs/example` are treated as project-root relative
- explicit `./foo` or `../foo` paths are treated as config-file relative

## Suggested Next Cleanup Step

This pass organizes the project without rewriting your research code. A good second pass would be:

1. move the legacy logic into `khitan_restore/backends/`
2. replace dynamic legacy loading with direct package imports
3. standardize training code paths and dataset metadata
4. split the retrieval scoring logic into testable units
