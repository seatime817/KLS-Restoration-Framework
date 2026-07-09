# Khitan Character Image Restoration

Official PyTorch implementation of **Digital Restoration of Khitan Large Script Inscriptions: A Cultural Heritage Perspective**.

> Authors: TODO: add author names  
> Affiliation: TODO: add affiliation  
> Paper / Preprint: TODO: add paper link  
> Project page: TODO: add project page link

[[`Paper`](TODO)] [[`Dataset`](TODO)] [[`Weights`](TODO)] [[`Results`](docs/results/README.md)]

---

![Framework](docs/results/framework.png)

We propose a restoration pipeline for blurry Khitan character/text images. The method first enhances low-quality inputs using SwinIR, segments character components, retrieves component-level structural priors, reconstructs candidate glyphs with a Stage1 codebook + StyleGAN model, and finally refines the image with prior-guided fusion.

## News

- [ ] Public dataset link will be released.
- [ ] Trained model weights will be released.
- [ ] Qualitative restoration examples will be added under `docs/results/`.

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/<YOUR_NAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

If the default PyTorch wheel does not match your CUDA version, install PyTorch from the official PyTorch selector first, then run the remaining commands.

## Dataset

Please organize the dataset as follows:

```text
dataset/
  components/
    mask/
    soft/
    component_annotations.json
    component_vocab.json
  data/
    train/
      trainL/
      trainH/
    test/
      testL/
      testH/
```

|         Dataset          | Description                                                  | Link |
| :----------------------: | :----------------------------------------------------------- | :--: |
| Khitan paired LR/HR data | Paired low-resolution and high-resolution images for learned refinement. | TODO |
|     Khitan test set      | Low-quality test images and optional ground truth.           | TODO |

If annotation files are not included, build them with:

```bash
python scripts/build_component_annotations.py --components-root dataset/components
```

## Trained Models

Large checkpoint files are not stored in this repository. Download them from the links below and place them in the target paths.

|       Model        | Usage                                              | Target Path                                                  |    Weights    |
| :----------------: | :------------------------------------------------- | :----------------------------------------------------------- | :-----------: |
| SwinIR real-SR x4  | Super-resolution backbone                          | `model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | Auto-download |
|     Stage1 GAN     | Component retrieval and structure prior generation | `checkpoints/stage1_best.pth`                                |     TODO      |
| Instance restorer  | Conditional component restoration                  | `checkpoints/instance_restoration_best.pth`                  |     TODO      |
| Learned refinement | ROI-based SR/prior fusion                          | `checkpoints/learned_refinement_best.pth`                    |     TODO      |

## Training Tutorial

- [x] Generate SwinIR outputs for the training LR images
- [x] Segment the super-resolved training images
- [x] Generate component-level restoration priors
- [x] Train the learned ROI-based refinement network

The learned refinement model is trained after three preprocessing stages. These stages create the `retrieval_results_all.json` file used by `configs/refinement_learned.data1.yaml`.

### Step 1: Super-Resolve Training Images

```bash
python -m khitan_restore.cli super-resolve \
  --config configs/pipeline.server.sy.yaml \
  --input /home/sy/dataset/data1/data/train1/trainL \
  --output runs/data2_train/01_super_resolution
```

Input:

- `/home/sy/dataset/data1/data/train1/trainL`: low-resolution training images.

Output:

- `runs/data2_train/01_super_resolution/images`: SwinIR super-resolved images.
- `runs/data2_train/01_super_resolution/manifest.json`: super-resolution stage metadata.

### Step 2: Segment Super-Resolved Images

```bash
python -m khitan_restore.cli segment \
  --config configs/pipeline.server.sy.yaml \
  --input runs/data2_train/01_super_resolution/images \
  --output runs/data2_train/02_segmentation
```

Input:

- `runs/data2_train/01_super_resolution/images`: output images from Step 1.

Output:

- `runs/data2_train/02_segmentation/segment_results.json`: component boxes and segmentation results.
- `runs/data2_train/02_segmentation/patches`: segmented component patches.
- `runs/data2_train/02_segmentation/visualizations`: segmentation visualization files.

### Step 3: Generate Component Restoration Priors

```bash
nohup python -m khitan_restore.cli restore-components \
  --config configs/pipeline.server.sy.yaml \
  --segment-json runs/data2_train/02_segmentation/segment_results.json \
  --output runs/data2_train/03_restoration &
```

Input:

- `runs/data2_train/02_segmentation/segment_results.json`: output JSON from Step 2.
- The component restoration checkpoint configured in `configs/pipeline.server.sy.yaml`.

Output:

- `runs/data2_train/03_restoration/retrieval_results_all.json`: retrieval and generated-prior metadata.
- `runs/data2_train/03_restoration/<image_name>/reconstructed_rank1.png`: generated structure prior for each image.
- `runs/data2_train/03_restoration/prototype_bank_preview.png`: optional component bank preview.

This command is usually slower, so `nohup ... &` is used to keep it running after disconnecting from the server.

### Step 4: Train Learned Refinement

```bash
nohup python -m khitan_restore.cli train-learned-refinement \
  --config configs/refinement_learned.data1.yaml &
```

`configs/refinement_learned.data1.yaml` expects:

- `data.restoration_results_path`: `/home/sy/item/swin/runs/data2_train/03_restoration/retrieval_results_all.json`
- `data.lr_dir`: `/home/sy/dataset/data1/data/train1/trainL`
- `data.gt_dir`: `/home/sy/dataset/data1/data/train1/trainH`

The training output is written to:

- `runs/learned_refinement_data2/checkpoints/learned_refinement_best.pth`
- `runs/learned_refinement_data2/checkpoints/learned_refinement_last.pth`
- `runs/learned_refinement_data2/previews`
- `runs/learned_refinement_data2/metrics_history.json`

### Optional: Train Stage1 / Instance Restoration From Scratch

If the released checkpoints are not available, the Stage1 and instance restoration models can be trained with:

```bash
python scripts/train_stage1.py \
  --config configs/stage1_components.example.yaml \
  --output runs/stage1_training

python scripts/train_instance_restorer.py \
  --config configs/instance_restorer.server.sy.quality.yaml \
  --output runs/instance_restoration_training
```

All training commands support checkpoint resume with `--resume <checkpoint.pth>`.

## Testing

Run the full restoration pipeline:

```bash
python -m khitan_restore.cli pipeline \
  --config configs/pipeline.example.yaml \
  --input dataset/data/test/testL \
  --output runs/test_pipeline
```

Run each stage separately:

```bash
python -m khitan_restore.cli super-resolve \
  --config configs/pipeline.example.yaml \
  --input dataset/data/test/testL \
  --output runs/sr_only

python -m khitan_restore.cli segment \
  --config configs/pipeline.example.yaml \
  --input runs/sr_only/images \
  --output runs/seg_only

python -m khitan_restore.cli restore-components \
  --config configs/pipeline.example.yaml \
  --segment-json runs/seg_only/segment_results.json \
  --output runs/restore_only \
  --ckpt checkpoints/stage1_best.pth

python -m khitan_restore.cli refine \
  --config configs/pipeline.example.yaml \
  --restoration-input runs/restore_only \
  --output runs/final_only
```

If `segment_results.json` is already available, restoration can start from the retrieval stage:

```bash
python -m khitan_restore.cli restore-components \
  --config configs/pipeline.example.yaml \
  --segment-json path/to/segment_results.json \
  --output runs/restore_only \
  --ckpt checkpoints/stage1_best.pth
```

Note: the current code expects `khitan_auto_component_segmenter_paper_experiment/segment_core.py` as the segmentation backend. Add this folder before release if the full pipeline should run directly after cloning.

## Results

### Quantitative Results

|        Method         | PSNR | SSIM | LPIPS | STSC | IoU  |     Model     |
| :-------------------: | :--: | :--: | :---: | :--: | :--: | :-----------: |
|        SwinIR         | TODO | TODO | TODO  | TODO | TODO | Auto-download |
| SwinIR + Stage1 Prior | TODO | TODO | TODO  | TODO | TODO |     TODO      |
|      Full Model       | TODO | TODO | TODO  | TODO | TODO |     TODO      |

### Qualitative Results

![Qualitative comparison](docs/results/comparison_01.png)

Add result images to `docs/results/` before publishing. Recommended files:

- `framework.png`
- `comparison_01.png`
- `comparison_02.png`
- `failure_cases.png`

## Evaluation

Evaluation can be performed as follows:

```bash
python -m khitan_restore.evaluate_refinement \
  --input runs/test_pipeline/04_refinement \
  --gt-dir dataset/data/test/testH \
  --output runs/test_pipeline/evaluation.json \
  --device auto
```

The evaluator reports PSNR, SSIM, LPIPS, STSC, IoU, precision, recall, F1, and Dice when ground-truth images are available.

## Repository Structure

| File / Folder                                      | Description                                                  |
| :------------------------------------------------- | :----------------------------------------------------------- |
| `configs/pipeline.example.yaml`                    | Example configuration for the full restoration pipeline.     |
| `configs/pipeline.server.sy.yaml`                  | Server-side full-pipeline configuration used in the original experiments. |
| `configs/pipeline.server.sy.learned.yaml`          | Server-side full-pipeline configuration with learned refinement enabled. |
| `configs/stage1_components.example.yaml`           | Example configuration for Stage1 component codebook + StyleGAN training. |
| `configs/stage1_components.server.sy.yaml`         | Server-side Stage1 training configuration.                   |
| `configs/stage1_components.server.sy.quality.yaml` | Quality-focused server-side Stage1 training configuration.   |
| `configs/instance_restorer.server.sy.quality.yaml` | Configuration for instance retrieval and conditional component restoration training. |
| `configs/refinement_learned.data1.yaml`            | Configuration for learned ROI-based refinement training.     |
| `configs/refinement_learned.server.sy.yaml`        | Server-side learned refinement configuration.                |
| `khitan_restore/cli.py`                            | Main command-line entry point for training, testing, and pipeline inference. |
| `khitan_restore/config.py`                         | Dataclass-based configuration definitions and config loading utilities. |
| `khitan_restore/pipeline.py`                       | Orchestrates super-resolution, segmentation, restoration, and refinement stages. |
| `khitan_restore/super_resolution.py`               | SwinIR wrapper for low-quality image super-resolution.       |
| `khitan_restore/segmentation.py`                   | Wrapper for component segmentation backend.                  |
| `khitan_restore/restoration.py`                    | Component restoration stage wrapper.                         |
| `khitan_restore/instance_restoration.py`           | Instance retrieval and conditional component restoration implementation. |
| `khitan_restore/refinement.py`                     | Rule-based prior-guided refinement stage.                    |
| `khitan_restore/learned_refinement.py`             | Learned refinement network and inference utilities.          |
| `khitan_restore/stage1_training.py`                | Stage1 component codebook + StyleGAN training code.          |
| `khitan_restore/instance_restoration_training.py`  | Instance restoration training code.                          |
| `khitan_restore/learned_refinement_training.py`    | Learned refinement training code.                            |
| `khitan_restore/evaluate_refinement.py`            | Evaluation script for final restoration outputs.             |
| `khitan_restore/evaluate_refinement1.py`           | Alternative evaluation script retained from experiments.     |
| `khitan_restore/component_annotations.py`          | Builds component annotation and vocabulary files from component data. |
| `khitan_restore/io_utils.py`                       | Shared filesystem, JSON/YAML, and image-listing utilities.   |
| `khitan_restore/legacy_loader.py`                  | Loads legacy experimental modules behind the unified interface. |
| `models/network_swinir.py`                         | SwinIR network architecture definition.                      |
| `scripts/run_pipeline.py`                          | Thin wrapper for the unified CLI.                            |
| `scripts/train_stage1.py`                          | Thin wrapper for Stage1 training.                            |
| `scripts/train_instance_restorer.py`               | Thin wrapper for instance restoration training.              |
| `scripts/train_learned_refinement.py`              | Thin wrapper for learned refinement training.                |
| `scripts/build_component_annotations.py`           | Thin wrapper for building component annotation files.        |
| `test/component_stylegan.py`                       | Legacy component StyleGAN implementation used by Stage1 and retrieval. |
| `test/test_stage1_gan_best_retrieval_modified.py`  | Legacy Stage1 retrieval and generation test script.          |
| `test/cospnet.yaml`                                | Legacy model/config file used by component retrieval.        |
| `utils/util_calculate_psnr_ssim.py`                | PSNR, SSIM, and PSNR-B metric utilities.                     |
| `main_test_swinir.py`                              | Original SwinIR test script retained for compatibility.      |
| `requirements.txt`                                 | Python dependency list.                                      |
| `pyproject.toml`                                   | Python package metadata and install configuration.           |
| `README_PROJECT.md`                                | Earlier project notes kept for reference.                    |
| `docs/results/README.md`                           | Placeholder instructions for result figures.                 |

Generated files such as `runs/`, `dataset/`, `checkpoints/`, `model_zoo/`, `*.pth`, and `__pycache__/` are ignored by Git.

## Acknowledgement

This repository uses and adapts code or ideas from the following projects:

- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [timm](https://github.com/huggingface/pytorch-image-models)
- PyTorch, TorchVision, TorchMetrics, OpenCV, and scikit-image

## License

This project is released under the Apache-2.0 License. Please see [LICENSE](LICENSE) for more information.

## Citation

If you find this repository helpful, please consider citing:

```bibtex
@misc{khitan_restoration_2026,
  title={Khitan Character Image Restoration with SwinIR, Component Retrieval, and Prior-Guided Refinement},
  author={TODO},
  year={2026},
  howpublished={\url{https://github.com/<YOUR_NAME>/<YOUR_REPO>}}
}
```
