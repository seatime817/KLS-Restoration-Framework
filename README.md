# Khitan Character Image Restoration

Code for "Digital Restoration of Khitan Large Script Inscriptions: A Cultural Heritage Perspective"

## Highlights

- SwinIR-based super-resolution for low-quality input images.
- Component-level segmentation and retrieval.
- Stage1 component codebook + StyleGAN restoration.
- Instance-level conditional component restoration.
- Learned ROI-based refinement that fuses SR output and GAN priors.
- Unified CLI for training, testing, evaluation, and full-pipeline inference.

## Repository Structure

```text
configs/                  # Example and server configs
khitan_restore/           # Unified Python package and training/evaluation code
models/                   # SwinIR network definition
scripts/                  # Thin command-line wrappers
test/                     # Legacy Stage1 / retrieval code used by the wrapper
utils/                    # PSNR / SSIM utilities
main_test_swinir.py       # Original SwinIR test script
README_PROJECT.md         # Earlier project notes
```

Generated files such as `__pycache__`, `runs/`, `checkpoints/`, `dataset/`, and `*.pth` are intentionally ignored by Git.

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

Install the PyTorch build that matches your CUDA version if the default wheel is not suitable for your machine.

## Data And Checkpoints

Large files are not committed to this repository. Upload datasets and `.pth` weights to GitHub Releases, Hugging Face, Google Drive, or Baidu Netdisk, then replace the placeholder links below.

### Dataset Links

| Name | Description | Link |
| --- | --- | --- |
| Khitan components | Component masks/soft images used by Stage1 and instance restoration. Expected root: `dataset/components/`. | TODO: add public URL |
| Khitan paired LR/HR data | Paired low-resolution and high-resolution images for learned refinement. Expected roots: `dataset/data/train/trainL` and `dataset/data/train/trainH`. | TODO: add public URL |
| Khitan test images | Low-quality test images for inference. Expected root: `dataset/data/test/testL`. | TODO: add public URL |

### Checkpoint Links

| File | Used By | Target Path | Link |
| --- | --- | --- | --- |
| `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | SwinIR real-world x4 SR | `model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | Auto-downloaded from the official SwinIR GitHub release |
| `stage1_best.pth` | Component retrieval / Stage1 GAN prior | `checkpoints/stage1_best.pth` | TODO: add public URL |
| `instance_restoration_best.pth` | Instance retrieval + conditional restoration | `checkpoints/instance_restoration_best.pth` | TODO: add public URL |
| `learned_refinement_best.pth` | Learned refinement | `checkpoints/learned_refinement_best.pth` | TODO: add public URL |

Recommended local layout after downloading:

```text
dataset/
  components/
    component_annotations.json
    component_vocab.json
    ...
  data/
    train/
      trainL/
      trainH/
    test/
      testL/
checkpoints/
  stage1_best.pth
  instance_restoration_best.pth
  learned_refinement_best.pth
model_zoo/
  swinir/
    003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
```

## Training

Build component annotations if they are not already included:

```bash
python scripts/build_component_annotations.py --components-root dataset/components
```

Train Stage1 component codebook + StyleGAN:

```bash
python scripts/train_stage1.py \
  --config configs/stage1_components.example.yaml \
  --output runs/stage1_training
```

Train the instance restoration model:

```bash
python scripts/train_instance_restorer.py \
  --config configs/instance_restorer.server.sy.quality.yaml \
  --output runs/instance_restoration_training
```

Train the learned refinement model after generating restoration results:

```bash
python scripts/train_learned_refinement.py \
  --config configs/refinement_learned.data1.yaml \
  --output runs/learned_refinement_training
```

All training scripts support `--resume <checkpoint.pth>`.

## Testing / Inference

Run the full pipeline:

```bash
python -m khitan_restore.cli pipeline \
  --config configs/pipeline.example.yaml \
  --input dataset/data/test/testL \
  --output runs/test_pipeline
```


If you already have a `segment_results.json`, you can skip segmentation and start from component restoration:

```bash
python -m khitan_restore.cli restore-components \
  --config configs/pipeline.example.yaml \
  --segment-json path/to/segment_results.json \
  --output runs/restore_only \
  --ckpt checkpoints/stage1_best.pth
```

Note: the current project references `khitan_auto_component_segmenter_paper_experiment/segment_core.py` as the segmentation backend. Add that folder before publishing if you want the `segment` and full `pipeline` commands to run out of the box.

## Evaluation

Evaluate final refinement outputs against ground truth:

```bash
python -m khitan_restore.evaluate_refinement \
  --input runs/test_pipeline/04_refinement \
  --gt-dir dataset/data/test/testH \
  --output runs/test_pipeline/evaluation.json \
  --device auto
```

The evaluator reports PSNR, SSIM, LPIPS, STSC, IoU, precision, recall, F1, and Dice where ground-truth images are available.

## Test Result Figures

Place qualitative results in `docs/results/` and update the image links below.

| Input | SwinIR | GAN Prior | Final | Ground Truth |
| --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO |

Example Markdown after adding figures:

```markdown
![Qualitative comparison](docs/results/comparison_01.png)
```

## Known Publishing Checklist

- Replace every `TODO: add public URL` with the final dataset/checkpoint link.
- Add the missing segmentation backend folder if it is part of the experiment.
- Add representative result images under `docs/results/`.
- Check that `configs/pipeline.example.yaml` uses relative public paths instead of private server paths.
- Create a GitHub release for large `.pth` files instead of committing them.

## License

This project is released under the Apache-2.0 License. See `LICENSE` for details.
