# Anime Video Super-Resolution via GAN Upscaling

Master's thesis project, HSE University, Faculty of Computer Science (2025). Defended with the grade "excellent". Supervisor: E. O. Kantonistova.

The project upscales low-resolution anime video (×2) while keeping line art and the drawing style intact. It covers the whole pipeline: building a frame dataset from anime episodes, training and comparing super-resolution models under different synthetic degradations, and a video inference module accelerated with TensorRT.

![example](https://github.com/user-attachments/assets/bea82d91-abe5-40a7-b1fa-2d23b9cdda93)

## Pipeline

1. **Dataset preparation** ([`dataset_preparation/`](dataset_preparation/README.md)): frames are extracted from episodes with FFmpeg scene-change detection (threshold 0.1), filtered by a ResNet18 quality score and deduplicated with a perceptual hash (`imagehash`, threshold 10).
2. **Degradation**: high-resolution frames are turned into LR/HR training pairs on the fly by one of three degradation pipelines.
3. **Training** (`train.py`): a GAN or a transformer model is trained on the pairs; configuration is managed by Hydra.
4. **Evaluation** (`inference.py`): the models are compared on the test split with PSNR, SSIM and LPIPS.
5. **Video inference** ([`inference_video/`](inference_video/README.md)): a video is split into frames, each frame is upscaled in overlapping tiles by the RRDBNet generator, and the video is reassembled.

## What was compared

**Degradation pipelines** (`src/degradation/`, configs in `src/configs/degradation/`):

| Pipeline | Description |
|---|---|
| Real-ESRGAN | Second-order degradation from Real-ESRGAN: blur kernels, resize, Gaussian/Poisson noise, JPEG, sinc filter |
| APISR | Degradation from APISR, with JPEG/WebP compression |
| Custom (author's) | Randomised operation order plus three anime-specific effects: DCT-based compression artifacts with adaptive quantisation matrices (separate Y/CbCr), intermittent edges (Canny), rainbow edges that simulate chromatic aberration (Sobel, per-channel shifts) |

**Models** (`src/model/`):

| Model | Role | Parameters | Loss |
|---|---|---|---|
| RRDBNet generator + U-Net discriminator with spectral normalisation (GAN) | Generator fine-tuned from Real-ESRGAN x2 weights | 21,080,068 (gen 16,703,171 + disc 4,376,897) | BCE adversarial + L1 pixel |
| Swin2SR (lightweight, x2) | Transformer SR | 1,005,348 | L1 |
| ESRT | Transformer SR | 677,783 | L1 |

3 architectures × 3 training degradations = 9 trained models. The code also contains SwinIR, but it is not part of the thesis comparison.

**Training setup** (one RTX 3070): gradient accumulation to an effective batch of 50 (batch 4 for the GAN, 1 for ESRT/Swin2SR); Adam with lr 3e-7 for the GAN generator and discriminator, `disc_steps=10`. Training time per degradation variant: GAN 25 h, ESRT 27 h, Swin2SR 18 h.

**Metrics** (`src/metrics/`): PSNR, SSIM, LPIPS.

**Tooling**: PyTorch, Hydra configs (`src/configs/`), experiment tracking with Weights & Biases or Comet ML (`src/configs/writer/`), TensorRT FP16 via torch2trt for video inference.

## Dataset

Frames were taken from anime series (full list below): about 148k frames, 425.46 GB in total.

| Split | Frames | Size | Share |
|---|---|---|---|
| Train | 127.9k (146 videos) | 351.79 GB | 86.5% |
| Val | 9.9k | 37.9 GB | 6.7% |
| Test | 10.1k | 35.77 GB | 6.8% |

To build your own dataset, follow [dataset_preparation/README.md](dataset_preparation/README.md).

<details>
<summary>Anime episodes used for the dataset</summary>

#### Train
- **Jujutsu Kaisen S2** - Episodes: 1, 5, 8, 12, 16, 21
- **Oshi No Ko S2** - Episodes: 1, 4, 8, 10
- **Dededede** - Episodes: 4, 8, 11, 15
- **Dungeon Meshi** - Episodes: 1, 3, 7, 11, 15, 19, 22, 24
- **Fruits Basket the Final** - Episodes: 3, 8, 11, 13
- **Giji Harem** - Episodes: 2, 5, 9, 11
- **Kaguya-sama wa Kokurasetai: Ultra Romantic** - Episodes: 4, 6, 10, 13
- **Karasu wa Aruji o Erabanai** - Episodes: 5, 9, 12, 16
- **Kusuriya no Hitorigoto** - Episodes: 4, 7, 10, 14, 18, 21
- **Make Heroine ga Oosugiru!** - Episodes: 2, 5, 7, 10
- **Nige Jouzu no Wakagimi** - Episodes: 1, 4, 6, 9
- **Re:Zero kara Hajimeru Isekai Seikatsu S3** - Episodes: 1, 2, 3
- **Sousou no Frieren** - Episodes: 4, 7, 10, 14, 18, 22, 25, 27
- **Tsue to Tsurugi no Wistoria** - Episodes: 1, 4, 6, 9
- **Vinland Saga S2** - Episodes: 3, 7, 12, 16, 19, 21
- **Kimi ni Todoke S3** - Episodes: 1, 2, 3, 4, 5
- **Pluto** - Episodes: 1, 3, 6, 8
- **Karakai Jouzu no Takagi-san S3** - Episodes: 1, 3, 4, 7, 10, 12
- **Kage no Jitsuryokusha ni Naritakute! S2** - Episodes: 1, 3, 5, 8, 10, 12
- **Chi. Chikyuu no Undou ni Tsuite** - Episodes: 1, 4, 6, 9, 12, 15
- **Kaijuu 8-gou** - Episodes: 2, 6, 9, 12
- **Ore dake Level Up na Ken** - Episodes: 1, 3, 5, 7, 9, 11
- **Kimetsu no Yaiba: Katanakaji no Sato-hen** - Episodes: 1, 3, 5, 7, 9, 11
- **Ao Ashi** - Episodes: 1, 5, 9, 12, 15, 18, 20, 22, 24
- **Dr. Stone: New World S3** - Episodes: 1, 3, 5, 7, 9, 11
- **Horimiya: Piece** - Episodes: 1, 3, 5, 7, 9, 11, 12, 13
- **Lycoris Recoil** - Episodes: 1, 3, 5, 7, 9, 11, 12, 13

#### Val
- **Dandadan** - Episodes: 3, 10
- **Spy x Family** - Episodes: 2, 8
- **Bocchi the Rock** - Episodes: 3, 8
- **Tengoku Daimakyou** - Episodes: 1, 9
- **Lycoris Recoil** - Episodes: 3, 12

#### Test
- **Spy x Family** - Episodes: 5, 11
- **Bocchi the Rock** - Episodes: 5, 11
- **Mob Psycho 100 S3** - Episodes: 5, 11
- **Kimi wa Houkago Insomnia** - Episodes: 5, 11
- **Atarashii Joushi wa Do Tennen** - Episodes: 5, 11

</details>

## Results

PSNR (dB) on the test split. Rows: degradation used for training. Columns: degradation applied to the test images. The best mean per model is in bold.

| Model | Trained on | jpeg | APISR | RealESRGAN | custom | mean |
|---|---|---|---|---|---|---|
| ESRT | APISR | 31.786 | 29.847 | 29.146 | 31.453 | 30.558 |
| ESRT | RealESRGAN | 31.760 | 29.570 | 30.062 | 31.276 | 30.667 |
| ESRT | custom | 32.999 | 29.796 | 29.440 | 33.082 | **31.329** |
| RRDBNet + U-Net (GAN) | APISR | 32.705 | 31.139 | 30.489 | 32.681 | 31.754 |
| RRDBNet + U-Net (GAN) | RealESRGAN | 33.124 | 30.844 | 31.434 | 32.364 | 31.942 |
| RRDBNet + U-Net (GAN) | custom | 33.926 | 30.771 | 30.620 | 32.465 | **31.946** |
| Swin2SR | APISR | 32.284 | 30.428 | 29.238 | 32.099 | 31.012 |
| Swin2SR | RealESRGAN | 32.639 | 29.915 | 30.487 | 32.043 | 31.271 |
| Swin2SR | custom | 33.583 | 30.063 | 29.571 | 33.915 | **31.783** |

Training with the custom degradation gives the best mean PSNR for every architecture. The gain over the Real-ESRGAN degradation is large for the light models (ESRT +0.66 dB, Swin2SR +0.51 dB) and negligible for the heavy GAN (31.946 vs 31.942 dB). Source: the defended thesis, section 3.2.

## Video inference performance

Video is processed in 128×128 overlapping tiles (8 px overlap), stitched back with a weight mask, and the RRDBNet generator runs in TensorRT FP16 via torch2trt. Measured on a GeForce RTX 3070:

| Input → Output | Average FPS |
|---|---|
| 480×360 → 960×720 | 12 |
| 640×480 → 1280×960 | 5 |

TensorRT / torch2trt installation notes are in [inference_video/README.md](inference_video/README.md).

## Installation

```bash
git clone https://github.com/alekseevpavel04/Anime-Video-Super-Resolution-via-GAN-Upscaling.git
cd Anime-Video-Super-Resolution-via-GAN-Upscaling
pip install -r requirements.txt
./download_assets.sh   # Real-ESRGAN x2 generator and discriminator weights -> assets/weights/
```

The degradation pipelines import `basicsr`, which is not listed in `requirements.txt`; install it separately (`pip install basicsr`).

## Usage

Training (Hydra config `src/configs/baseline.yaml`; select the model type, degradation and pretrained weights there or override from the command line):

```bash
python train.py
python train.py degradation=train_APISR   # example override
```

Image inference and metric calculation (config `src/configs/inference.yaml`):

```bash
python inference.py
```

Video upscaling:

```bash
cd inference_video
./download_assets.sh   # final RRDBNet weights and two sample videos
python inference.py    # reads input/, writes output/
```

## Repository structure

```
├── dataset_preparation/   # frame extraction, filtering, deduplication
├── inference_video/       # tiled RRDBNet video upscaling with TensorRT
├── src/
│   ├── configs/           # Hydra configs (models, degradations, metrics, writers)
│   ├── datasets/
│   ├── degradation/       # Real-ESRGAN, APISR, custom, JPEG pipelines
│   ├── logger/            # W&B and Comet ML writers
│   ├── loss/              # adversarial, pixel (L1), generator/discriminator criteria
│   ├── metrics/           # PSNR, SSIM, LPIPS
│   ├── model/             # RRDBNet, U-Net discriminator, ESRT, SwinIR, Swin2SR
│   └── trainer/
├── train.py
├── inference.py
└── download_assets.sh
```

## License

MIT, see [LICENSE](LICENSE).
