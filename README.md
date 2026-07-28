# dinov3-ggml

DINOv3 Vision Transformer inference in Rust, powered by [GGML](https://github.com/ggml-org/ggml).
CPU by default, CUDA with the `cuda` feature.

The forward pass numerically matches the official Hugging Face `DINOv3ViTModel`
(verified against a PyTorch F32 reference: cosine ≈ 0.999999, max element diff < 5e-5):

- Patch embedding (conv 16×16) + `[CLS, registers, patches]` token order
- DINOv3-style RoPE: patch-center coords normalized to [-1, 1], `rope_theta = 100`,
  rotate-half application, prefix tokens not rotated
- LayerScale, exact-erf GELU, LayerNorm eps 1e-5

## Requirements

- Rust toolchain (MSVC on Windows)
- CPU build: a C/C++ compiler (no CMake needed)
- CUDA build (`--features cuda`): CUDA Toolkit (nvcc) + CMake
- Model conversion: Python 3 + PyTorch + NumPy

Clone with the GGML submodule:

```bash
git clone --recursive <this-repo>
# or, if already cloned:
git submodule update --init --recursive
```

## Getting the model

### Option A: download the pre-converted weights (ModelScope)

The converted ViT-S/16 binary (`dinov3_vits16.bin`, ~86MB) is available on ModelScope:

- Model page: https://www.modelscope.cn/models/cjc1887415157/dinov3-ggml

```bash
modelscope download cjc1887415157/dinov3-ggml dinov3_vits16.bin --local_dir ./models
```

Place the file at `models/dinov3_vits16.bin` (or point `GGML_MODEL_PATH` at it).

### Option B: convert from the official checkpoint

The DINOv3 weights are released by Meta under the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
and are **not** included in this repository. Download the ViT-S/16 checkpoint
(`dinov3_vits16_pretrain.pth`, e.g. from the official DINOv3 release or a mirror),
then convert it to the raw binary format:

```bash
# fix key name (storage_tokens -> register_tokens), then convert
python -c "import torch; sd = torch.load('dinov3_vits16_pretrain.pth', map_location='cpu', weights_only=True); sd['register_tokens'] = sd.pop('storage_tokens', sd.get('register_tokens')); torch.save(sd, 'dinov3_vits16_fixed.pth')"
python scripts/convert_weights.py -i dinov3_vits16_fixed.pth -o models/dinov3_vits16.bin -m vit_small_16
```

## Usage

```rust
use dinov3_ggml::{FeatureExtractor, VitConfig, l2_normalize, cosine_similarity};

let extractor = FeatureExtractor::load("models/dinov3_vits16.bin", VitConfig::vit_small_16())?;

let mut a = extractor.extract(&std::fs::read("a.jpg")?)?;
let mut b = extractor.extract(&std::fs::read("b.jpg")?)?;
l2_normalize(&mut a);
l2_normalize(&mut b);
println!("similarity = {:.4}", cosine_similarity(&a, &b));

// Batch: parallel preprocessing (rayon) + serialized backend inference
let feats = extractor.extract_batch(&[&img1[..], &img2[..]])?;
```

Run the demo (put jpg/png files in `./test`, or pass paths as arguments):

```bash
# CPU
cargo run --release --example demo

# CUDA
cargo run --release --features cuda --example demo
```

## Performance

RTX 4060 Laptop, ViT-S/16 @ 518×518 (1029 tokens), single image including
JPEG decode + resize:

| Backend | per image | batch throughput |
|---|---|---|
| CPU (AVX2, all cores) | ~1.6 s | — |
| CUDA | ~50 ms | ~22 img/s |

## Environment variables

| Variable | Effect |
|---|---|
| `GGML_MODEL_PATH` | model path for the demo (default `models/dinov3_vits16.bin`) |
| `GGML_VIT_FORCE_CPU=1` | force CPU even when built with `cuda` |
| `GGML_CUDA_ARCHS` | override CUDA architectures for the CMake build (default: auto-detect via nvidia-smi, e.g. `89`) |

## License

- This crate: [MIT](LICENSE)
- GGML (submodule): MIT
- DINOv3 weights: [Meta DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) — do not redistribute converted weights
