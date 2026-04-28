# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PLaTune (ISMIR 2025) — a research codebase that learns time-varying musical controls on top of an already-trained neural audio codec, *without* retraining the codec. A latent-diffusion transformer (rectified flow) maps between the codec's latent `z` and a disentangled `(c, s)` space where `c` is the explicit control variables (pitch, octave, onsets, dynamics, instrument, plus optional continuous descriptors like loudness/centroid) and `s` is a unit-Gaussian style residual.

The repository has no test suite, no linter config, and no CI.

## Common commands

Setup (Python ≥ 3.12):

```bash
pip install -r requirements.txt
# For CUDA training:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .   # exposes the `platune` package to the scripts
```

Three-stage pipeline (each script is a `click` CLI, run with `--help` for full options):

```bash
# 1. Build LMDB of (codec latents, attributes) from raw audio.
python scripts/prepare_dataset.py \
    -i /path/to/audios -o /path/to/processed_lmdb \
    -p medley_solos_mono_parser -c medley_solos -m music2latent \
    -n 131072 --sr 44100 -b 32 --gpu 0 \
    --cut_silences --save_waveform --use_basic_pitch --midi_attributes \
    -l rms -l centroid -l bandwidth -l booming -l sharpness \
    -l integrated_loudness -l loudness1s

# 2. (Only if using continuous attributes.) Compute per-attribute min/max,
#    quantization bins, and an `lmdb_keys.pkl` that drops NaN examples.
python scripts/compute_min_max_dataset.py --data_path /path/to/processed_lmdb

# 3. Train.
python scripts/train.py \
    -d /path/to/processed_lmdb -n run_name -c v1 \
    -s /path/to/checkpoints --gpu 0 --max_steps 300000 --val_every 10000
# Optional flags: --build_cache, --lmdb_keys_filename lmdb_keys,
#                 --bins_values_file bins_values.pkl OR --min_max_file metadata_attributes.json
```

TensorBoard logs land under `<save_path>/<run_name>/`; checkpoints `last.ckpt` and (if validation runs) `best.ckpt` are written there. Resume training with `--ckpt <path-or-dir>` (a directory is searched recursively for the most recent `*last*.ckpt`).

`README.md` documents a known music2latent + PyTorch ≥ 2.6 issue: the codec ships a checkpoint that needs `weights_only=False` in its `torch.load` call inside `music2latent/inference.py`.

## Architecture

### Configuration layer (gin)

Two separate gin namespaces:

- `platune/configs/*.gin` — model + training hyperparameters. Bound by `scripts/train.py` via `--config <name>` (looks up `platune/configs/<name>.gin`). `v1.gin` is the canonical example.
- `platune/datasets/configs/*.gin` — attribute-processing hyperparameters (note alphabet, octave boundaries, dynamics buckets, instrument vocabulary, sample rate). Bound by `scripts/prepare_dataset.py` via `--config <name>`.

Both scripts mutate gin under `gin.unlock_config()` to inject runtime values (data path, `z_length`, `num_signal`, descriptor list, min/max, bins). Anything `@gin.configurable` (e.g. `dataset.load_data`, `transformerv2.Denoiser`, `model.PLaTune`, `process_attributes.process_midi_attributes`, `audio_descriptors.compute_all`) gets its default args from gin.

### Data layer

`SimpleDataset` (`platune/datasets/base.py`) wraps an LMDB store of `AudioExample` records. Each record carries arbitrary keyed arrays: `z` (codec latents `(C, T)`), `waveform`, `midi`, `metadata`, plus per-attribute `(T,)` arrays for each attribute name.

`LatentsContinuousDiscreteAttritbutesDataset` (`platune/datasets/dataset.py`) is what training uses. It:

- Pads the time dimension to the next power of 2 (`replicate` on the last frame) so transformer sequence lengths line up.
- For each requested key, decides discrete vs. continuous from the module-level `DISCRETE_ATTRIBUTES` / `CONTINUOUS_ATTRIBUTES` lists. **If you add a new attribute, register it in one of those lists** or `__getitem__` raises.
- Resamples mismatched-length attributes to `z_length`: `nearest` for discrete, `linear` for continuous. The `jamendo` dataset has special-cased windowed re-interpolation.
- Returns `(z, attr_discrete, attr_continuous)`. Splits 95/5 train/val with a fixed seed.

`prepare_dataset.py` is the producer: it parses an audio collection (parser registry in `parsers.py`: `simple_parser`, `medley_solos_mono_parser`, `urmp_parser`, `slakh_parser`, `synthetic_parser`, `maestro_parser`), chunks each file to `num_signal` samples, runs the codec to get `z`, optionally runs `BasicPitchPytorch` for MIDI, computes audio descriptors via `audio_descriptors.compute_all`, and processes MIDI into discrete attributes via `process_attributes.process_midi_attributes` (v1) or `process_midi_attributesv2`. The codec is selected at runtime: `-m music2latent` uses the `music2latent` package; otherwise `-m <path.ts>` loads a TorchScript codec (the two have different input shapes — see the `if emb_model_path == "music2latent"` branches).

### Model

`model.py:PLaTune` is a `pl.LightningModule`. Key invariant: **`latent_dim = control_dim + style_dim`**, where `control_dim = len(discrete_keys) + len(continuous_keys)`. The control channels of `z` are aligned 1:1 with attributes; the rest are style.

Forward attribute pipeline:

1. `process_attributes` converts raw attribute tensors into ints. Discrete attributes are mapped through `classes_attr_discrete` (a list of allowed-value lists per discrete key) via `searchsorted`. Continuous attributes are either passed through, or bucketized using `bins_values` if provided (treats them as discrete classes).
2. `normalize_attr` rescales each channel to `[-1, 1]` using `min_max_attr` (derived from class counts for discrete / quantized continuous, or from `min_max_attr_continuous` otherwise).
3. `get_cs_distributions` builds a Normal centered at the (normalized) attribute with per-channel σ — for discrete: `σ = (2 / n_classes) · r`; for continuous (un-quantized): a flat `sigma_target_continuous`. During training, σ is **warmed up** from `sigma_init` decaying with `sigma_decay**(global_step/50)` toward the target (`get_sigma(..., warmup=True)`).
4. The style block is N(0, I) of width `style_dim`.

Training step (rectified flow):
```
target      = z - cs
interpolant = (1 - t) * cs + t * z      # t ~ U(0,1) per sample
loss        = MSE(flow(interpolant, t), target)
```
Manual optimization (`automatic_optimization = False`) with optional `clip_grad_norm_(_, 1.0)`.

Inference uses Euler integration of the learned flow:
- `cs_to_z(cs, nb_steps)` integrates `t: 0 → 1` (synthesis).
- `z_to_cs(z, nb_steps)` integrates `t: 1 → 0` (feature extraction).

Validation logs `validation` (NLL of the inverted `cs_rec` under the prior), `rec_loss` (round-trip MSE through `cs → z`), and `rec_loss_zerosigma` (same but with σ → 0, i.e. deterministic attribute conditioning). It also plots `c_gt` vs. `c_rec` per control channel via `helpers/data_visualization.plot_features_extraction`.

### Network

`networks/transformerv2.py:Denoiser` is the `flow`. It is an AdaLN-conditioned transformer:

- Input `x: (B, n_channels, T)` is projected via `b c t → b t c → linear → LayerNorm` to `embed_dim`.
- Time `t` is Fourier-embedded (`PositionalEmbedding`) and fed as the AdaLN conditioning `(α, β)` per block.
- Position encoding is selectable: `learnable`, `rotary` (uses `RotaryEmbedding` from `networks/rotary_embedding.py`), or `none`.
- A KV-cache path (`max_cache_size > 0`, `roll_cache`) exists for streaming inference; training uses `max_cache_size=0` with non-causal attention as in `v1.gin`.
- `transformer.py` is an older variant kept around but `v1.gin` wires `transformerv2.Denoiser`.

### Auxiliary modules

- `platune/datasets/audio_descriptors.py` — librosa + `pyloudnorm` + the bundled `timbral_models/` package (booming, brightness, depth, hardness, roughness, sharpness, warmth). `compute_all` resamples each descriptor to `z_length`.
- `platune/datasets/basic_pitch_torch/` — vendored PyTorch port of Spotify's basic-pitch for MIDI extraction; loaded via `transforms.BasicPitchPytorch`.
- `platune/datasets/audio_example/` — LMDB record format (key/value byte arrays + a metadata blob); both `prepare_dataset.py` (writer) and `SimpleDataset` (reader) go through this.
- `platune/helpers/model_loaders.py` — utilities for instantiating a trained model from a run directory (config + checkpoint).
