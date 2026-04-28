# PLaTune: Pretrained Latents Tuner - Adding temporal musical controls on top of pretrained generative models

This repository is linked to our paper submission to the [ISMIR 25](https://ismir2025.ismir.net/) Conference. 

## Abstract 
Recent advances in deep generative modeling have enabled      high-quality models for musical audio synthesis. However, these approaches remain difficult to control, confined to simple, static attributes and, most importantly, entail retraining a different computationally-heavy architecture for each new control. This is inefficient and impractical as it requires substantial computational resources.\
In this paper, we propose a novel approach allowing to add time-varying musical controls on top of any pretrained generative models with an exposed latent space (e.g. neural audio codecs), without retraining or finetuning. Our method supports both discrete and continuous attributes by adapting a rectified flow approach with a latent diffusion transformer. We learn an invertible mapping between pretrained latent variables and a new space disentangling explicit control attributes and *style* variables that capture the remaining factors of variation.\
This enables both feature extraction from an input, but also editing those features to generate transformed audio samples. Finally, this also introduces the ability to perform synthesis directly from the audio descriptors. We validate our method with 4 datasets going from different musical instruments up to full music recordings, on which we outperform state-of-the-art task-specific baselines in terms of both generation quality and accuracy of the control by inferring transferred attributes.


# Install

Create a virtual environment with Python>=3.12 and install dependencies:
```bash
$ pip install -r requirements.txt
```

For training on GPUs, install the Pytorch libraries compatible with your CUDA installation:

```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*Note: If you encounter some intall errors please check this [section](#fixing-some-install-errors).*


# Data preparation

Process your audio data into a LMDB database and precompute both the latent representations of the pretrained neural audio codec you want to control and the control attributes that we will define your control space.

To do so, use the `prepare_dataset.py` script. The specifications are:
```bash
Usage: prepare_dataset.py [OPTIONS]

Options:
  -i, --input_path TEXT        folder with the audio files
  -o, --output_path TEXT       lmdb save path
  -s, --db_size INTEGER        Max Size of lmdb database
  -p, --parser_name TEXT       parser function to obtain the list audio files
                               and metadatas
  -c, --config TEXT            Name of the gin configuration file to use
  -m, --emb_model_path TEXT    code to use. Either "music2latent" or a
                               torchscript path
  --gpu INTEGER                device for basic-pitch and codec (-1 for cpu)
  -n, --num_signal INTEGER     chunk sizes
  --sr INTEGER                 sample rate
  -b, --batch_size INTEGER     Batch size (for embedding model inference)
  --normalize                  Normalize audio waveform (done once per file
                               and not chunks ! )
  --cut_silences               Remove silence chunks
  --save_waveform              Wether to save the waveform in the lmdb
  -l, --descriptors_list TEXT  list of audio descriptors to compute on audio
                               chunks
  --use_basic_pitch            use basic pitch for midi extraction from audio
  --midi_attributes            Whether to compute and save midi attributes on
                               the midi chunks
  --help                       Show this message and exit.
```

For instance, to prepare MedleySolos data using Music2Latent official pretrained codec with controls on melody, instruments, and basic audio descriptors, you can do:
```bash
(myenv)$ python scripts/prepare_dataset.py -i /path/to/audios -o /path/to/processed_m2l -s 64 -p medley_solos_mono_parser -c medley_solos -m music2latent --gpu 1 -n 131072 --sr 44100 -b 32 --cut_silences --save_waveform -l rms -l centroid -l bandwidth -l booming -l sharpness -l integrated_loudness -l loudness1s --use_basic_pitch --midi_attributes
```

# Train

(Optional, recommended when using continuous attributes.) Compute per-attribute min/max, quantization bins, and a NaN-free key list over the LMDB built above:

```bash
(myenv)$ python scripts/compute_min_max_dataset.py --data_path /path/to/processed_m2l
```

This writes `metadata_attributes.json` (min/max per continuous attribute, value counts per discrete attribute), `bins_values.pkl` (quantization bins per continuous attribute), and `lmdb_keys.pkl` (LMDB keys with the NaN examples filtered out) inside the dataset folder.

Pick or write a gin config in `platune/configs/`. The shipped `v1.gin` defines the discrete control set `["pitch", "octave", "onsets", "dynamics", "instrument"]` (no continuous controls) for a 32-channel codec like Music2Latent, with a 6-layer transformer denoiser and rotary positional embeddings. To control different attributes or change model size, edit `DISCRETE_KEYS`, `CONTINUOUS_KEYS`, `CLASSES_ATTR_DISCRETE`, `LATENT_DIM`, `SEQ_LENGTH`, etc.

Launch training:

```bash
(myenv)$ python scripts/train.py \
    -d /path/to/processed_m2l \
    -n my_run \
    -c v1 \
    -s /path/to/runs \
    --gpu 0 \
    --max_steps 300000 \
    --val_every 10000
```

Specifications:
```bash
Usage: train.py [OPTIONS]

Options:
  -d, --db_path TEXT             dataset path
  -n, --name TEXT                Name of the run
  -c, --config TEXT              Name of the gin configuration file to use
  -s, --save_path TEXT           path to save models checkpoints
  --max_steps INTEGER            Maximum number of training steps
  --val_every INTEGER            Checkpoint model every n steps
  --gpu INTEGER                  GPU to use (-1 for cpu)
  --ckpt TEXT                    Path to previous checkpoint of the run
  --build_cache                  Load dataset in cache memory for training
  --lmdb_keys_filename TEXT      lmdb keys filename (e.g. lmdb_keys)
  --bins_values_file TEXT        bins_values pkl file to quantize continuous attributes
  --min_max_file TEXT            metadata json file (min/max for continuous attributes)
  --help                         Show this message and exit.
```

When using continuous attributes, pass **either** `--bins_values_file bins_values.pkl` to treat them as quantized classes, **or** `--min_max_file metadata_attributes.json` to keep them continuous and only normalize them to `[-1, 1]` — the two options are mutually exclusive. `--lmdb_keys_filename lmdb_keys` skips the NaN examples filtered by the previous step. To resume from a previous run, point `--ckpt` at the run directory or directly at a `.ckpt` file.

Checkpoints (`last.ckpt` and `best.ckpt`) and TensorBoard logs are written under `<save_path>/<name>/`. The exact gin configuration used is also saved next to them as `config.gin` for reproducibility.

```bash
(myenv)$ tensorboard --logdir /path/to/runs/my_run
```

# Inference

A trained PLaTune model is paired with the same pretrained codec used at data preparation time. Load both with the `load_model` helper, then use `z_to_cs` to extract the disentangled `(c, s)` representation from a codec latent, and `cs_to_z` to map an edited `(c, s)` back to a codec latent that you can decode to audio.

```python
import torch, librosa, soundfile as sf
from platune.helpers.model_loaders import load_model

device = "cuda:0"

# load PLaTune + the pretrained codec
model, codec = load_model(
    ckpt_path="/path/to/runs/my_run/best.ckpt",
    config_path="/path/to/runs/my_run/config.gin",
    emb_model_path="music2latent",         # or path to a torchscript codec
    device=device,
)

# encode an audio file to the codec's latent space
audio, _ = librosa.load("input.wav", sr=44100, mono=True)
x = torch.from_numpy(audio).to(device).reshape(-1, 131072)        # (B, num_signal)
z = codec.encode(x)                                                # (B, latent_dim, T)

# extract control + style: c is the first `control_dim` channels, s the rest
cs = model.z_to_cs(z, nb_steps=model.nb_steps)
c, s = cs[:, :model.control_dim], cs[:, model.control_dim:]

# --- example edit: transpose the melody up by one octave ---
# c is normalized to [-1, 1] per channel. To set an absolute attribute value,
# build it in raw class/value space and run model.normalize_attr(...).
# Here we just shift the "octave" channel (index 1 in v1.gin's DISCRETE_KEYS).
c_edit = c.clone()
c_edit[:, 1] = c[:, 1] + 0.2

# resynthesize the latent and decode to audio
cs_edit = torch.cat([c_edit, s], dim=1)
z_edit = model.cs_to_z(cs_edit, nb_steps=model.nb_steps)
audio_out = codec.decode(z_edit).cpu().numpy().reshape(-1)
sf.write("edited.wav", audio_out, 44100)
```

To synthesize directly from attributes (no input audio), build a tensor `a` of shape `(B, control_dim, T)` whose channels follow the order `discrete_keys + continuous_keys` (raw class indices for discrete keys, raw values for continuous keys), then sample `(c, s)` from the prior:

```python
c = model.normalize_attr(a.to(device))                             # → [-1, 1]
c_dist, s_dist = model.get_cs_distributions(c, warmup=False, zero_var=True)
cs = model.get_cs_samples(c_dist, s_dist)
z = model.cs_to_z(cs, nb_steps=model.nb_steps)
audio_out = codec.decode(z).cpu().numpy().reshape(-1)
```

Use `zero_var=True` for deterministic conditioning on the attribute values, or `zero_var=False` (default) to draw `c` from the per-attribute Gaussian learned during training.

---
---
## Fixing some install errors

- If you are using music2latent pretrained codec with Python>=3.12, you may encounter the following error:
```bash
 File "/path/to/env/lib/python3.12/site-packages/torch/serialization.py", line 1524, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 

        (1) In PyTorch 2.6, we changed the default value of the weights_only argument in torch.load from False to True. Re-running torch.load with weights_only set to False will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with weights_only=True please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default. Please use torch.serialization.add_safe_globals([numpy.core.multiarray.scalar]) or the torch.serialization.safe_globals([numpy.core.multiarray.scalar]) context manager to allowlist this global if you trust this class/function.
```

A fix quick is to specify `weights_only=False` in  `/path/to/env/lib/python3.12/site-packages/music2latent/inference.py`:
```python
class EncoderDecoder:
    ...
    def get_models(self):
        gen = UNet().to(self.device)
        checkpoint = torch.load(self.load_path_inference, map_location=self.device, weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        self.gen = gen
    ...
```

# Citation
```
@inproceedings{nabi2025platune,
  title={Adding temporal musical controls on top of pretrained generative models},
  author={Nabi, Sarah and Demerl{\'e}, Nils and Peeters, Geoffroy and Bevilacqua, Fr{\'e}d{\'e}ric and Esling, Philippe},
  booktitle={International Society for Music Information Retrieval, ISMIR 2025},
  year={2025}
}
```
