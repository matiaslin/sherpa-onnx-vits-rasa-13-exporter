# sherpa-onnx-vits-rasa-13-exporter

Export [`ai4bharat/vits_rasa_13`](https://huggingface.co/ai4bharat/vits_rasa_13) to an ONNX file consumable by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

Pre-exported models are hosted at: <https://huggingface.co/matiaslin/sherpa-onnx-vits-rasa-13>.

## Motivation

`optimum-cli` does not recognize the model's custom `indic_vits_model` type (registered via `trust_remote_code`), and the upstream `forward()` has an extra `emotion_id` input that stock `sherpa-onnx` does not pass. This exporter:

1. Loads the HF model with `trust_remote_code=True`.
2. Patches `_rational_quadratic_spline` to remove two tensor to bool guards that break ONNX tracing.
3. Wraps the model to bypass `forward()`'s tensor to bool `sid`/`emotion_id` range checks, and re-implements the inference path calling `text_encoder`, `duration_predictor`, `flow`, and `decoder` directly.
4. Matches `sherpa-onnx`'s multi-speaker VITS signature: `(x, x_length, noise_scale, length_scale, noise_scale_w, sid[, emotion_id])`.
5. Writes `tokens.txt` from the tokenizer's single-codepoint vocab entries (multi-codepoint entries are dropped).
6. Sets the ONNX metadata `sherpa-onnx` expects.

## Requirements

- Python 3.9+
- An HF account with access to the (gated) `ai4bharat/vits_rasa_13` repo, and a token exported as `HF_TOKEN`.
- Depending on which `sherpa-onnx` you plan to run against:
  - **Default (`--expose-emotion`)**: requires `sherpa-onnx` with emotion-input support (PR TBD). The resulting model has a 7th `emotion_id` input.
  - **`--no-expose-emotion`**: bakes one emotion id as a constant. The resulting model has the standard 6-input VITS signature.

```bash
pip install torch transformers onnx
```

## Usage

```bash
export HF_TOKEN=<your_huggingface_token>

# Default: emotion_id exposed as a 7th ONNX input.
./export_onnx_vits_rasa_13.py --output-dir ./vits-rasa-13-onnx

# Bake a specific emotion (e.g. HAPPY=8) for stock sherpa-onnx.
./export_onnx_vits_rasa_13.py --no-expose-emotion --emotion-id 8 \
    --output-dir ./vits-rasa-13-onnx-happy
```

Produces:

```
<output-dir>/model.onnx
<output-dir>/tokens.txt
```

### CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--model-id` | `ai4bharat/vits_rasa_13` | HF model to load. |
| `--output-dir` | `vits_rasa_13_onnx` | Where to write `model.onnx` and `tokens.txt`. |
| `--expose-emotion`/`--no-expose-emotion` | `--expose-emotion` | Add `emotion_id` as a 7th ONNX input (requires patched `sherpa-onnx`) or bake a constant. |
| `--emotion-id` | `0` | Emotion index in `[0, num_emotions)`. Baked when `--no-expose-emotion`. |
| `--opset` | `13` | ONNX opset. |
| `--language` | `multilingual-indic` | Value for the `language` metadata key. |

### Emotion / style ids

Documented ids in [vits_rasa_13](https://huggingface.co/ai4bharat/vits_rasa_13#speaker-style-identifier-overview)

| ID | Label |
| --- | --- |
| 0 | ALEXA (default; sounds like plain narration) |
| 1 | ANGER |
| 2 | BB |
| 3 | BOOK |
| 4 | CONV |
| 5 | DIGI |
| 6 | DISGUST |
| 7 | FEAR |
| 8 | HAPPY |
| 10 | NEWS |
| 12 | SAD |
| 14 | SURPRISE |
| 15 | UMANG |
| 16 | WIKI |

Ids 9, 11, 13, 17–31 are undocumented. Sending them will not crash but output quality is not guaranteed.

## Running the exported model

```bash
sherpa-onnx-offline-tts \
  --vits-model=./vits-rasa-13-onnx/model.onnx \
  --vits-tokens=./vits-rasa-13-onnx/tokens.txt \
  --sid=0 \
  --emotion-id=0 \
  --output-filename=./out.wav \
  "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்"
```

Omit `--emotion-id` if the model was exported with `--no-expose-emotion`.

## License

Model weights and architecture: see [`ai4bharat/vits_rasa_13`](https://huggingface.co/ai4bharat/vits_rasa_13)'s license.
