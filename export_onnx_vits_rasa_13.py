#!/usr/bin/env python3

"""
This script exports ai4bharat/vits_rasa_13 to a sherpa-onnx-compatible ONNX
model. The upstream HF model uses a custom `indic_vits_model` type via
`trust_remote_code` (which optimum-cli does not recognize) and has an extra
`emotion_id` input that stock sherpa-onnx (pre-emotion-support) does not pass.

Usage:

(1) Install deps

    pip install torch transformers onnx

(2) Run

    ./export-onnx-vits-rasa-13.py --output-dir ./vits-rasa-13-onnx

    # or bake a specific emotion for use with pre-emotion-support sherpa-onnx:
    ./export-onnx-vits-rasa-13.py --no-expose-emotion --emotion-id 3

It will produce:

    ./vits-rasa-13-onnx/model.onnx
    ./vits-rasa-13-onnx/tokens.txt
"""

import argparse
import inspect
import onnx
import sys
import torch

from pathlib import Path
from torch import nn
from transformers import AutoModel, AutoTokenizer

def neutralize_spline_domain_checks(model: nn.Module) -> None:
    """Remove tensor to bool sanity asserts in `_rational_quadratic_spline` that break tracing.

    The stochastic duration predictor's spline helper guards against out-of-domain inputs
    and negative discriminants with Python-bool checks over tensors. Both trip torch.onnx
    tracing (bool-of-tensor and/or the raise firing on random dummy inputs). They are
    asserts, not part of the numerical output, so no-op them for export.
    """
    mod = sys.modules[type(model).__module__]
    src = inspect.getsource(mod._rational_quadratic_spline)

    replacements = [
        (
            '    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:\n'
            '        raise ValueError("Input to a transform is not within its domain")\n',
            "    # domain check disabled for ONNX tracing\n",
        ),
        (
            "        if not (discriminant >= 0).all():\n"
            '            raise RuntimeError(f"invalid discriminant {discriminant}")\n',
            "        # discriminant check disabled for ONNX tracing\n",
        ),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(
                "Could not locate expected guard block in _rational_quadratic_spline; "
                "the upstream model file may have changed."
            )
        src = src.replace(old, new)

    ns = dict(mod.__dict__)
    exec(compile(src, mod.__file__ + " [patched]", "exec"), ns)
    mod._rational_quadratic_spline = ns["_rational_quadratic_spline"]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="ai4bharat/vits_rasa_13")
    p.add_argument("--output-dir", default="vits_rasa_13_onnx")
    p.add_argument(
        "--emotion-id",
        type=int,
        default=0,
        help="Emotion index in [0, num_emotions). Used as the dummy trace value when "
        "--expose-emotion is set (the default), or baked as a constant with --no-expose-emotion.",
    )
    p.add_argument(
        "--expose-emotion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add emotion_id as a 7th ONNX input (default). Requires sherpa-onnx with "
        "emotion support. Use --no-expose-emotion to bake --emotion-id as a constant.",
    )
    p.add_argument("--opset", type=int, default=13)
    p.add_argument(
        "--language",
        default="multilingual-indic",
        help="Value for the ONNX 'language' metadata key.",
    )
    return p.parse_args()


class OnnxVitsWrapper(nn.Module):
    """Adapts IndicVitsModel to sherpa-onnx's multi-speaker VITS forward signature.

    Rather than calling the HF model's forward (which has `not 0 <= sid < num_speakers`
    checks that break under tracing when sid is a tensor), this reimplements the
    inference path using the model's submodules directly, and plumbs sherpa-onnx's
    noise_scale, length_scale and noise_scale_w as proper tensor inputs.

    References:
    - https://huggingface.co/ai4bharat/vits_rasa_13/blob/refs%2Fpr%2F1/modeling_vits.py
    - https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/vits/export-onnx-vctk.py
    """

    def __init__(self, model: nn.Module, emotion_id: int, expose_emotion: bool):
        super().__init__()
        self.model = model
        self.expose_emotion = expose_emotion
        if not expose_emotion:
            self.register_buffer(
                "_baked_emotion_id",
                torch.tensor([emotion_id], dtype=torch.long),
                persistent=False,
            )

    def _synthesize(self, x, x_length, noise_scale, length_scale, noise_scale_w, sid, emotion_id):
        model = self.model
        config = model.config

        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        attention_mask = (positions < x_length.unsqueeze(1)).long()
        input_padding_mask = attention_mask.unsqueeze(-1).float()

        speaker_and_style_embeddings = model.embed_speaker(sid).unsqueeze(-1)
        emotion_embeddings = model.embed_emotion(emotion_id).unsqueeze(-1)
        speaker_and_style_embeddings = speaker_and_style_embeddings + emotion_embeddings

        text_encoder_output = model.text_encoder(
            input_ids=x,
            padding_mask=input_padding_mask,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = text_encoder_output.last_hidden_state.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output.prior_means
        prior_log_variances = text_encoder_output.prior_log_variances

        if config.use_stochastic_duration_prediction:
            log_duration = model.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_and_style_embeddings,
                reverse=True,
                noise_scale=noise_scale_w,
            )
        else:
            log_duration = model.duration_predictor(
                hidden_states, input_padding_mask, speaker_and_style_embeddings
            )

        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        max_len = predicted_lengths.max()
        indices = torch.arange(max_len, dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = (
            (indices.unsqueeze(0) < predicted_lengths.unsqueeze(1))
            .unsqueeze(1)
            .to(input_padding_mask.dtype)
        )

        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = (indices.unsqueeze(0) < cum_duration).to(attn_mask.dtype).view(
            batch_size, input_length, output_length
        )
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = (
            prior_means
            + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * noise_scale
        )
        latents = model.flow(prior_latents, output_padding_mask, speaker_and_style_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask
        return model.decoder(spectrogram, speaker_and_style_embeddings)  # (N, 1, T)

    def forward(self, x, x_length, noise_scale, length_scale, noise_scale_w, sid, emotion_id=None):
        if self.expose_emotion:
            return self._synthesize(x, x_length, noise_scale, length_scale, noise_scale_w, sid, emotion_id)
        return self._synthesize(x, x_length, noise_scale, length_scale, noise_scale_w, sid, self._baked_emotion_id)


def write_tokens(tokenizer, out_path: Path) -> int:
    """Write tokens.txt in sherpa-onnx format, skipping multi-codepoint vocab entries.

    sherpa-onnx's character frontend rejects any token whose UTF-32 length != 1. The
    upstream vocab includes entries like `t̺` (t + combining mark) that are unreachable
    via the Python tokenizer's normal path anyway. Returns the count of skipped tokens.
    """
    vocab = tokenizer.get_vocab()  # token -> id
    id_to_tok = sorted(vocab.items(), key=lambda kv: kv[1])
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for tok, idx in id_to_tok:
            if tok is None or len(tok) != 1:
                skipped += 1
                continue
            f.write(f"{tok} {idx}\n")
    return skipped


def set_meta(onnx_path: Path, meta: dict) -> None:
    model = onnx.load(str(onnx_path))
    while model.metadata_props:
        model.metadata_props.pop()
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)
    onnx.save(model, str(onnx_path))


def main() -> None:
    args = get_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()
    neutralize_spline_domain_checks(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = model.config

    if not 0 <= args.emotion_id < config.num_emotions:
        raise ValueError(
            f"--emotion-id must be in [0, {config.num_emotions}); got {args.emotion_id}"
        )

    wrapper = OnnxVitsWrapper(model, args.emotion_id, args.expose_emotion)
    wrapper.eval()

    x = torch.randint(low=1, high=config.vocab_size, size=(1, 32), dtype=torch.long)
    x_length = torch.tensor([32], dtype=torch.long)
    noise_scale = torch.tensor(config.noise_scale, dtype=torch.float32)
    length_scale = torch.tensor(1.0 / config.speaking_rate, dtype=torch.float32)
    noise_scale_w = torch.tensor(config.noise_scale_duration, dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.long)

    input_names = ["x", "x_length", "noise_scale", "length_scale", "noise_scale_w", "sid"]
    dynamic_axes = {
        "x": {0: "N", 1: "L"},
        "x_length": {0: "N"},
        "sid": {0: "N"},
        "y": {0: "N", 2: "T"},
    }
    inputs = (x, x_length, noise_scale, length_scale, noise_scale_w, sid)
    if args.expose_emotion:
        eid = torch.tensor([args.emotion_id], dtype=torch.long)
        inputs = inputs + (eid,)
        input_names.append("emotion_id")
        dynamic_axes["emotion_id"] = {0: "N"}

    onnx_path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["y"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
        )

    tokens_path = out_dir / "tokens.txt"
    skipped = write_tokens(tokenizer, tokens_path)
    if skipped:
        print(f"Skipped {skipped} multi-codepoint vocab entries (unreachable via char frontend)")

    # Metadata keys extracted from:
    # - https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-vits-impl.h
    # - https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-character-frontend.cc
    meta = {
        "model_type": "vits",
        "comment": "ai4bharat/vits_rasa_13",
        "language": args.language,
        "frontend": "characters",
        "add_blank": int(getattr(tokenizer, "add_blank", True)),
        "n_speakers": config.num_speakers,
        "sample_rate": config.sampling_rate,
        "punctuation": "",
    }
    if args.expose_emotion:
        # Signal to sherpa-onnx that this model expects a 7th emotion_id input.
        meta["num_emotions"] = config.num_emotions
    else:
        # Record which emotion was baked so downstream users know what they got,
        # but do NOT set num_emotions — the model is single-emotion at runtime.
        meta["baked_emotion_id"] = args.emotion_id
    set_meta(onnx_path, meta)

    print(f"Wrote {onnx_path}")
    print(f"Wrote {tokens_path}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
