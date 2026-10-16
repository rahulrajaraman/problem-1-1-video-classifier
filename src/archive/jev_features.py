#!/usr/bin/env python3
"""Extract frozen jev-visual (Qwen3.5-0.8B, 4-bit, MLX) features for a linear probe.

Each clip grid image saved by jev_zero_shot.py is run through the model with
the same domain question. Every layer's hidden state is kept in two forms:
the mean over the image tokens, and the state at the final prompt position,
where the model is about to answer. Each video is saved as a float16 array

    [clips, 2, layers + 1, hidden]    pool 0 = image mean, pool 1 = answer position

where the last layer entry is the final normed output. Run it with the
jev-visual environment, which has MLX but no torch:

    ../jev-visual/.venv/bin/python src/jev_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jev_zero_shot import JEV_ROOT, PROMPTS, QUESTION, ROOT, read_jsonl, split_prompts

SPLITS = ("train", "val", "test")


def image_forward(adapter, prompt: str, image, letter_ids=None):
    """One frozen pass over an image and prompt.

    Returns float16 features [2, layers + 1, hidden] (image-token mean, answer
    position) and, if `letter_ids` is given, the raw logits of those tokens at
    the answer position.
    """
    mx, model = adapter.mx, adapter.model
    language = model.language_model
    adapter.reset()
    inputs = adapter.prepare(prompt, image)
    ids = inputs["input_ids"]
    embedded = model.get_input_embeddings(
        ids, pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"), mask=inputs.get("attention_mask"),
    )
    out = language(
        ids, inputs_embeds=embedded.inputs_embeds, cache=language.make_cache(),
        position_ids=embedded.position_ids, rope_deltas=embedded.rope_deltas,
        skip_logits=True, return_hidden=True,
        capture_layer_ids=list(range(model.config.text_config.num_hidden_layers)),
    )
    image_positions = mx.array(np.flatnonzero(np.array(ids[0]) == model.config.image_token_id))
    pooled = mx.stack([
        mx.stack([hidden[0, image_positions].mean(axis=0) for hidden in out.hidden_states]),
        mx.stack([hidden[0, -1] for hidden in out.hidden_states]),
    ])
    letters = None
    if letter_ids is not None:
        final = out.hidden_states[-1][0, -1][None]
        logits = language.model.embed_tokens.as_linear(final) if language.args.tie_word_embeddings else language.lm_head(final)
        letters = logits[0, mx.array(letter_ids)].astype(mx.float32)
        mx.eval(pooled, letters)
        letters = letters.tolist()
    else:
        mx.eval(pooled)
    return np.array(pooled.astype(mx.float32)).astype(np.float16), letters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--grid-tag", default="jev-qwen35-08b-4bit_g2_clips_l60_c8_s42")
    parser.add_argument("--grid-dir", type=Path, default=ROOT / "data" / "debug" / "jev_grids")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "embeddings")
    parser.add_argument("--model-path", type=Path, default=JEV_ROOT / ".models" / "Qwen3.5-0.8B-4bit")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(JEV_ROOT))
    from jev_visual.engine import Engine
    from jev_visual.preprocessing import build_prompts, read_image
    from jev_visual.schema import Question, Request

    rows = [row for split in SPLITS for row in read_jsonl(args.split_dir / f"{split}.jsonl")]
    grid_dir = args.grid_dir / args.grid_tag
    out_dir = args.cache_dir / f"{args.grid_tag}_prompt-{args.prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (out_dir / f"{row['video_id']}.npy").is_file()]
    print(f"features={out_dir.name} videos={len(rows)} pending={len(pending)}")
    if not pending:
        return

    adapter = Engine(model_path=args.model_path).adapter
    domains = split_prompts(args.prompt, args.split_dir)
    question = Question(type="choice", instructions=QUESTION, criteria={name: domains[name] for name in sorted(domains)})
    first_image = next(grid_dir.glob(f"{pending[0]['video_id']}_c0_*.jpg"))
    prefix, plans = build_prompts(adapter.processor, Request(image=str(first_image), questions={"domain": question}))
    prompt = prefix + plans[0].suffix

    with adapter.execution_context():
        for done, row in enumerate(pending, start=1):
            images = sorted(grid_dir.glob(f"{row['video_id']}_c*_*.jpg"), key=lambda p: int(p.name.split("_c")[-1].split("_")[0]))
            if not images:
                raise SystemExit(f"no grid images for {row['video_id']} in {grid_dir}; run jev_zero_shot.py first")
            features = np.stack([image_forward(adapter, prompt, read_image(str(path)))[0] for path in images])
            np.save(out_dir / f"{row['video_id']}.npy", features)
            print(f"embedded {done}/{len(pending)} {row['video_id']} {tuple(features.shape)}", flush=True)


if __name__ == "__main__":
    main()
