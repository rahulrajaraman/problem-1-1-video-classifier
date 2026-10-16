#!/usr/bin/env python3
"""Classify video domains zero-shot with quantized Qwen2.5-Omni-3B (GGUF) in llama.cpp.

The language model is 4-bit (Q4_K_M) and the vision/audio projector 8-bit,
from ggml-org/Qwen2.5-Omni-3B-GGUF, served by llama-server on the Mac GPU.
Only frames are sent, no audio. Each video's clip grid images, saved earlier
by jev_zero_shot.py (one 2x2 frame grid per clip), go in a single request in
time order, with the same lettered domain options as the jev-visual runs. The
model's log-probabilities for the option letters at the first answer token
are the scores; letters outside the server's top candidates get a floor.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import string
import subprocess
import time
import urllib.request
from pathlib import Path

from jev_zero_shot import PROMPTS, read_jsonl, report, split_prompts

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.parent / "models" / "qwen25-omni-3b-gguf"
SPLITS = ("train", "val", "test")
QUESTION = (
    "These {count} images come from one video, in time order. Each image shows four frames "
    "from one short clip. Which category does the video belong to?"
)


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def wait_for_server(base: str, process: subprocess.Popen, timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                if json.load(response).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("llama-server did not become healthy in time")


def letter_scores(response: dict, letters: list[str]) -> tuple[list[float], str]:
    """Log-probability per option letter from the first generated token's top candidates."""
    choice = response["choices"][0]
    top = choice["logprobs"]["content"][0]["top_logprobs"]
    best: dict[str, float] = {}
    for candidate in top:
        token = candidate["token"].strip()
        if token in letters:
            best[token] = max(best.get(token, -math.inf), candidate["logprob"])
    floor = min([candidate["logprob"] for candidate in top] + [0.0]) - 1.0
    return [best.get(letter, floor) for letter in letters], choice["message"]["content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--grid-tag", default="jev-qwen35-08b-4bit_g2_clips_l60_c8_s42")
    parser.add_argument("--grid-dir", type=Path, default=ROOT / "data" / "debug" / "jev_grids")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "zero_shot")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "Qwen2.5-Omni-3B-Q4_K_M.gguf")
    parser.add_argument("--mmproj", type=Path, default=MODEL_DIR / "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--max-videos", type=int, default=0, help="Use the first N videos of each split. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = split_prompts(args.prompt, args.split_dir)
    names = sorted(domains)
    letters = list(string.ascii_uppercase[: len(names)])
    options = "\n".join(f"{letter}. {name}: {domains[name]}" for letter, name in zip(letters, names))
    rows = []
    for split in SPLITS:
        split_rows = read_jsonl(args.split_dir / f"{split}.jsonl")
        if args.max_videos > 0:
            split_rows = split_rows[: args.max_videos]
        rows.extend({**row, "split": split} for row in split_rows)

    grid_dir = args.grid_dir / args.grid_tag
    tag = f"omni3b-gguf-q4km_{args.grid_tag.removeprefix('jev-qwen35-08b-4bit_')}_prompt-{args.prompt}"
    out_path = args.out_dir / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["video_id"]: record for record in read_jsonl(out_path)} if out_path.is_file() else {}
    pending = [row for row in rows if row["video_id"] not in done]
    print(f"run={tag} videos={len(rows)} pending={len(pending)}", flush=True)

    if pending:
        base = f"http://127.0.0.1:{args.port}"
        log = (ROOT / "results" / "logs" / "llama_server.log").open("w")
        server = subprocess.Popen(
            ["llama-server", "-m", str(args.model), "--mmproj", str(args.mmproj), "-c", str(args.ctx),
             "--port", str(args.port), "-ngl", "99", "--no-webui"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            wait_for_server(base, server)
            with out_path.open("a") as handle:
                for index, row in enumerate(pending, start=1):
                    images = sorted(grid_dir.glob(f"{row['video_id']}_c*_*.jpg"),
                                    key=lambda p: int(p.name.split("_c")[-1].split("_")[0]))
                    if not images:
                        raise SystemExit(f"no grid images for {row['video_id']} in {grid_dir}")
                    content = [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}}
                        for p in images
                    ]
                    content.append({"type": "text", "text": f"{QUESTION.format(count=len(images))}\n{options}\nAnswer with one option letter only."})
                    started = time.time()
                    response = post(f"{base}/v1/chat/completions", {
                        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": content}],
                        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                    })
                    scores, answer = letter_scores(response, letters)
                    prediction = names[max(range(len(names)), key=scores.__getitem__)]
                    record = {
                        **row,
                        "prediction": prediction,
                        "scores": {name: round(score, 4) for name, score in zip(names, scores)},
                        "raw_answer": answer,
                        "seconds": round(time.time() - started, 1),
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    done[row["video_id"]] = record
                    mark = "ok " if prediction == row["domain"] else "MISS"
                    print(f"{index}/{len(pending)} {mark} {row['video_id']} [{row['split']}/{row['duration']}] "
                          f"{record['seconds']}s answer={answer!r} true={row['domain']} pred={prediction}", flush=True)
        finally:
            server.terminate()
            server.wait(timeout=30)
            log.close()

    report([done[row["video_id"]] for row in rows if row["video_id"] in done], names)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
