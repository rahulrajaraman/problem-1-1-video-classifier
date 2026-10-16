# Archived experiment outputs

Outputs from earlier methods (Omni, jev, CLAP, probes, fusion, text-init, …).
The **current** pipeline only writes eval artifacts to `results/mobileclip_zero_shot/`.

| Directory | Contents |
| --- | --- |
| `zero_shot/` | Per-video zero-shot JSONL (Omni, jev, CLAP, …) |
| `probe/`, `probe_5class/` | Linear probe sweeps |
| `text_prompts/`, `text_prompts_5class/` | Text-prompt / text-init runs |
| `fusion/` | MobileCLIP + CLAP score fusion |
| `logs/` | Full stdout logs |

Write-up of these runs: [`docs/experiments.md`](../docs/experiments.md).
Code: [`src/archive/`](../src/archive/).
