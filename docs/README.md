# Documentation

Current behaviour of the tool. Tunables live in [`config.toml`](../config.toml);
the CLI is [`blur_plates.py`](../blur_plates.py).

| Doc | Contents |
|-----|----------|
| [pipeline.md](pipeline.md) | Detect → cache → interpolate → redact. What is in, what is out. |
| [config.md](config.md) | `config.toml` knobs and CLI flags. |
| [debug.md](debug.md) | `--debug` / `--debug-overlay` colours. |
| [FINETUNE_V2M.md](../FINETUNE_V2M.md) | Optional plate-model fine-tune (only if the shipped weights miss your camera). |

Install, first run, and troubleshooting stay in the [root README](../README.md).
