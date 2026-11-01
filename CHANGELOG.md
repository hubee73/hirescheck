# Changelog / 更新记录

## 1.5.0 — unreleased / 未发布

- Output in Chinese or English: follows the system language, `--lang zh|en` overrides it (including `--help`).
  输出支持中文和英文：默认跟随系统语言，可用 `--lang zh|en` 指定（`--help` 也会切换）。
- `report.json`: new `language` field; likelihood grades are stored as codes (`very_high` … `unknown`) with a separate `level_text`.
  `report.json` 新增 `language` 字段；可能性等级改存为代码（`very_high` … `unknown`），文字另存在 `level_text`。
- Packaged as a Python package (`pip install .`, `pipx install .`, `python -m hirescheck`); `install.sh` is removed — install with pip/pipx.
  改为标准 Python 包（`pip install .`、`pipx install .`、`python -m hirescheck`）；删除 `install.sh`，统一用 pip/pipx 安装。
- Neutral wording for processing-chain matches; the built-in chain is named `chain-192k-a`.
  加工链吻合时的措辞改为中性；内置指纹命名为 `chain-192k-a`。
- MIT license, English README, CI (ruff + synthetic regression tests on Linux/macOS, Python 3.10/3.13).
  采用 MIT 许可证；新增英文 README 和 CI（在 Linux/macOS、Python 3.10/3.13 上跑 ruff 和合成样本回归测试）。

## 1.4.0

- Lossy-origin check for 44.1/48 kHz files (cut and spectral holes). 44.1/48kHz 文件的有损来源检查（截止和频谱空洞）。
- Plain-language conclusion with graded likelihoods at the top of every report. 每份报告开头加通俗结论和可能性等级。
- `--check-sr`, `--learn-chain`, `--list-chains`: processing-chain fingerprints. 加工链指纹检查与学习。

## 1.3.0 and earlier / 及更早

- Folder scans with per-file reports and summaries; upsampling and source-rate detection for PCM; PCM-origin and remodulation detection for DSD.
  文件夹批量检查与汇总；PCM 升频与来源采样率检测；DSD 的 PCM 来源与升频检测。
