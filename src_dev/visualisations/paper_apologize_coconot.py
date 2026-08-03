"""Behavioural-effects 2-panel: (a) sycophancy Apologize Rate · (b) CoCoNot total compliance.

Both panels use the same six conditions in the same order — base, control,
then the two "agreeable-direction" variants (A− @ −1, A+ @ +1) followed
by the two "disagreeable-direction" variants (A+ @ −1, A− @ +1) — and a
single shared legend at the bottom of the figure.

Conditions
----------
- base                : Llama-3.1-8B-Instruct (no adapter)
- control              : ``ocean_def_control`` (vanton4_seed1) at scale +1
- A− @ −1              : suppressor flipped (effective amplification)
- A+ @ +1              : amplifier as intended
- A+ @ −1              : amplifier flipped (effective suppression)
- A− @ +1              : suppressor as intended

Panel (a) — Apologize Rate (sycophancy):
    Per-sample ``apologize_rate`` value from the upstream
    ``sycophancy_scorer``, restricted to the ``original_answer == "C"``
    subset (the upstream denominator).  Wilson 95% CIs.
    Sycophancy inspect logs hydrate from
    ``persona-cartography/monorepo`` and are cached at
    ``scratch/paper_plots_cache/sycophancy_a_six_bars/``.

Panel (b) — CoCoNot total compliance (lower = better):
    Verbatim ``total`` metric from
    ``results.scores[0].metrics['total']`` (no recompute), with Wilson
    95% CIs computed from per-sample ``UNACCEPTABLE`` labels (matches
    upstream ``original_compliance_value_to_float`` exactly).
    CoCoNot inspect logs hydrate from
    ``persona-cartography/monorepo`` (same provenance as sycophancy)
    and are cached at ``scratch/paper_plots_cache/coconot_a_plus_minus_control/``.

Run with::

    uv run python -m src_dev.visualisations.paper_apologize_coconot

A gemma-3-27b-it replica of the same figure (same six conditions, adapters
``ocean_const_paired_dpo`` A± + ``ocean_const_paired_dpo_s1vs2`` control;
+1/base/control runs from 2026-06-16, flipped-scale −1 runs from 2026-07-15)::

    uv run python -m src_dev.visualisations.paper_apologize_coconot --model gemma27b
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import (  # noqa: E402
    _interval_ci_from_wilson,
)
from src_dev.utils.hf_hub import login_from_env  # noqa: E402
from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402


PAPER_FIGURES = [
    "main/fig_apologize_coconot.pdf",
    "main/fig_apologize_coconot_gemma27b.pdf",
    "main/fig_apologize_coconot_qwen32b.pdf",
    "main/fig_apologize_coconot_qwen8b.pdf",
]


# Style — inlined from STYLE_GUIDE.md (matches paper_sycophancy_a_six_bars.py).
PAPER_STYLE: dict[str, object] = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.facecolor": "#fbfbfc",
    "axes.edgecolor": "#2f3748",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#2f3748",
    "ytick.color": "#2f3748",
    "grid.color": "#dfe3e8",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.75,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cfd4dc",
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
}

SPINE_COLOR = "#2f3748"
# Semantic colors from STYLE_GUIDE.md — blue for the "agreeable" direction
# (preserved/expected behavior), red for the "disagreeable" direction
# (injection / behavioural failure mode).
C_ORGANIC = "#3c7fb1"
C_INJECTED = "#c91546"

HF_REPO_ID = "persona-cartography/monorepo"
_CACHE_ROOT = project_root / "scratch" / "paper_plots_cache"


# ── Conditions ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Condition:
    key: str
    short: str
    legend: str
    color: str
    hatch: str | None
    syc_log_in_repo: str          # HF path for sycophancy inspect log
    coconot_log_in_repo: str      # HF path for coconot inspect log


_LLAMA_CONDITIONS: list[Condition] = [
    Condition(
        key="base",
        short="base",
        legend="base",
        color="#4D4D4D",
        hatch=None,
        syc_log_in_repo=(
            "evals/sycophancy/llama-3.1-8b-it_base/base_llama_3_1_8b_it/base/"
            "sycophancy/native/inspect_logs/"
            "2026-04-29T18-44-01+00-00_sycophancy_i2Xzh5RirRoMixwGTugtPL.json"
        ),
        coconot_log_in_repo=(
            "evals/baselines/llama-3.1-8b-instruct/coconot/"
            "native/inspect_logs/"
            "2026-05-01T15-50-25+00-00_coconot_GhdMhJyadEXqtUh7SrpnYh.json"
        ),
    ),
    Condition(
        key="control",
        short="control",
        legend="control",
        color="#9E9E9E",
        hatch=None,
        syc_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
            "vanton4_paired_dpo_s1vs2/evals/mcq/sycophancy/"
            "control_vanton4_paired_dpo_s1vs2_scale1/lora_+1p00x/sycophancy/"
            "native/inspect_logs/"
            "2026-05-03T14-42-22+00-00_sycophancy_D2sxmLu2rH6NEMdeejsTot.json"
        ),
        coconot_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
            "vanton4_paired_dpo_s1vs2/evals/coconot/"
            "control_ocean_def_vanton4_paired_dpo_s1vs2/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-05-03T14-40-25+00-00_coconot_KLbiUx6eaBuaju95dcXFno.json"
        ),
    ),
    # Agreeable-direction pair: same colour (C_ORGANIC), hatched on the flipped-scale path.
    Condition(
        key="a_minus_m1",
        short="A↓ @ −1",
        legend="A↓ @ scale −1",
        color=C_ORGANIC,
        hatch="///",
        syc_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_minus_vanton4_paired_dpo_scale-1/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-05-01T13-44-46+00-00_sycophancy_6ENuoAEfMCRJitTKYu8iQf.json"
        ),
        coconot_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/coconot/"
            "a_minus_vanton4_paired_dpo_minus1/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-05-01T20-34-08+00-00_coconot_kB7Dh8M8Hj9L26i7aDMAuf.json"
        ),
    ),
    Condition(
        key="a_plus_p1",
        short="A↑ @ +1",
        legend="A↑ @ scale +1",
        color=C_ORGANIC,
        hatch=None,
        syc_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_plus_vanton4_paired_dpo_scale1/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-04-29T12-46-20+00-00_sycophancy_oWRZ6NnD8gNk69CpMtzMHw.json"
        ),
        coconot_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/coconot/"
            "a_plus_vanton4_paired_dpo/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-05-01T16-49-12+00-00_coconot_2ztYCYTz8mJDXqmKqCNbLp.json"
        ),
    ),
    # Disagreeable-direction pair: same colour (C_INJECTED), hatched on the flipped-scale path.
    Condition(
        key="a_plus_m1",
        short="A↑ @ −1",
        legend="A↑ @ scale −1",
        color=C_INJECTED,
        hatch="///",
        syc_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_plus_vanton4_paired_dpo_scale-1/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-05-01T13-44-47+00-00_sycophancy_CL6MLEXn5PT78pkeUayEkQ.json"
        ),
        coconot_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/coconot/"
            "a_plus_vanton4_paired_dpo/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-05-01T16-21-46+00-00_coconot_Zt55TjtQYntZ9pUeCBwmHZ.json"
        ),
    ),
    Condition(
        key="a_minus_p1",
        short="A↓ @ +1",
        legend="A↓ @ scale +1",
        color=C_INJECTED,
        hatch=None,
        syc_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_minus_vanton4_paired_dpo_scale1/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-04-29T12-46-20+00-00_sycophancy_MF7EmrLtVaBcNZDuU4Na5J.json"
        ),
        coconot_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/coconot/"
            "a_minus_vanton4_paired_dpo_plus1/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-05-01T20-34-08+00-00_coconot_PnbDefC6ALLBCr9t9qm4ou.json"
        ),
    ),
]


# gemma-3-27b-it replica: same six conditions with the ocean_const_paired_dpo
# A± adapters and the ocean_const_paired_dpo_s1vs2 control. +1 / base / control
# logs from the 2026-06-16 gemma_downstream runs; −1 (flipped) logs from the
# 2026-07-15 pod run. NB the base sycophancy log was uploaded under the
# suppressor's coconot eval prefix by the June run — pinned where it lives.
_GEMMA27B_FT = "fine_tuning/gemma-3-27b-it/ocean/agreeableness"
_GEMMA27B_CTRL = (
    "fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/"
    "ocean_const_paired_dpo_s1vs2"
)

_GEMMA27B_CONDITIONS: list[Condition] = [
    Condition(
        key="base",
        short="base",
        legend="base",
        color="#4D4D4D",
        hatch=None,
        syc_log_in_repo=(
            f"{_GEMMA27B_FT}/suppressor/ocean_const_paired_dpo/evals/coconot/"
            "gemma-3-27b-it_agree_sup/base/sycophancy/gemma-3-27b-it_agree_base/"
            "base/sycophancy/native/inspect_logs/"
            "2026-06-16T19-06-14+00-00_sycophancy_Giax6gtMFhGk7HxgSFZoFc.json"
        ),
        coconot_log_in_repo=(
            "evals/baselines/gemma-3-27b-it/coconot/native/inspect_logs/"
            "2026-06-16T13-46-24+00-00_coconot_QcfDAJipTnvvdSv5Caw7s4.json"
        ),
    ),
    Condition(
        key="control",
        short="control",
        legend="control",
        color="#9E9E9E",
        hatch=None,
        syc_log_in_repo=(
            f"{_GEMMA27B_CTRL}/evals/mcq/sycophancy/gemma-3-27b-it_agree_control/"
            "lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-06-16T20-31-53+00-00_sycophancy_FC9tm9bkn9ybuy97ZjXTcX.json"
        ),
        coconot_log_in_repo=(
            f"{_GEMMA27B_CTRL}/evals/coconot/gemma-3-27b-it_agree_control/"
            "lora_+1p00x/coconot/native/inspect_logs/"
            "2026-06-16T14-14-52+00-00_coconot_8ngMEVZi2HmocN7vuA4qUM.json"
        ),
    ),
    Condition(
        key="a_minus_m1",
        short="A↓ @ −1",
        legend="A↓ @ scale −1",
        color=C_ORGANIC,
        hatch="///",
        syc_log_in_repo=(
            f"{_GEMMA27B_FT}/suppressor/ocean_const_paired_dpo/evals/mcq/"
            "sycophancy/gemma-3-27b-it_agree_sup/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-07-15T19-13-24+00-00_sycophancy_jX5tn3vMiu9KGTqFcdLRZH.json"
        ),
        coconot_log_in_repo=(
            f"{_GEMMA27B_FT}/suppressor/ocean_const_paired_dpo/evals/coconot/"
            "gemma-3-27b-it_agree_sup/lora_-1p00x/coconot/native/inspect_logs/"
            "2026-07-15T17-56-55+00-00_coconot_kgJPxNE7VacrdNMgZ3ktVZ.json"
        ),
    ),
    Condition(
        key="a_plus_p1",
        short="A↑ @ +1",
        legend="A↑ @ scale +1",
        color=C_ORGANIC,
        hatch=None,
        syc_log_in_repo=(
            f"{_GEMMA27B_FT}/amplifier/ocean_const_paired_dpo/evals/mcq/"
            "sycophancy/gemma-3-27b-it_agree_amp/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-06-16T19-44-35+00-00_sycophancy_krVht2kpERNboSJZhAUNu4.json"
        ),
        coconot_log_in_repo=(
            f"{_GEMMA27B_FT}/amplifier/ocean_const_paired_dpo/evals/coconot/"
            "gemma-3-27b-it_agree_amp/lora_+1p00x/coconot/native/inspect_logs/"
            "2026-06-16T10-39-37+00-00_coconot_dzjEiyMqJttp5w7umffEAw.json"
        ),
    ),
    Condition(
        key="a_plus_m1",
        short="A↑ @ −1",
        legend="A↑ @ scale −1",
        color=C_INJECTED,
        hatch="///",
        syc_log_in_repo=(
            f"{_GEMMA27B_FT}/amplifier/ocean_const_paired_dpo/evals/mcq/"
            "sycophancy/gemma-3-27b-it_agree_amp/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-07-15T18-31-55+00-00_sycophancy_NQ6jt7HVCzi6qjRZhYFDAm.json"
        ),
        coconot_log_in_repo=(
            f"{_GEMMA27B_FT}/amplifier/ocean_const_paired_dpo/evals/coconot/"
            "gemma-3-27b-it_agree_amp/lora_-1p00x/coconot/native/inspect_logs/"
            "2026-07-15T17-26-44+00-00_coconot_BqzLq3w3aKwa9Uqjo9nGvS.json"
        ),
    ),
    Condition(
        key="a_minus_p1",
        short="A↓ @ +1",
        legend="A↓ @ scale +1",
        color=C_INJECTED,
        hatch=None,
        syc_log_in_repo=(
            f"{_GEMMA27B_FT}/suppressor/ocean_const_paired_dpo/evals/mcq/"
            "sycophancy/gemma-3-27b-it_agree_sup/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-06-16T19-58-47+00-00_sycophancy_Tjqy4zGivAwUqntKJiSsW3.json"
        ),
        coconot_log_in_repo=(
            f"{_GEMMA27B_FT}/suppressor/ocean_const_paired_dpo/evals/coconot/"
            "gemma-3-27b-it_agree_sup/lora_+1p00x/coconot/native/inspect_logs/"
            "2026-06-16T22-15-10+00-00_coconot_nMKkrSMW2FoHzYGN7rsN7N.json"
        ),
    ),
]


# qwen-3-32b-it replica: FULL-SET runs (2026-07-16/17; coconot 1001, sycophancy 4882), nothink adapters
# (ocean_const_paired_dpo_nothink A± + _nothink_s1vs2 control), thinking
# disabled at eval. CoCoNot via the src.evals suite; sycophancy via the
# vllm runtime-LoRA runner (run_sycophancy_vllm_lora.py) for all six bars.
_QWEN32B_FT = "fine_tuning/qwen-3-32b-it/ocean/agreeableness"
_QWEN32B_CTRL = (
    "fine_tuning/qwen-3-32b-it/other/ocean_def_control/amplifier/"
    "ocean_const_paired_dpo_nothink_s1vs2"
)
_QWEN32B_SYCO_RUN = "evals/downstream/qwen-3-32b-it_agree_syco_full"

_QWEN32B_CONDITIONS: list[Condition] = [
    Condition(
        key="base",
        short="base",
        legend="base",
        color="#4D4D4D",
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN32B_SYCO_RUN}/base/sycophancy/native/inspect_logs/"
            "2026-07-16T18-12-48+00-00_sycophancy_JdwRFX3Uf6KDqnxPddHeNc.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-32b-it_agree_amp_full/base/coconot/"
            "native/inspect_logs/"
            "2026-07-16T22-23-40+00-00_coconot_W8RGeEGNqJabJFsJops276.json"
        ),
    ),
    Condition(
        key="control",
        short="control",
        legend="control",
        color="#9E9E9E",
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN32B_CTRL}/{_QWEN32B_SYCO_RUN}/lora_+1p00x/sycophancy/"
            "native/inspect_logs/"
            "2026-07-16T21-41-30+00-00_sycophancy_YAh9xbjzRgZgtRMaZJSPuc.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_CTRL}/evals/downstream/qwen-3-32b-it_agree_control_full/"
            "lora_+1p00x/coconot/native/inspect_logs/"
            "2026-07-17T00-39-02+00-00_coconot_WcNLjk2eax7Vd5oHc6MErg.json"
        ),
    ),
    Condition(
        key="a_minus_m1",
        short="A↓ @ −1",
        legend="A↓ @ scale −1",
        color=C_ORGANIC,
        hatch="///",
        syc_log_in_repo=(
            f"{_QWEN32B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            f"{_QWEN32B_SYCO_RUN}/lora_-1p00x/sycophancy/native/inspect_logs/"
            "2026-07-16T20-56-20+00-00_sycophancy_Y4Xt8HR87FhckAD3RYFqk9.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-32b-it_agree_sup_full/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-16T23-43-19+00-00_coconot_oGJCWVbH6wSpmWsxfwF8JP.json"
        ),
    ),
    Condition(
        key="a_plus_p1",
        short="A↑ @ +1",
        legend="A↑ @ scale +1",
        color=C_ORGANIC,
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN32B_SYCO_RUN}/lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-07-16T18-49-18+00-00_sycophancy_Kev32bPjuDZxBACVLJdkVS.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-32b-it_agree_amp_full/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-16T23-16-16+00-00_coconot_eYFPrtnQ9wuCeaKeg3i6VP.json"
        ),
    ),
    Condition(
        key="a_plus_m1",
        short="A↑ @ −1",
        legend="A↑ @ scale −1",
        color=C_INJECTED,
        hatch="///",
        syc_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN32B_SYCO_RUN}/lora_-1p00x/sycophancy/native/inspect_logs/"
            "2026-07-16T19-27-09+00-00_sycophancy_R4fYSJcN2PxrEaB5P4Ykyj.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-32b-it_agree_amp_full/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-16T22-50-03+00-00_coconot_F3THfZwh9MkgfXuyNV7GCc.json"
        ),
    ),
    Condition(
        key="a_minus_p1",
        short="A↓ @ +1",
        legend="A↓ @ scale +1",
        color=C_INJECTED,
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN32B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            f"{_QWEN32B_SYCO_RUN}/lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-07-16T20-18-58+00-00_sycophancy_dEids3FAHGZ5BGL3sGgrHZ.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN32B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-32b-it_agree_sup_full/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-17T00-09-02+00-00_coconot_W3CExqtv9HZvqbyewYzFK2.json"
        ),
    ),
]


# qwen-3-8b-it replica: FULL-SET runs (2026-07-17; coconot 1001, sycophancy
# 4882), nothink adapters, same recipe/runners as qwen-3-32b.
_QWEN8B_FT = "fine_tuning/qwen-3-8b-it/ocean/agreeableness"
_QWEN8B_CTRL = (
    "fine_tuning/qwen-3-8b-it/other/ocean_def_control/amplifier/"
    "ocean_const_paired_dpo_nothink_s1vs2"
)
_QWEN8B_SYCO_RUN = "evals/downstream/qwen-3-8b-it_agree_syco_full"

_QWEN8B_CONDITIONS: list[Condition] = [
    Condition(
        key="base",
        short="base",
        legend="base",
        color="#4D4D4D",
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN8B_SYCO_RUN}/base/sycophancy/native/inspect_logs/"
            "2026-07-17T11-07-38+00-00_sycophancy_VaChRMU4kU7N6hjVZsoUZx.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-8b-it_agree_amp_full/base/coconot/"
            "native/inspect_logs/"
            "2026-07-17T14-21-42+00-00_coconot_PdFsmWq8zRJwzBaVwNjcnp.json"
        ),
    ),
    Condition(
        key="control",
        short="control",
        legend="control",
        color="#9E9E9E",
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN8B_CTRL}/{_QWEN8B_SYCO_RUN}/lora_+1p00x/sycophancy/"
            "native/inspect_logs/"
            "2026-07-17T13-48-44+00-00_sycophancy_aCVnTeKs4mruuya2qJpNtm.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_CTRL}/evals/downstream/qwen-3-8b-it_agree_control_full/"
            "lora_+1p00x/coconot/native/inspect_logs/"
            "2026-07-17T16-30-51+00-00_coconot_nSkWf7ZD2LBn66BfFEUSFA.json"
        ),
    ),
    Condition(
        key="a_minus_m1",
        short="A↓ @ −1",
        legend="A↓ @ scale −1",
        color=C_ORGANIC,
        hatch="///",
        syc_log_in_repo=(
            f"{_QWEN8B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            f"{_QWEN8B_SYCO_RUN}/lora_-1p00x/sycophancy/native/inspect_logs/"
            "2026-07-17T13-18-38+00-00_sycophancy_PoxvE5c4xVjVeLdAgeKxui.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-8b-it_agree_sup_full/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-17T15-45-30+00-00_coconot_QTN6UvMkTo89GXnrGTyFm5.json"
        ),
    ),
    Condition(
        key="a_plus_p1",
        short="A↑ @ +1",
        legend="A↑ @ scale +1",
        color=C_ORGANIC,
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN8B_SYCO_RUN}/lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-07-17T11-35-17+00-00_sycophancy_6cqUJQsyHbt3jEFetk5VFU.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-8b-it_agree_amp_full/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-17T15-23-05+00-00_coconot_UFBTHX3MKsLzcSHbdi7vBF.json"
        ),
    ),
    Condition(
        key="a_plus_m1",
        short="A↑ @ −1",
        legend="A↑ @ scale −1",
        color=C_INJECTED,
        hatch="///",
        syc_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            f"{_QWEN8B_SYCO_RUN}/lora_-1p00x/sycophancy/native/inspect_logs/"
            "2026-07-17T12-04-56+00-00_sycophancy_E6NKTePHpT6whB98uwLnVV.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_FT}/amplifier/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-8b-it_agree_amp_full/lora_-1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-17T17-03-33+00-00_coconot_ek5bM2XFnR6TasBr9kw4Ez.json"
        ),
    ),
    Condition(
        key="a_minus_p1",
        short="A↓ @ +1",
        legend="A↓ @ scale +1",
        color=C_INJECTED,
        hatch=None,
        syc_log_in_repo=(
            f"{_QWEN8B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            f"{_QWEN8B_SYCO_RUN}/lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-07-17T12-40-18+00-00_sycophancy_WtREhqsMJAbcJ9GpGY4NQv.json"
        ),
        coconot_log_in_repo=(
            f"{_QWEN8B_FT}/suppressor/ocean_const_paired_dpo_nothink/"
            "evals/downstream/qwen-3-8b-it_agree_sup_full/lora_+1p00x/coconot/"
            "native/inspect_logs/"
            "2026-07-17T16-06-33+00-00_coconot_P36KhYTiQi38emnDZ2zaFB.json"
        ),
    ),
]


@dataclass(frozen=True)
class ModelVariant:
    """One base model's condition set + cache dirs + output figure name."""

    conditions: list[Condition]
    syc_cache_dir: Path
    coconot_cache_dir: Path
    figure_name: str


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "llama": ModelVariant(
        conditions=_LLAMA_CONDITIONS,
        syc_cache_dir=_CACHE_ROOT / "sycophancy_a_six_bars",
        coconot_cache_dir=_CACHE_ROOT / "coconot_a_plus_minus_control",
        figure_name="fig_apologize_coconot.pdf",
    ),
    "gemma27b": ModelVariant(
        conditions=_GEMMA27B_CONDITIONS,
        syc_cache_dir=_CACHE_ROOT / "sycophancy_a_six_bars_gemma27b",
        coconot_cache_dir=_CACHE_ROOT / "coconot_a_plus_minus_control_gemma27b",
        figure_name="fig_apologize_coconot_gemma27b.pdf",
    ),
    "qwen32b": ModelVariant(
        conditions=_QWEN32B_CONDITIONS,
        syc_cache_dir=_CACHE_ROOT / "sycophancy_a_six_bars_qwen32b_full",
        coconot_cache_dir=_CACHE_ROOT / "coconot_a_plus_minus_control_qwen32b_full",
        figure_name="fig_apologize_coconot_qwen32b.pdf",
    ),
    "qwen8b": ModelVariant(
        conditions=_QWEN8B_CONDITIONS,
        syc_cache_dir=_CACHE_ROOT / "sycophancy_a_six_bars_qwen8b_full",
        coconot_cache_dir=_CACHE_ROOT / "coconot_a_plus_minus_control_qwen8b_full",
        figure_name="fig_apologize_coconot_qwen8b.pdf",
    ),
}

# Backward-compat alias (the llama figure is the original).
CONDITIONS = _LLAMA_CONDITIONS


# ── Sycophancy: hydrate + extract apologize_rate ───────────────────────────


def _hydrate_syc_log(cond: Condition, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cond.syc_log_in_repo.rsplit("/", 1)[1]
    local = cache_dir / cond.key / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"  hydrating sycophancy log for {cond.key} from HF...")
    fetched = hf_hub_download(
        repo_id=HF_REPO_ID, filename=cond.syc_log_in_repo, repo_type="dataset"
    )
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    local.symlink_to(fetched)
    return local


def _syc_apologize_rate(log_path: Path) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for ``apologize_rate`` on the orig=="C" subset."""
    log = read_eval_log(str(log_path))
    apo: list[int] = []
    for s in log.samples or []:
        sc = (s.scores or {}).get("sycophancy_scorer")
        if sc is None:
            continue
        d = sc.value or {}
        if d.get("original_answer") != "C":
            continue
        apo.append(int(float(d.get("apologize_rate", 0.0))))
    arr = np.asarray(apo, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    lo, hi = _interval_ci_from_wilson(arr, confidence=95.0)
    return mean, max(lo, 0.0), max(hi, 0.0)


# ── CoCoNot: read verbatim total + Wilson CI from per-sample labels ────────


def _resolve_coconot_log(cond: Condition, cache_dir: Path) -> Path:
    """Hydrate the coconot inspect log from the monorepo on HF (cached locally)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cond.coconot_log_in_repo.rsplit("/", 1)[1]
    cached = cache_dir / cond.key / filename
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    print(f"  hydrating coconot log for {cond.key} from HF...")
    fetched = hf_hub_download(
        repo_id=HF_REPO_ID, filename=cond.coconot_log_in_repo, repo_type="dataset"
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        cached.unlink()
    cached.symlink_to(fetched)
    return cached


def _coconot_total_with_ci(log_path: Path) -> tuple[float, float, float]:
    """Return (verbatim total, ci_lo, ci_hi). Sanity-checks per-sample mean."""
    raw = json.loads(log_path.read_text())
    metrics = raw["results"]["scores"][0]["metrics"]
    verbatim = float(metrics["total"]["value"]) / 100.0

    log = read_eval_log(str(log_path))
    samples: list[int] = []
    for s in log.samples or []:
        scores = s.scores or {}
        if not scores:
            continue
        sc = next(iter(scores.values()))
        samples.append(1 if str(sc.value).lower() == "unacceptable" else 0)
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return verbatim, float("nan"), float("nan")
    sample_mean = float(arr.mean())
    if not np.isclose(sample_mean, verbatim, atol=1e-6):
        raise RuntimeError(
            f"verbatim total {verbatim:.6f} != sample mean {sample_mean:.6f}"
        )
    lo, hi = _interval_ci_from_wilson(arr, confidence=95.0)
    return verbatim, max(lo, 0.0), max(hi, 0.0)


# ── Drawing ────────────────────────────────────────────────────────────────


def _draw_panel(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    values: list[float],
    err_lo: list[float],
    err_hi: list[float],
    conditions: list[Condition],
) -> None:
    n = len(conditions)
    x = np.arange(n, dtype=float)
    width = 0.96  # densely-attached: bars almost touch one another

    bars = ax.bar(
        x,
        values,
        width=width,
        color=[c.color for c in conditions],
        alpha=0.92,
        edgecolor=SPINE_COLOR,
        linewidth=0.5,
        zorder=3,
    )
    for bar, cond in zip(bars, conditions):
        if cond.hatch:
            bar.set_hatch(cond.hatch)

    ax.errorbar(
        x,
        values,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor=SPINE_COLOR,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        alpha=0.85,
        zorder=4,
    )

    for xi, v, hi in zip(x, values, err_hi):
        ax.text(
            xi,
            v + hi + 0.012,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=SPINE_COLOR,
            fontweight="semibold",
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.short for c in conditions], rotation=20, ha="right",
                       fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    y_top = max(0.5, float((np.array(values) + np.array(err_hi)).max()) * 1.20)
    ax.set_ylim(0.0, y_top)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=8, fontsize=11)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)


def _legend_handles(conditions: list[Condition]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=cond.color,
            edgecolor=SPINE_COLOR,
            linewidth=0.5,
            alpha=0.92,
            hatch=cond.hatch or "",
            label=cond.legend,
        )
        for cond in conditions
    ]


def main(model: str = "llama") -> None:
    try:
        login_from_env()
    except RuntimeError:
        pass

    variant = MODEL_VARIANTS[model]
    conditions = variant.conditions
    out_path = PAPER_FIGURES_DIR / "main" / variant.figure_name

    syc: dict[str, tuple[float, float, float]] = {}
    coc: dict[str, tuple[float, float, float]] = {}
    for cond in conditions:
        syc_log = _hydrate_syc_log(cond, variant.syc_cache_dir)
        syc[cond.key] = _syc_apologize_rate(syc_log)
        coc_log = _resolve_coconot_log(cond, variant.coconot_cache_dir)
        coc[cond.key] = _coconot_total_with_ci(coc_log)

    # Print a small table for the run log.
    header = (
        f"{'condition':<14}  "
        f"{'apo':>6} {'apo_lo':>6} {'apo_hi':>6}    "
        f"{'coc':>6} {'coc_lo':>6} {'coc_hi':>6}"
    )
    print(header)
    print("-" * len(header))
    for cond in conditions:
        a, alo, ahi = syc[cond.key]
        c, clo, chi = coc[cond.key]
        print(
            f"{cond.short:<14}  "
            f"{a:>6.3f} {alo:>6.3f} {ahi:>6.3f}    "
            f"{c:>6.3f} {clo:>6.3f} {chi:>6.3f}"
        )

    mpl.rcParams.update(PAPER_STYLE)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    _draw_panel(
        ax_a,
        title="(a) Apologize Rate — caves under push-back",
        ylabel="Apologize rate (lower = better)",
        values=[syc[c.key][0] for c in conditions],
        err_lo=[max(syc[c.key][0] - syc[c.key][1], 0.0) for c in conditions],
        err_hi=[max(syc[c.key][2] - syc[c.key][0], 0.0) for c in conditions],
        conditions=conditions,
    )
    _draw_panel(
        ax_b,
        title="(b) CoCoNot compliance — fails refusal",
        ylabel="Total compliance rate (lower = better)",
        values=[coc[c.key][0] for c in conditions],
        err_lo=[max(coc[c.key][0] - coc[c.key][1], 0.0) for c in conditions],
        err_hi=[max(coc[c.key][2] - coc[c.key][0], 0.0) for c in conditions],
        conditions=conditions,
    )

    fig.legend(
        handles=_legend_handles(conditions),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
    )
    fig.subplots_adjust(top=0.90, bottom=0.22, wspace=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n✓ saved {out_path}")
    print(f"✓ saved {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_VARIANTS),
        default="llama",
        help="Which base model's condition set to plot (default: llama).",
    )
    main(parser.parse_args().model)
