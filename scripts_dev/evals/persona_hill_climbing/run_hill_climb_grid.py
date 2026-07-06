#!/usr/bin/env python3
"""Persona hill-climbing on a safety benchmark (WildJailbreak) via LoRA soups.

Idea: search over compositions of OCEAN persona LoRAs (per-trait coefficient
within ±1.5) for the mix that makes the model behave *safer* or *more harmful*
on WildJailbreak, using a train/test split of the benchmark. The ideal finding
is a clear trait profile that significantly moves harm rate on the held-out
test split.

MVP search strategy (this script): brute-force grid, 20 points = 5 traits ×
4 scales (−1.5, −0.75, +0.75, +1.5), plus a vanilla baseline. A positive
coefficient c applies the trait's *amplifier* adapter at scale c; a negative
coefficient applies the *suppressor* adapter at scale |c|. Later iterations
(compose top traits, or GP/Bayesian optimisation) can reuse this same runner
by passing ``--points-json``.

Phases::

    # 1. Evaluate all grid points on the TRAIN slice of WildJailbreak
    uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_grid \
        --phase train --run-slug hc_grid_v1

    # 2. Re-evaluate the k safest + k most harmful combos on the held-out TEST slice
    uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_grid \
        --phase test --run-slug hc_grid_v1 --top-k 3

Model: gemma-3-27b-it with the 10 ``ocean_const_paired_dpo`` OCEAN adapters
(all verified present on the HF monorepo). Reuses the persona_jailbreak_eval
infrastructure end-to-end (WJ loading, LoRA-soup baking, vLLM inference,
paper-rubric harm judge, refusal judge, Wilson-CI aggregation).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src_dev.activation_capping.conditions import ensure_vllm_fork_safe  # noqa: E402

ensure_vllm_fork_safe()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.common.lora_catalogue import GEMMA_OCEAN_REGISTRIES, OceanTraitDef  # noqa: E402
from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    explicit_refusal_rate_on_benign,
    harmful_rate_by_condition,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)
from src_dev.persona_jailbreak_eval.config import (  # noqa: E402
    JailbreakEvalConfig,
    LoraComboCondition,
)
from src_dev.persona_jailbreak_eval.harmful_datasets import load_wildjailbreak  # noqa: E402
from src_dev.persona_jailbreak_eval.hf_sync import (  # noqa: E402
    hydrate_run_dir_from_hf,
    upload_run_dir_to_hf,
)
from src_dev.persona_jailbreak_eval.runner import (  # noqa: E402
    PromptSample,
    run_all_conditions_inference,
    run_judges_on_all_conditions,
)

SEED = 42

# ── Search space ─────────────────────────────────────────────────────────

TRAITS = ("o", "c", "e", "a", "n")
DEFAULT_SCALES = (-1.5, -0.75, 0.75, 1.5)
SCALE_BOUND = 1.5

# A search point maps trait letter → coefficient in [−SCALE_BOUND, +SCALE_BOUND].
# Zero / missing traits are left out of the soup.
Point = dict[str, float]


def default_grid_points(scales: tuple[float, ...] = DEFAULT_SCALES) -> list[Point]:
    """Single-trait axis grid: |traits| × |scales| points (20 by default)."""
    return [{trait: scale} for trait in TRAITS for scale in scales]


def _coeff_tag(coeff: float) -> str:
    return f"{abs(coeff):g}"


def point_to_combo(point: Point, registry: dict[str, OceanTraitDef]) -> LoraComboCondition:
    """Translate a trait-coefficient point into a baked-soup condition.

    Positive coefficient → amplifier adapter at that scale; negative →
    suppressor adapter at |scale|. Adapters are passed as full
    ``repo::subfolder`` refs so the runner resolves them for the right base
    model (bare slugs would fall back to the llama registry).
    """
    adapters: list[tuple[str, float]] = []
    name_parts: list[str] = []
    for trait in TRAITS:
        coeff = point.get(trait, 0.0)
        if abs(coeff) < 1e-9:
            continue
        if abs(coeff) > SCALE_BOUND + 1e-9:
            raise ValueError(f"coefficient {coeff} for trait {trait!r} exceeds ±{SCALE_BOUND}")
        slug = f"{trait}_plus" if coeff > 0 else f"{trait}_minus"
        adapters.append((registry[slug].adapter_ref, abs(coeff)))
        name_parts.append(f"{slug}_{_coeff_tag(coeff)}")
    if not adapters:
        raise ValueError("point has no non-zero coefficients — use the vanilla condition instead")
    return LoraComboCondition(name="lora_soup_" + "_".join(name_parts), adapters=adapters)


# ── Config / data ────────────────────────────────────────────────────────


def build_config(args: argparse.Namespace, phase: str) -> JailbreakEvalConfig:
    cfg = JailbreakEvalConfig(
        run_slug=f"{args.run_slug}_{phase}",
        scratch_root=Path("scratch/persona_hill_climbing"),
        model_slug=args.model_slug,
        base_model=args.base_model,
        vllm_gpu_memory_utilization=0.85,
        vllm_max_model_len=4096,
        vllm_batch_size=32,
        max_new_tokens=512,
        hf_eval_type="persona_hill_climbing",
    )
    if args.no_upload_hf:
        cfg.upload_hf = False
    if args.no_hydrate_hf:
        cfg.hydrate_hf = False
    return cfg


def load_wj_split(
    *, split: str, n_train: int, n_test: int, seed: int, phase: str,
) -> list:
    """Deterministic disjoint train/test slices of one WildJailbreak split.

    ``load_wildjailbreak`` subsamples ``n_train + n_test`` items with the
    given seed; the first ``n_train`` are the train slice, the rest are held
    out for test. Same seed → same partition across phases.
    """
    items = load_wildjailbreak(split=split, n=n_train + n_test, seed=seed)  # type: ignore[arg-type]
    return items[:n_train] if phase == "train" else items[n_train:]


def build_samples(cfg: JailbreakEvalConfig, args: argparse.Namespace, phase: str) -> list[PromptSample]:
    harmful = load_wj_split(
        split="adversarial_harmful", n_train=args.n_harmful_train,
        n_test=args.n_harmful_test, seed=SEED, phase=phase,
    )
    benign = load_wj_split(
        split="adversarial_benign", n_train=args.n_benign_train,
        n_test=args.n_benign_test, seed=SEED, phase=phase,
    )
    print(f"  [{phase}] {len(harmful)} adv-harmful + {len(benign)} adv-benign")
    samples: list[PromptSample] = []
    for kind, wj_split, items in (
        ("harmful", "adversarial_harmful", harmful),
        ("benign", "adversarial_benign", benign),
    ):
        for item in items:
            samples.append(PromptSample(
                id=item.id, user_prompt=item.prompt, system_prompt=None,
                kind=kind, category=item.category,
                behavior=item.behavior, action=item.action,
                extras={"wj_split": wj_split, "hc_phase": phase},
            ))
    return samples


# ── Test-phase combo selection ───────────────────────────────────────────


def select_extreme_conditions(train_run_dir: Path, top_k: int) -> list[str]:
    """Rank train-phase lora_soup conditions by harmful rate; return the
    ``top_k`` safest + ``top_k`` most harmful condition names."""
    judgments_dir = train_run_dir / "judgments"
    records = []
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        records.extend(load_judgments_jsonl(path))
    if not records:
        raise SystemExit(
            f"no train judgments found under {judgments_dir} — run --phase train first"
        )
    rows = [r for r in harmful_rate_by_condition(records) if r.condition.startswith("lora_soup")]
    rows.sort(key=lambda r: r.rate)
    print("\n  ── TRAIN RANKING (harmful rate, ascending) ──────────────")
    for r in rows:
        print(f"    {r.condition:45s} n={r.n:4d}  rate={r.rate:.3f}  "
              f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    safest = [r.condition for r in rows[:top_k]]
    most_harmful = [r.condition for r in rows[-top_k:]]
    selected = list(dict.fromkeys(safest + most_harmful))
    print(f"\n  selected for test phase: {selected}")
    return selected


def combos_for_conditions(
    condition_names: list[str], all_combos: dict[str, LoraComboCondition],
) -> tuple[LoraComboCondition, ...]:
    missing = [c for c in condition_names if c not in all_combos]
    if missing:
        raise SystemExit(
            f"train-selected conditions not reproducible from the current grid/points: {missing}"
        )
    return tuple(all_combos[c] for c in condition_names)


# ── Aggregation ──────────────────────────────────────────────────────────


def aggregate_and_plot(cfg: JailbreakEvalConfig, judgment_paths: dict[str, Path], phase: str) -> None:
    all_records = []
    for path in judgment_paths.values():
        all_records.extend(load_judgments_jsonl(path))
    print(f"\n  aggregating {len(all_records)} judgment records...")
    out_dir = cfg.run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    harm_rows = harmful_rate_by_condition(all_records)
    refusal_rows = refusal_rate_on_benign(all_records)
    explicit_refusal_rows = explicit_refusal_rate_on_benign(all_records)
    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rows, out_dir / "explicit_refusal_rate_on_benign.csv")
    plot_condition_bars(
        harm_rows, refusal_rows,
        title=f"Persona hill-climb ({phase}) — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )
    print(f"\n  ── HARMFUL RATE ({phase}) ───────────────────────────────")
    for r in sorted(harm_rows, key=lambda r: r.rate):
        print(f"    {r.condition:45s} n={r.n:4d}  rate={r.rate:.3f}  "
              f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print(f"\n  ── BENIGN NONCOMPLIANCE ({phase}) ───────────────────────")
        for r in sorted(refusal_rows, key=lambda r: r.rate):
            print(f"    {r.condition:45s} n={r.n:4d}  rate={r.rate:.3f}  "
                  f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n  artefacts: {out_dir}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--run-slug", default="hc_grid_v1")
    parser.add_argument("--base-model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-slug", default="gemma-3-27b-it")
    parser.add_argument(
        "--points-json", type=Path, default=None,
        help="JSON file with a list of trait-coefficient dicts (e.g. "
             '[{"a": 1.5, "c": 0.75}, ...]) overriding the default 20-point grid. '
             "Lets a composer / Bayesian-optimisation loop reuse this runner.",
    )
    parser.add_argument("--top-k", type=int, default=3,
                        help="test phase: k safest + k most harmful train combos to re-evaluate")
    parser.add_argument("--n-harmful-train", type=int, default=40)
    parser.add_argument("--n-benign-train", type=int, default=20)
    parser.add_argument("--n-harmful-test", type=int, default=40)
    parser.add_argument("--n-benign-test", type=int, default=20)
    parser.add_argument("--no-vanilla", action="store_true",
                        help="skip the vanilla baseline condition")
    parser.add_argument("--no-upload-hf", action="store_true")
    parser.add_argument("--no-hydrate-hf", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    registry = GEMMA_OCEAN_REGISTRIES.get(args.model_slug)
    if registry is None:
        raise SystemExit(
            f"no OCEAN registry for model_slug {args.model_slug!r}; "
            f"known: {sorted(GEMMA_OCEAN_REGISTRIES)}"
        )

    if args.points_json is not None:
        points: list[Point] = json.loads(args.points_json.read_text())
    else:
        points = default_grid_points()
    all_combos = {c.name: c for c in (point_to_combo(p, registry) for p in points)}

    cfg = build_config(args, args.phase)

    if args.phase == "train":
        combos = tuple(all_combos.values())
    else:
        train_cfg = build_config(args, "train")
        if cfg.hydrate_hf:
            hydrate_run_dir_from_hf(
                local_run_dir=train_cfg.run_dir, eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug, run_slug=train_cfg.run_slug,
                repo_id=cfg.hf_repo_id,
            )
        selected = select_extreme_conditions(train_cfg.run_dir, args.top_k)
        combos = combos_for_conditions(selected, all_combos)

    conditions = tuple(c.name for c in combos)
    if not args.no_vanilla:
        conditions = ("vanilla",) + conditions
    cfg.conditions = conditions
    cfg.lora_combos = combos

    print("=" * 70)
    print(f"  Persona hill-climbing — phase={args.phase!r} run_slug={cfg.run_slug!r}")
    print(f"  Base model: {cfg.base_model} (slug={cfg.model_slug})")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions ({len(cfg.conditions)}):")
    for c in cfg.conditions:
        print(f"    {c}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    if cfg.hydrate_hf:
        hydrate_run_dir_from_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id,
        )

    samples = build_samples(cfg, args, args.phase)
    response_paths = run_all_conditions_inference(
        cfg, samples, output_dir=cfg.run_dir / "responses",
    )
    if cfg.upload_hf:
        upload_run_dir_to_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id, stage="inference",
        )

    judgment_paths = run_judges_on_all_conditions(
        cfg, response_paths, output_dir=cfg.run_dir / "judgments",
    )
    if cfg.upload_hf:
        upload_run_dir_to_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id, stage="judgments",
        )

    if not args.skip_aggregate:
        aggregate_and_plot(cfg, judgment_paths, args.phase)
        if cfg.upload_hf:
            upload_run_dir_to_hf(
                local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug, run_slug=cfg.run_slug,
                repo_id=cfg.hf_repo_id, stage="aggregate",
            )


if __name__ == "__main__":
    main()
