# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read the paper first

**At the start of every new session, before making non-trivial code changes, read the source paper.** Order of preference:

1. Local copy: `paper/iclr2026_conference.tex` (full LaTeX source, single file). Use the `Read` tool. `paper/figures/` and the original arXiv tarball `paper/source.tar.gz` are also there.
2. Fallback: `https://arxiv.org/html/2510.24256v2` via `WebFetch`. This is summarized through a smaller model, so prefer the local copy.

Blog (lighter intro): `https://www.goodfire.ai/research/understanding-memorization-via-loss-curvature`.

## Paper context (load-bearing facts)

**Core claim.** Per-example, memorized sequences sit in sharp loss directions, but those sharp directions point all over the place across examples and cancel under averaging. At the **population** level, memorization lives in **low-curvature** directions of the Fisher; shared/generalizing computation lives in **moderate-curvature** directions. The edit removes the low-curvature mass — the opposite of "keep the small singular values."

**What the code computes, mapped to the paper.** For each MLP linear `W`, K-FAC approximates the layer Fisher as `F_W ≈ G ⊗ A` with `A = E[aᵀa]` (input cov) and `G = E[gᵀg]` (pre-activation gradient cov). Eigendecompose both; form coefficients `C = U_Gᵀ W U_A`; rank pairs `(i,j)` by joint mass `λ_i μ_j`; keep the top pairs until cumulative mass ≥ ρ; zero the rest in `C` and rebuild `W = U_G C_masked U_Aᵀ`. This is `KFACTreatmentPairwise.apply_kfac_by_product` and is **the paper's method**. `KFACTreatment.apply_kfac` (per-side variance ratio) is a baseline/ablation, not the main result.

**Headline numbers (OLMo-2 7B), useful as sanity bounds.**
- In-distribution memorized strings: strict accuracy 99.9% → **3.4%**.
- OOD famous quotes: K-FAC **16.1%** strict vs BSN 60% (BSN overfits its supervised forget-set).
- Perplexity 19.04 → **22.84** (vs BSN 23.59). A few-point rise is the published behavior, not a regression.
- Logical reasoning 95–106% retained; open-book QA 93–99%.
- Arithmetic drops to 66–74% — the model writes correct reasoning steps then miscalculates. Closed-book recall is similarly hit.
- ViT with noisy labels: memorization 81.6% → 3.5%; validation accuracy 67% → **71.7%** (improves).

**Practical recipe.**
- **LM**: edit *middle-late* MLPs together — paper headlines layers **23–25** with **gate + up** projections at **ρ ≈ 0.6**. The repo's eval lets you also touch `down`; the paper's main LM result does not.
- **ViT**: layers 0 and 11, up + down, ρ ≈ 0.75.
- Multi-layer edits beat single-layer.
- Layer/ρ selection still needs a memorization validation signal — the unsupervised step is *what to remove inside a chosen layer*, not *which layer*.

## Commands

Use `uv run` (not `python`). Two commands cover the full reproduction path:

**Collect K-FAC factors** (writes `kfac_factors_blk_<...>.pt` per pass):
```bash
uv run data/collect_kfac_multilayer.py \
  --model allenai/OLMo-2-1124-7B \
  --target_blocks 19 23 27 31 \
  --layers_per_pass 3 \
  --batch_size 48 --nbytes 100000000 --sample_labels \
  --save_dir <out_dir>
```

**Apply edit + evaluate**:
```bash
uv run evaluations/eval_mem_kfac.py \
  --model-size 7b \
  --layers-json '{"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}' \
  --use-cache
```

`--layers-json` keys are layer indices, values are keep-mass ρ ∈ [0,1] per MLP projection. ρ ≥ 0.9999 skips that projection. Use `--layers-file` for larger configs and `--order` to control application sequence.

## Architecture

Three-stage pipeline. Each stage lives in its own top-level package and they communicate through files on disk.

**1. Factor collection (`data/collect_kfac_multilayer.py`)**: Streams text from HF dolmino/olmo-mix shards through OLMo-2, accumulates per-MLP-projection K-FAC factors via forward + full-backward hooks. `A = E[aᵀa]` from pre-activation inputs; `G = E[gᵀg]` from gradients of pre-activations. Processes `layers_per_pass` blocks at a time with gradient checkpointing — only those blocks have `requires_grad=True`. Drops the last position from both `a` and `g` to keep them aligned (the last token has no next-token gradient). Saves `{"blkN.gate": {"A":..., "G":..., "n_tokens":N}, ...}` per pass.

**2. Edit (`kfac_treatment_pairwise.py`)**: Two compression strategies on a single class hierarchy.
- `KFACTreatment.apply_kfac((var_A, var_G))` — variance-retained per side: independently picks the smallest `rA, rG` whose cumulative eigenvalue mass exceeds the target, then `W ← U_G U_Gᵀ W U_A U_Aᵀ` (rank `≤ rG × rA`).
- `KFACTreatmentPairwise.apply_kfac_by_product(ρ)` — joint product ranking: selects `(i,j)` pairs by largest `λ_i μ_j` until cumulative mass ≥ ρ of `(Σλ)(Σμ)`. Uses a k-way merge heap to avoid materializing the full `m×n` outer product. The eval script uses **only** this pairwise variant.

Both share the same eigendecomposition (`torch.linalg.eigh` on float A, G, sorted descending) and original-weight bookkeeping. Original weights are cloned on init so `restore_original_weights()` is always available.

**Layer-name → factor-key convention**: `model.layers.31.mlp.up_proj` ↔ `blk31.up` (similarly `gate`, `down`). `KFACTreatment._get_kfac_key` does the conversion. Factor files are stored *per group* of blocks; `evaluations/eval_mem_kfac.py:KFAC_FACTORS_{1B,7B}` maps layer indices to the bundle file that contains them.

Dimension contract per layer: `evc_G.shape[0] == out_features`, `evc_A.shape[0] == in_features` — note `down_proj` is `[4096, 11008]` (out, in) so G corresponds to 4096, A to 11008. There is an explicit assert.

**3. Evaluation (`evaluations/eval_mem_kfac.py` + `metrics/`)**: Loads model, runs baseline `MemorizationEvaluator.run_all_evals` (memorization Levenshtein, quotes Levenshtein, nDCG@10), applies K-FAC layer-by-layer, then re-runs evals. Optionally computes BSN-style perplexity if `--perplexity` is set, but this requires the pre-tokenized cache at `evaluations/bsn_dependencies/data/olmo2_clean_pt_cache_112.pt` which is not in the repo. Edited weights are cached at `cache/kfac_weights/<model>__L<idx>__<proj>__rho<ρ>__<dtype>.pt` keyed by ρ; `--use-cache` reuses, `--refresh-cache` overwrites.

`MemorizationEvaluator` (`metrics/memorization_evaluator.py`) is an orchestrator — it does not reimplement evals, it dispatches to `compute_memorization_metrics_fixed_ids` (token-level), `compute_memorization_metrics_levenshtein` (text-level for quotes), and `NDCGEvaluator`. `MODEL_CONFIGS` there is the source of truth for default datasets and `num_layers` per size.

## Paths and configuration

`data/paths.py` centralizes all data file locations. Relevant environment overrides:
- `MEM_EVAL_ROOT` — repo root (default: parent of `data/`)
- `MEM_EVAL_DATA_ROOT` — JSONL datasets directory
- `MEM_EVAL_EDITED_ROOT` — where edited model state dicts are saved
- `MEM_KFAC_FACTORS_ROOT` — K-FAC factor `.pt` files (default: `<repo>/assets/kfac_factors`, **not committed**)
- `MEM_KFAC_CACHE_DIR` — cached edited weights (default: `<repo>/cache/kfac_weights`)
- `MEM_EVAL_PT_CACHE` — pre-tokenized perplexity cache

The K-FAC factor files referenced in `KFAC_FACTORS_1B` / `KFAC_FACTORS_7B` and `pile10k_None.txt` are not in the repo — they must be generated (factors via `collect_kfac_multilayer.py`) or supplied separately.

## Conventions

- File removal: never run `rm` directly — propose to the user and let them execute (per global policy).
- Always `uv run` for Python execution and `uv pip` / `uv add` for package management.
