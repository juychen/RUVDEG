# RUVDEG

Single-cell RNA-seq differential expression (DEG) analysis with a **variational autoencoder that explicitly models and removes unwanted variation (UV)**.

> Companion code for a research project on a mouse thalamus Glu neuron dataset (chronic social defeat stress vs control), where donor / sequencing-batch effects are deeply confounded with the biological condition of interest.

---

## What this repo contains

| Path | Description |
|---|---|
| `model.py` | Main model `RUVVAE_DEG` — three-channel log-additive decomposition (`y_bio + Δ_lat + Δ_cov`) with MSE or ZINB reconstruction |
| `model_scvi_disease.py` | `SCVIWithDiseaseEffect` — scVI baseline with an explicit group-level log-rate effect |
| `model_scvi_ruv.py` | `SCVIRUVWithDisease` — scVI + separate biological / UV latents + company embedding |
| `scVI.py`, `scviHarmony.py` | Vanilla scVI and scVI+Harmony baselines |
| `lrtest.R` | MAST-style zero-inflated LRT utility |
| `testRUVVAE.ipynb` | Main analysis notebook (results in `ppt_figs/`, `slide_figs/`) |
| `testRUVVAE_ZINB.ipynb` | ZINB reconstruction variant + company embedding extraction |
| `test_scvi_ruv.ipynb` | RUV–scVI model sanity checks |
| `toy.ipynb` | 2-gene × 4-condition toy dataset for debugging forward / loss logic |
| `60_SummerizeHigenes*.ipynb` | High-expression gene summary, old / new data DEG |
| `darw_gene.py`, `draw_neggene.py`, `build_pptx.py`, `prep_figs.py` | Plotting and slide-building scripts |
| `run_scvi_*.sh` | scVI training driver scripts (all / nobatch / nocov / harmony variants) |
| `RUV-PRPS-VAE-Notes.md` | Theory notes: PRPS paper, pseudo-replication, RUV-VAE / Hybrid designs |
| `RUVVAE_slides.md` / `.pptx` | Project slides (problem → model → results → limits) |
| `summary.md` | Detailed project summary (Chinese) |

The `scvi-tools/` directory is a local checkout of `scvi-tools` used to wire up the scVI baselines; `scviRUV_output.model/`, `ppt_figs/`, `slide_figs/`, `__pycache__/` and similar generated folders are listed in `.gitignore`.

---

## The problem

Standard scRNA-seq DEG methods assume that biological group (CON vs CSRES in this dataset) and donor / sequencing batch are distinguishable. In this dataset they are not:

- `donor` is **fully confounded** with `status` (every `CSRS*` donor is CSRES, every `MW*` donor is CON).
- `company` (`beirui / seekgene / yunzhun`) is partially confounded with status.
- `date` of sequencing is partially confounded.
- UV magnitude is **2–3× the biological signal** (`mean |logFC_uv| ≈ 0.08` vs `mean |logFC_group| ≈ 0.026`).

Naïve per-gene tests pick up donor / company effects, not biology.

---

## The idea

1. **RUV × VAE** — replace the SVD-based linear UV factors of RUV with VAE latents, and turn the negative-control (NC) constraint into a differentiable loss term that anchors the UV branch throughout training instead of only once.
2. **Three log-additive channels** — biology, latent UV, and known covariates separate cleanly on the log scale: `y_recon = y_bio + Δ_lat + Δ_cov`, so `y_bio` is directly usable for group comparisons with no post-processing.
3. **Per-sample embeddings** — donor-level UV parameterisation; complexity scales with sample count (~7) rather than cell count (~5k).
4. **ZINB-compatible** — the same RUV decomposition is reused as the mean of a ZINB likelihood (scVI style), so the model can run either on log-scale MSE or on raw counts.
5. **Data-driven NC + held-out HKG validation** — choose negative controls by smallest |logFC| rather than from a literature list to avoid circular validation; the 37 literature housekeeping genes are reserved as held-out validation.

---

## Quick start

```bash
# 1. Create the environment
conda create -n scvi-env python=3.11 -y
conda activate scvi-env
pip install scvi-tools scanpy anndata pytorch lightning harmonypy numpy pandas matplotlib seaborn tqdm jupyter

# 2. Train the main RUVVAE model and reproduce the headline numbers
jupyter nbconvert --to notebook --execute testRUVVAE.ipynb
```

For baseline comparisons:

```bash
bash run_scvi_all.sh           # vanilla scVI
bash run_scvi_harmony.sh       # scVI + Harmony (note: --harmony-batch defaults to `company`)
bash run_scvi_nobatch.sh       # scVI with batch key disabled
bash run_scvi_nocov.sh         # scVI with no covariates
```

---

## Repository layout (annotated)

```
.
├── model.py                       # RUVVAE_DEG — main method (MSE / ZINB)
├── model_scvi_disease.py        # SCVIWithDiseaseEffect
├── model_scvi_ruv.py            # SCVIRUVWithDisease
├── scVI.py / scviHarmony.py     # scVI baselines
├── lrtest.R                     # MAST-style LRT (R)
│
├── testRUVVAE.ipynb             # main results notebook
├── testRUVVAE_ZINB.ipynb        # ZINB + company embedding
├── test_scvi_ruv.ipynb          # RUV–scVI sanity checks
├── toy.ipynb                    # 2-gene toy dataset
├── 60_SummerizeHigenes*.ipynb   # gene-level summary
│
├── darw_gene.py                 # plotting
├── draw_neggene.py              # plot helpers
├── build_pptx.py                # slide builder
├── prep_figs.py                 # figure prep
│
├── run_scvi_*.sh                # training drivers
│
├── RUV-PRPS-VAE-Notes.md        # theory + design notes (Chinese)
├── RUVVAE_slides.md / .pptx     # slides
├── summary.md                   # full project summary (Chinese)
│
├── ppt_figs/   slide_figs/      # generated plots
├── scviRUV_output.model/        # trained model weights
└── scvi-tools/                  # local scvi-tools checkout (gitignored)
```

---

## Reproducing the headline results

| Result | Notebook / script |
|---|---|
| Variance decomposition (`y_recon` r² ≈ 0.71, `y_bio` r² ≈ 0.31) | `testRUVVAE.ipynb` |
| Held-out HKG flatten after UV removal | `testRUVVAE.ipynb` |
| DEG counts before / after correction | `testRUVVAE.ipynb` |
| ZINB reconstruction comparison | `testRUVVAE_ZINB.ipynb` |
| Company embedding (`scviRUV_output.company_embedding.csv`) | `testRUVVAE_ZINB.ipynb` |
| Disease effects (`scviRUV_output.disease_effect.csv`) | `testRUVVAE.ipynb` |
| scVI / scVI+Harmony baselines | `run_scvi_all.sh`, `run_scvi_harmony.sh` |

---

## Known limitations

1. **Pseudo-replication** — current inference treats each cell as an independent sample, but the true replication unit is donor (n=7). Donor-level pseudobulk analyses (DESeq2-pseudobulk, etc.) are required for publishable conclusions.
3. **ZINB validation** — ZINB-mode reconstruction has not yet been quantitatively compared against the MSE variant on final DEG sets.
4. **Hyperparameter search** — `k_unk ∈ {2,4,8,16}` and `d_bio ∈ {32,64,128}` sweeps are pending.
6. **NC sensitivity** — effect of NC count / selection mode (literature HKG vs data-driven) not yet characterised.

See `summary.md` §5 for the full backlog.

---

## License & citation

Internal research code — no public license granted at this time. A citation entry will be added on first public release.

---

## Contact

Project: `RUVDEG`  
Repo: https://github.com/juychen/RUVDEG