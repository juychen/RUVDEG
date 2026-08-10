# RUVAEDEG 项目总结

> 单细胞 RNA-seq 差异表达（DEG）分析中，用变分自编码器（VAE）显式建模并去除**非期望变异（unwanted variation, UV）**的研究项目。
> 中文详细笔记见 `RUV-PRPS-VAE-Notes.md`，成果展示见 `RUVVAE_slides.md`（含 pptx）。

---

## 1. 要解决的问题

### 1.1 核心困难：组间差异 ≠ 生物学差异

单细胞 DEG 的最大障碍是批次/技术变异与生物学分组混淆。本项目数据集（小鼠丘脑 TH 谷氨酸能神经元，CSRES 慢性社会挫败应激 vs CON）的混杂结构非常极端：

| 维度 | 取值 | 与 status 的关系 |
|---|---|---|
| `status` | CON (2681) / CSRES (2140) | — |
| `donor` | 7 个 | **与 status 完全共线/嵌套**（CSRS* 全为 CSRES，MW* 全为 CON） |
| `company` | beirui / seekgene / yunzhun | 与 status 部分共线 |
| `date` | 4 个批次 | 20250728 = 全部 CSRES |
| `n_genes_on` | 3548 ± 1202 | 测序深度差异 |

> ⚠️ donor 与 status 完全共线 ⇒ 任何「按组求均值相减」都无法区分应激效应与送样公司/建库批次效应。**UV 的量级是生物信号的 2–3 倍**（mean |logFC_uv| ≈ 0.08 vs mean |logFC_group| ≈ 0.026），不做校正的 DEG 基本是在测「哪家公司测的序」。

### 1.2 经典 RUV 的困境

RUV 分解 `Y = Xβ + Wα + ε` 中，未知变异因子 `W` 只能靠 SVD/PCA **线性估计**，且需要负对照基因（NC）锚定——NC 信息只在训练时用一次，无法非线性外推。

---

## 2. 主要模型

### 2.1 RUVVAE（项目主方法，`model.py`）

把 RUV 的三项分解搬进 VAE，`log(y) = y_bio + Δ_lat + Δ_cov` 三通道**在 log 尺度可加**：

```
y_bio     = decoder_bio(z) + bias + c_group @ W_group   # 生物学通道（z: d_bio=64）
Δ_lat     = sample_emb[batch_idx] + decoder_w(w)        # 潜在 UV（w: k_unk=4 + per-donor embedding）
Δ_cov     = Σ c_k @ W_cov[k]                            # 已知技术协变量（batch, n_genes_on）
y_recon   = y_bio + Δ_lat + Δ_cov                       # 完整观测重建
```

- **编码器**：双分支 `encoder_z`（生物）与 `encoder_w`（UV），各为 512→256 的 MLP + reparameterization。
- **per-donor 嵌入**：`sample_emb` 形状 `(n_batch, n_genes)`，随**样本数**（~7）而非细胞数（~1e4）扩展——这是模型能 scale 的关键设计。
- **两种重建损失**：MSE（log 尺度，默认）或 **ZINB**（scVI 风格，`use_zinb=True`）。ZINB 模式复用 RUV 分解 `log_mu = y_bio + Δ_lat + Δ_cov`，`theta = softplus(px_r)`、dropout 从 `decoder_dropout(z)` 预测（logits 形式）。
- **group 是显式线性项** `c_group @ W_group`（设计变量，不属于 UV），其行差 `W_group[d] - W_group[c]` 直接就是 logFC。

### 2.2 NC 损失：整个模型的「锚」

```python
L = L_recon + KL_z + KL_w + L_NC     # 四项等权
L_NC = MSE(Δ_total[:, nc], y_nc − nc_mean)   # Δ_total = Δ_lat + Δ_cov
```

NC 基因被假定不受 CSRES 影响，其全部跨细胞变异都应是 UV。该可微损失强迫 UV 通道**在训练全程**被 NC 拉住：
1. UV 通道被校准到真实 UV 幅度（KL_w 训练后坍缩到 0.0084）；
2. `y_bio[nc] ≈ nc_mean`——NC 基因在生物通道被「压平」；
3. 由于 UV 通道是全基因共享的，NC 上学到的 UV 模式**外推到全部基因**（RUV 核心假设）。

**NC 基因选择**（`NC_MODE="data_only"`）：按 |logFC| 升序取 500 个变化最小的基因，其中仅 4 个命中文献 HKG 池 ⇒ 37 个文献管家基因是真正的 **held-out 验证**。

### 2.3 scVI 基线扩展（`model_scvi_disease.py` / `model_scvi_ruv.py`）

- **`SCVIWithDiseaseEffect`**：继承 scVI，把重建分解为「scVI 部分 × 疾病组线性效应」，`W_group` 在 log-rate 上（乘性 `exp(W_group)`），`get_disease_logfc_by_group` 输出 status vs 对照的 logFC。
- **`SCVIRUVWithDisease`**（RUV-scVI）：在上一基础上加入 **`z_bio`（生物 latent）+ `z_uv`（UV latent）+ company embedding**——公司信息应被 UV 分支吸收而 `z_bio` 保留疾病差异（检验设计见 `test_scvi_ruv.ipynb`）。
- `scVI.py` / `scviHarmony.py`：vanilla scVI 与 scVI+Harmony 基线脚本（`run_scvi_all.sh` 系列）。

### 2.4 早期/其他模型（笔记中的设计）

- `RUVVAE` / `HybridRUVVAE`（笔记 Part III/IV）：RUV-VAE 初步方案；Hybrid 为**显式 W 协变量矩阵 + 隐内参向量**的双通道架构。
- `RUVVAE_log2`：log2 尺度的早期版本。
- **RUV-III + PRPS**（笔记 Part I/II）：Nature Biotech 2023 论文方案——伪样本的伪重复，作为理论来源与未来对比基线。
- 玩具数据集（`toy.ipynb`）：2 基因 × 4 condition × 4 batch 的最小可运行实例，用于调试模型/scVI 前向逻辑。

---

## 3. 核心想法（一句话版）

1. **RUV × VAE**：用 VAE 隐变量替换 RUV 的 SVD 线性因子，把 NC 约束变成**可微损失项**，让模型在整个训练中持续被 NC 校准（而非只用一次）。
2. **三通道 log 可加**：生物/潜在 UV/已知协变量在 log 尺度分离 ⇒ `y_bio` 直接当「去 UV 的干净表达」做组间比较，无需后处理。
3. **per-sample 嵌入**：donor 级 UV 参数化，复杂度随样本数而非细胞数扩展。
4. **ZINB 复用 RUV 分解**：scVI 式计数建模与 RUV 分解共用同一套 decoder/矩阵，一个模型两种重建。
5. **数据驱动 NC**：|logFC| 升序选 NC 规避「用文献 HKG 自证」的循环论证；held-out HKG 验证。
6. **RUV-VAE 双用途**：既可直接做 DEG 主方法，也可作 MAST 的 logFC 后处理器（笔记 Part V）。

---

## 4. 主要结果（`testRUVVAE.ipynb` / slides）

- **训练**：Total/Recon 损失单调降至 0.0907/0.0856；`KL_w` 0.51 → 0.0084（UV 潜变量被 NC 压紧）。
- **方差分配**：`y_recon` r²=0.706 / pearson=0.810，`y_bio` 仅 r²=0.307——**~40% 观测方差是技术性的**，恰是 UV 的体量。
- **DEG 校正前后**：同一 40 个基因的 dotplot 从「7 个 donor 千篇一律」变为「CSRS vs MW 清晰双块」——模型没有抹掉信号，而是把 donor 效应归入 UV。
- **Held-out 验证**：37 个文献 HKG 校正前呈明显 donor 梯度（MW22B 低、MW47A 高），校正后表达压平、donor 差异消失；Δ_cov 与 Δ_lat 按 donor 分层的红蓝条带方向互补，共同还原批次偏移。
- **DEG 检验**：FDR<0.05 & |logFC|>0.1 ⇒ 校正前 Up 2640 / Down 636，校正后大幅减少（up/down 不对称是 rRNA/ambient 与应激响应的特征）。

---

## 5. 当前局限与待解决问题（下一步）

1. **伪重复（pseudo-replication）**：当前检验以**单细胞为独立样本**（n≈4821），真实独立单位是 7 个 donor，DEG 数量被严重高估——**必须切换到 donor-level pseudobulk**（如 DESeq2-pseudobulk）才能得到可发表的结论。
2. **与基线系统对比**：MAST / DESeq2-pseudobulk / RUV-III-PRPS / scVI / scVI+Harmony 的定量对比（DEG 重叠、UV 校正幅度、FDR 校准）。
3. **ZINB 模式验证**（`testRUVVAE_ZINB.ipynb`）：计数尺度重建是否优于 log1p+MSE。
4. **超参数搜索**：`k_unk ∈ {2,4,8,16}`、`d_bio ∈ {32,64,128}`，看 DEG 稳定性与 UV 吸收效果。
5. **下游验证**：对校正后 DEG 做 GO/KEGG 富集，确认生物学合理性；RNA 探针（rRNA/ambient）排除。
6. NC 敏感性：NC 数目/选择模式（HKG vs data-driven）对结果的影响。

---

## 6. 文件地图

| 文件 | 内容 |
|---|---|
| `RUV-PRPS-VAE-Notes.md` | 理论笔记：PRPS 论文解读、伪重复构造、RUV-VAE/Hybrid 方案、MAST 后处理 |
| `RUVVAE_slides.md` / `.pptx` | 成果幻灯片（问题→模型→结果→局限） |
| `model.py` | 主模型 `RUVVAE_DEG`（MSE/ZINB 双模式）+ `compute_deg_all_methods_legacy`/`compute_deg` 等 DEG 计算 |
| `model_scvi_disease.py` | `SCVIWithDiseaseEffect`（scVI + 疾病线性效应） |
| `model_scvi_ruv.py` | `SCVIRUVWithDisease`（+ z_uv + company embedding） |
| `scVI.py` / `scviHarmony.py` | vanilla scVI 与 scVI+Harmony 基线 |
| `lrtest.R` | MAST 风格零膨胀 LRT（二项表达 + t 统计） |
| `testRUVVAE.ipynb` | 主分析 notebook（结果图在 `ppt_figs/`、`slide_figs/`） |
| `testRUVVAE_ZINB.ipynb` | ZINB 模式 + company embedding 提取 |
| `test_scvi_ruv.ipynb` | RUV-scVI 模型测试 |
| `toy.ipynb` | 2 基因玩具数据，调试用 |
| `60_SummerizeHigenes.ipynb` | 高表达基因汇总、旧/新数据 DEG 分析 |
| `darw_gene.py` / `draw_neggene.py` / `build_pptx.py` | 绘图/幻灯片脚本 |
| `run_scvi_*.sh` | scVI 各配置训练脚本 |
| `scvi-tools/` | 本地 scvi-tools 源码（含修改） |
| `scviRUV_output.model/` | 训练好的模型输出（含 disease_effect.csv） |
