# RUVVAE-DEG
## 用变分自编码器做「显式去除未知变异」的单细胞差异表达分析

**数据集**：小鼠丘脑 (TH) `TH Tll1_Thsd7b Glut` 谷氨酸能神经元
**对比**：CSRES (慢性社会挫败应激) vs CON

---

## Slide 1 — 问题与思路

### 1.1 我们要解决的问题

单细胞 DEG 的核心困难：**组间差异 ≠ 生物学差异**。

本数据集的混杂结构非常典型：

| 维度 | 取值 | 与 status 的关系 |
|---|---|---|
| `status` | CON (2681) / CSRES (2140) | — |
| `donor` | 7 个 (CSRS1-3, CSRS9-1, CSRS10-3 / MW22B, MW45A, MW47A, MW51A) | **完全嵌套**：CSRS* 全是 CSRES，MW* 全是 CON |
| `company` | beirui / seekgene / yunzhun | 与 status 部分共线 |
| `date` | 4 个批次 | 20250728 = 全部 CSRES |
| `n_genes_on` | 3548 ± 1202 基因/细胞 | 测序深度差异 |

> ⚠️ **donor 与 status 完全共线**。任何简单的「按组求均值相减」都无法区分
> 「应激效应」和「送样公司/建库批次效应」。这正是 RUV 要解决的问题。

### 1.2 思路：把 RUV 的三项分解搬进 VAE

经典 RUV 模型：
```
Y  =  Xβ  +  Wα  +  ε
     ↑生物   ↑未知变异
```

RUV 的困境：`W`（未知变异因子）只能靠 SVD/PCA 线性估计，且需要**负对照基因 (NC)**
来锚定——但 NC 的信息只在训练时用一次，无法非线性外推。

**RUVVAE 的做法**：把 `W` 换成 VAE 的隐变量，把 NC 约束变成一个**可微的损失项**，
让模型在整个训练过程中持续被 NC 拉住。

```
log(y)  =  y_bio           +  Δ_lat            +  Δ_cov
           ↑ 生物学通道        ↑ 潜在未知变异      ↑ 已知协变量变异
           z (d_bio=64)      w (k_unk=4)        batch / n_genes_on
                             + sample_emb
```

三条通道**在 log 尺度上可加**，因此可以直接把 `y_bio` 拿出来当作
「去除了 UV 的干净表达」，也可以直接对 `y_bio` 做组间比较得到干净的 logFC。

### 1.3 关键设计选择

| 选择 | 理由 |
|---|---|
| **双编码器** (`encoder_z` / `encoder_w`) | z 与 w 参数完全分离，无梯度耦合 |
| **`W_group` 显式参数** `(2, 16428)` | 组效应是**线性可读**的，不藏在黑箱里 |
| **`sample_emb`** `(3, 16428)` | 每个送样公司一条可学习的 UV 基线 |
| **NC 损失** | 唯一强制 z/w 分离的结构性约束（无对抗训练） |
| **`n_genes_on` 标准化后作线性协变量** | 测序深度是**已知**的 UV，不该浪费 w 的容量 |

---

## Slide 2 — 模型架构

### 2.1 计算图

```
                    ┌──────────────────────────────────────────┐
                    │        y  (log1p, N×16428)               │
                    └────────┬────────────────────┬────────────┘
                             │                    │
              ┌──────────────▼─────┐   ┌──────────▼──────────┐
              │  encoder_z         │   │  encoder_w          │
              │  16428→512→256     │   │  16428→512→256      │
              │  → μ_z, logσ²_z    │   │  → μ_w, logσ²_w     │
              └──────────┬─────────┘   └──────────┬──────────┘
                         │ z ~ N(μ_z,σ_z)         │ w ~ N(μ_w,σ_w)
                         │ d_bio = 64             │ k_unk = 4
              ┌──────────▼─────────┐   ┌──────────▼──────────┐
              │  decoder_bio       │   │  decoder_w          │
              │  64→256→16428      │   │  4→256→16428        │
              └──────────┬─────────┘   └──────────┬──────────┘
                         │ + bias                 │ + sample_emb[batch]
                         │ + c_group @ W_group    │
                         ▼                        ▼
                   ┌─────────┐            ┌────────────┐    ┌────────────┐
                   │  y_bio  │            │  Δ_lat     │    │  Δ_cov     │
                   │ 生物信号 │            │ 潜在 UV     │    │ 已知 UV     │
                   └────┬────┘            └─────┬──────┘    └─────┬──────┘
                        └───────────────────────┴─────────────────┘
                                          │  (相加)
                                     ┌────▼─────┐
                                     │ y_recon  │
                                     └──────────┘
```

### 2.2 精确数学形式

```
y_bio_base  =  decoder_bio(z)  +  bias                  # (N, G)
group_effect =  c_group @ W_group                        # (N,2)@(2,G) → (N,G)
y_bio        =  y_bio_base  +  group_effect              # 干净生物通道

Δ_lat        =  sample_emb[batch_idx]  +  decoder_w(w)   # 潜在 UV
Δ_cov        =  Σ_k  c_k @ W_cov[k]                      # batch(3,G) + n_genes_on(1,G)

y_recon      =  y_bio  +  Δ_lat  +  Δ_cov                # 完整观测重建
```

### 2.3 损失函数 —— NC 损失是全模型的「锚」

```python
L = L_recon + KL_z + KL_w + L_NC          # 四项等权相加
```

| 项 | 公式 | 作用 |
|---|---|---|
| `L_recon` | `MSE(y_recon, y)` | 重建保真度（模型也支持 ZINB 模式，本次用 MSE） |
| `KL_z` | `−½·E[1+logσ²−μ²−σ²]` | z 正则到 N(0,I) |
| `KL_w` | 同上 | w 正则到 N(0,I)，**限制 UV 通道容量** |
| `L_NC` | `MSE(Δ_total[:,nc], y[:,nc] − nc_mean)` | **RUV 约束** |

**NC 损失的机制**（这是整个模型能 work 的核心）：

```python
y_nc     = y[:, neg_control_mask]          # NC 基因的观测
nc_mean  = y_nc.mean(dim=0, keepdim=True)  # NC 的跨细胞全局基线
L_NC     = MSE(Δ_lat[:,nc] + Δ_cov[:,nc],  y_nc − nc_mean)
```

对于 NC 基因，我们**假定它不受 CSRES 影响**，所以它的全部跨细胞变异
都应该是 UV。这个损失强迫 `Δ_total` 去精确吃掉 NC 基因的残差，
于是：

1. UV 通道被**校准**到真实 UV 的幅度上（不会过度或不足）；
2. `y_bio[nc] ≈ nc_mean`（NC 基因在生物通道上被「压平」）；
3. 由于 UV 通道是**全基因共享**的（`decoder_w` / `sample_emb` 输出 16428 维），
   在 NC 上学到的 UV 模式会**外推到全部基因**——这就是 RUV 的核心假设。

**NC 基因选择**：`NC_MODE = "data_only"`，从 16428 个基因中按 |logFC| 升序取
**500 个**变化最小的基因（logFC 范围 0.0000 ~ 0.0007），其中 4 个命中文献 HKG 池。

### 2.4 训练配置

| 项 | 值 |
|---|---|
| 优化器 | AdamW, lr=1e-3, weight_decay=1e-5 |
| 调度器 | ReduceLROnPlateau (patience=10, factor=0.5) |
| Epochs / batch | 150 / 256 |
| `d_bio` / `k_unk` | 64 / 4 |
| 设备 | CUDA |

![训练损失曲线](ppt_figs/fig1_training_loss.png)

**读图**：Total / Recon 单调下降至 0.0907 / 0.0856；`KL_w` 从 0.51 →
**0.0084**，说明 UV 潜变量被 NC 损失约束得非常紧、几乎坍缩到先验——
UV 主要由 `sample_emb` 和 `W_cov` 这些**显式**通道承担，而不是黑箱潜变量。
`NC` 损失稳定在 0.0045，说明 NC 约束从早期就已满足。

---

## Slide 3 — 模型是否真的把 UV 分离出来了？

![六联诊断图](ppt_figs/fig2_results_6panel.png)

### 3.1 六个诊断面板

| 面板 | 结论 |
|---|---|
| **Volcano** | FDR<0.05 且 \|logFC\|>0.1：**Up 2640 / Down 636** |
| **Biology vs UV** | 散点呈**弥散云状、无明显对角线** → 生物信号与 UV 未被系统混淆 |
| **UV 分解** | `logFC_uv_cov` 与 `logFC_uv_latent` 呈**负相关带状** → 两条 UV 通道在互补分担同一批效应 |
| **Bio latent PCA (z)** | 两组细胞**大幅重叠**但重心可分 → z 编码的是细胞状态连续谱，不是硬分组 |
| **UV latent PCA (w)** | beirui 与 yunzhun 沿 PC1 明显分开；**但 seekgene 点数少且与 beirui 大量重叠** → 分离是**部分的** |
| **Reconstruction** | 点紧贴 y=x 对角线 → 重建良好 |

> ⚠️ **注意不要过度解读 UV latent PCA**：`w` 沿 PC1 确实把 beirui 与 yunzhun 拉开了，
> 但 seekgene（仅 MW47A，633 细胞）几乎完全埋在 beirui 云团里。
> **分离是部分的，不是三簇干净分开。**
> 本工作最强的证据是**负对照基因找平 (0.244×)** 与**校正前后的 dotplot 对比**，
> 而不是这张 PCA。

### 3.2 定量：NC 基因找平诊断

| 指标 | 值 | 理想 | 判定 |
|---|---|---|---|
| `y_bio[nc]` vs `nc_mean` bias 均值 | **−0.0002** | ≈0 | ✅ |
| bias \|max\| | 0.0158 | 小 | ✅ |
| `Δ_total[nc]` vs `y−nc_mean` bias | 0.0002 | ≈0 | ✅ |
| `y_recon` vs `y` MAE | **0.0172** | 小 | ✅ |
| **跨 donor NC 标准差** raw → y_bio | **0.0501 → 0.0122** | <1 | ✅ **0.244×** |

> NC 基因跨 donor 的波动被压缩到原来的 **24.4%**，
> 且生物通道对 NC 的估计几乎无偏（bias 1e-4 量级）。

### 3.3 重建质量

| 指标 | `y_recon` (含UV) | `y_bio` (去UV) |
|---|---|---|
| `r2_total` | **0.7057** | 0.3067 |
| `pearson_per_cell_mean` | **0.8101** | 0.6443 |
| `pearson_per_cell_median` | 0.8155 | 0.7281 |
| `r2_per_cell_mean` | 0.6586 | 0.2259 |

`y_recon` 解释了 ~71% 的总方差；`y_bio` 只解释 ~31%——
**这个落差本身就是 UV 的体量**，说明约 40% 的观测方差是技术性的。

### 3.4 方差分配（全基因平均绝对 logFC）

```
mean |logFC_uv_cov|        0.0828   ← 已知协变量 UV（batch）  ████████
mean |logFC_uv_n_genes_on| 0.0822   ← 测序深度 UV            ████████
mean |logFC_bio_latent|    0.0573   ← 潜在生物效应            █████
mean |logFC_group|         0.0255   ← 显式组效应              ██
```

> **UV 的量级是生物信号的 2–3 倍**。如果不做校正，DEG 结果基本上
> 是在测「哪家公司测的序」。

---

## Slide 4 — 核心结果：校正前 vs 校正后

### 4.1 Top 20 上调 / Top 20 下调 DEG，按 donor 分组

**校正前**（raw log1p 表达）
![DEG 校正前](ppt_figs/fig3_deg_before_uv.png)

**校正后**（`y_bio`，UV 已移除）
![DEG 校正后](ppt_figs/fig4_deg_after_uv.png)

### 4.2 这两张图是整个工作的核心证据

| | 校正前 | 校正后 |
|---|---|---|
| **donor 间模式** | 7 个 donor 几乎**长得一模一样**，看不出分组结构 | CSRS1-3/9-1/10-3 与 MW22B/45A/47A/51A **清晰劈成两块** |
| **主导变异** | donor 的整体表达水平（MW22B 整体偏低、MW47A 整体偏高） | status (CON vs CSRES) |
| **上调基因** | 在所有 donor 中都是「一片红」 | 只在 3 个 CSRS donor 中亮起 |
| **下调基因** | 无区分度 | 只在 4 个 MW donor 中亮起 |

> 校正前，dotplot 上看到的**梯度是 donor 效应**（送样公司/建库深度）；
> 校正后，同样这 40 个基因呈现出**与 status 一致的双块结构**。
> 这说明模型没有「抹掉」信号，而是把 donor 层面的偏移从生物信号里剥离了出来。

### 4.3 DEG 统计与 Top 基因

| | |
|---|---|
| 检验方法 | **Welch's t-test**（模型也支持 wald / permutation / bayes / MAST-LRT） |
| 检验对象 | `logFC_bio = logFC_group + logFC_bio_latent`（**已去 UV**） |
| 基因总数 | 16 428 |
| p < 0.05 | 10 390 |
| FDR < 0.05 | 10 121 |

**Top 差异基因（按 |logFC_bio|）**

| 基因 | logFC_bio | logFC_group | logFC_bio_latent | 备注 |
|---|---|---|---|---|
| `Gm42418` | **+2.428** | +2.020 | +0.407 | ⚠️ 已知 rRNA/ambient 污染标志 |
| `Camk1d` | +0.888 | +0.458 | +0.430 | Ca²⁺/钙调蛋白激酶，突触可塑性 |
| `Rsrp1` | +0.842 | +0.698 | +0.143 | 剪接调控 |
| `Rbfox1` | +0.839 | +0.099 | +0.740 | 神经元剪接因子，应激相关 |
| `Atp6v0b` | **−0.815** | −0.539 | −0.277 | 突触囊泡酸化 |
| `Cacna1c` | +0.781 | +0.111 | +0.670 | L 型钙通道，**精神疾病风险基因** |
| `Lrrtm4` / `Dpp6` / `Shisa9` | +0.75 ~ +0.75 | 低 | 高 | 突触后组织 / AMPA 受体辅助亚基 |
| `Grm1` / `Rora` / `Nrxn1` / `Cntnap2` | +0.69 ~ +0.74 | 低 | 高 | 突触与神经发育 |
| `Meg3` | +0.703 | +0.189 | +0.513 | lncRNA，应激响应 |

> **一个值得注意的模式**：`Gm42418` / `Rsrp1` / `Bc1` / `Lars2` / `Ddx5` 这类
> **管家/污染类基因**的信号主要来自 `logFC_group`（显式线性项）；
> 而 `Cacna1c` / `Rbfox1` / `Lrrtm4` / `Nrxn1` 这些**真正的神经元功能基因**
> 信号主要来自 `logFC_bio_latent`（非线性潜通道）。
> 这个分工暗示 `W_group` 吸收了一部分残留的全局偏移，而 `z` 编码了细胞层面的真实生物态。

---

## Slide 5 — 管家基因验证、局限与下一步

### 5.1 HKG 验证：模型是否「误伤」了不该变的基因？

**校正前** —— MW22B 整体偏低、MW47A/MW51A 整体偏高，**管家基因呈现明显 donor 梯度**
![HKG 校正前](ppt_figs/fig5_hkg_before_uv.png)

**校正后** —— 表达水平大幅压平，donor 间差异基本消失
![HKG 校正后](ppt_figs/fig6_hkg_after_uv.png)

> 37 个文献管家基因（`Oaz1`, `Rps13`, `Rps20`, `Rpl27`, `Aars`, `Polr2f`,
> `Psmd6/7`, `Psma5`, `Ipo8`, `Pop4`, `Pes1`, `Rer1`, `Rpl13a`, `Cyc1`,
> `Sdha`, `Ubc`, `Sars`, `Ppia`, `Ywhaz`, `Gusb` …）
> **注意：这 37 个 HKG 基本不在那 500 个 NC 基因里（仅 4 个重合），
> 所以这是一次真正的 held-out 验证，不是自证。**

### 5.2 UV 的两条通道各自贡献了什么

| ![Δ_cov](ppt_figs/fig7_hkg_delta_cov.png) | ![Δ_lat](ppt_figs/fig8_hkg_delta_lat.png) |
|---|---|
| **Δ_cov**：已知协变量 UV（batch + n_genes_on）| **Δ_lat**：潜在 UV（sample_emb + decoder_w）|

两张图都呈现**按 donor 分层的红蓝条带**，且方向大体互补 ——
说明模型把 donor 效应同时拆给了「可解释的线性协变量」和「潜在因子」，
两者共同还原出观测到的批次偏移。

### 5.3 ⚠️ 需要注意的局限（诚实汇报）

| # | 问题 | 说明 |
|---|---|---|
| 1 | **FDR<0.05 有 10 121 / 16 428 基因（62%）** | Welch t 检验以**单细胞为独立样本**（n≈4821），存在严重**伪重复 (pseudo-replication)**。真实的独立单位是 **7 个 donor**。p 值被极度膨胀。 |
| 2 | **`r2_per_gene_mean = −2.46e7`** | 被极少数近零方差基因的 R² 爆炸主导。应以 **median (0.0812)** 为准，`mean` 无意义。 |
| 3 | **`Gm42418` 排名第一** | 该基因是公认的 **rRNA / ambient RNA 污染代理**，不应作为生物学结论。 |
| 3b | **上下调严重不对称：2640 上 vs 636 下** | 这种不对称本身可疑，提示可能仍有未被 UV 通道吸收的全局偏移或 ambient 残留。 |
| 4 | **donor 与 status 完全共线** | 模型用 NC 基因做锚点来「拆解」共线性，但这依赖 **NC 假设成立**（NC 基因确实不受应激影响）。当前 NC 是 data-driven 选出的（|logFC| 最小的 500 个），存在**循环论证风险**——按最小组间差异选基因，天然会把真实的「无差异」和「被 UV 抵消的差异」混在一起。 |
| 5 | **`y_bio` 的 r2 只有 0.31** | 去 UV 后保留的方差不多，需确认没有连生物信号一起去掉。 |
| 6 | **NC 与 HKG 重合仅 4 个** | data_only 模式选出的 NC（`Olfr1033`, `Neurod6`, `Gm27209`…）包含嗅觉受体、神经元转录因子等**生物学上不该当 NC** 的基因。 |

### 5.4 建议的下一步

| 优先级 | 行动 |
|---|---|
| 🔴 **高** | **改用 pseudobulk 检验**：把 4821 细胞聚合到 7 个 donor，用 donor 作为统计单位重跑 DEG。这会把 FDR<0.05 的基因数降到合理范围。 |
| 🔴 **高** | **NC 选择改为 `hybrid` 或 `hkg_only`**：用文献 HKG 池而非 data-driven，规避循环论证。对比三种 NC_MODE 的 DEG 一致性。 |
| 🟡 中 | **过滤 ambient 基因**：剔除 `Gm42418` / `AY036118` / `mt-*` / `Bc1` 等污染代理后重跑。 |
| 🟡 中 | **与基线方法对比**：MAST / DESeq2-pseudobulk / Harmony+Wilcoxon / RUV-III-PRPS，做 4 方法一致性分析。 |
| 🟡 中 | **切换 ZINB 模式**（`use_zinb=True`，已在 `testRUVVAE_ZINB.ipynb` 中实现）：直接建模 raw counts 的零膨胀，比 log1p+MSE 更贴合单细胞数据生成过程。 |
| 🟢 低 | **敏感性分析**：扫描 `k_unk ∈ {2,4,8,16}`、`d_bio ∈ {32,64,128}`，看 DEG 列表稳定性。 |
| 🟢 低 | **正交验证**：Top DEG 做 GO/KEGG 富集，看是否富集到应激/突触通路（`Cacna1c`/`Nrxn1`/`Shisa9` 已有提示）。 |

---

### 一句话总结

> RUVVAE 用**负对照基因损失**把 RUV 的线性因子模型升级成了可微的深度模型，
> 成功地把与 status 完全共线的 donor/批次效应从生物信号中剥离出来
> （NC 跨 donor 波动 ↓ 到 **0.244×**，校正后 dotplot 呈现清晰的 CSRS vs MW 分块）。
> **但当前的统计检验是细胞层面的，存在伪重复，DEG 数量被高估——
> 下一步必须切换到 donor-level pseudobulk 才能得到可发表的结论。**
