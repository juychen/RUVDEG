# RUV / PRPS / VAE 笔记

> 来源：对话整理。包含六块内容：
> 1. 论文 s41587-022-01440-w（PRPS）解读
> 2. 伪技术重复（pseudo-replicates）的具体构造方法
> 3. RUV 与变分自编码器（VAE）结合的方案草图
> 4. Hybrid RUV-VAE：显式 W 协变量矩阵 + 隐内参向量
> 5. RUV-VAE 作为 MAST DEG 的后处理器
> 6. RUV-VAE 作为 DEG 检测的主方法
>
> 作者：WINGH（PolyU 研究笔记）
> 日期：2026-07-23（Part I–IV），2026-07-24（Part V–VI）

---

## Part I. 论文解读：Removing unwanted variation from large-scale RNA sequencing data with PRPS

### 1.1 论文基本信息

- **标题**：Removing unwanted variation from large-scale RNA sequencing data with PRPS
- **作者**：Ramyar Molania, Momeneh Foroutan, Johann A. Gagnon-Bartsch, Luke C. Gandolfo, Aryan Jain, Abhishek Sinha, Gavriel Olshansky, Alexander Dobrovic, Anthony T. Papenfuss, Terence P. Speed
- **期刊**：Nature Biotechnology, Vol. 41, 2023年1月, 82–95
- **DOI**：https://doi.org/10.1038/s41587-022-01440-w

### 1.2 核心问题

TCGA（The Cancer Genome Atlas）这类大型癌症 RNA-seq 项目包含上万样本、跨多个中心、批次、测序化学和年份，存在三类主要 **非期望变异（unwanted variation, UV）**：

1. **文库大小（library size）** —— 不同样本测序深度不同
2. **肿瘤纯度（tumor purity）** —— 肿瘤样本中癌细胞比例差异
3. **批次效应（plate / flow cell / time effects）** —— 不同测序板/试剂/年份的系统偏差

传统 FPKM、FPKM.UQ 等归一化方法用一个**全局缩放因子**去除文库大小差异，假设"所有基因的 counts 都成比例地反映这个因子"。但文章通过 TCGA-READ 数据证明：

- 不同基因与文库大小呈现**四类不同关系**（正比、低于预期、不相关、相反关系），全局缩放无法同时去除所有类型
- 肿瘤纯度、批次效应则完全无法被传统方法处理

### 1.3 核心算法：RUV-III + PRPS

文章提出把已有的 **RUV-III（Removing Unwanted Variation III）** 方法与新策略 **PRPS（Pseudo-Replicates of Pseudo-Samples，伪样本的伪重复）** 配合使用，专门解决"没有技术重复（technical replicates）"时的归一化难题。

#### 1.3.1 RUV-III 的数学框架（线性模型）

设数据矩阵 $Y$（m 个 assay × n 个基因），引入映射矩阵 $M$（$m \times m_1$，把 assay 映射到唯一的样本），设计矩阵 $X$（表示感兴趣的生物因素），$W$ 为不需要的变异矩阵：

$$Y = \mathbf{1}\mu^T + MX\beta + W\alpha + \varepsilon$$

**关键假设**：存在一组**负对照基因（negative control genes）** $Y_c$，满足 $\beta_c = 0$，即不与感兴趣的生物因素相关，但受到非期望变异 $W$ 的影响。

**关键投影**：
- $P_M = M(M^TM)^{-1}M^T$：把每个 assay 的值替换成同一 unique sample 上所有 assay 的平均值
- $R_M = I - P_M$：对应残差投影，**几乎只反映非期望变异**

对 $R_M Y Y^T R_M$ 做谱分解 $U D U^T$，取前 $k$ 个特征向量 $U_{(k)}$。

#### 1.3.2 RUV-III 的三步归一化

**步骤 I**：从 $R_M Y$ 中提取非期望变异的轮廓：

$$\hat{\alpha}_{(k)} = U_{(k)}^T Y$$

**步骤 II**：用中心化后的负对照基因对 $\hat{\alpha}_{(k)}^T$ 做回归，估计 $W$：

$$\hat{W}_{(k)} = (I - P_1) Y_c \left( U_{(k)}^T Y_c \right)^T \left[ \left( U_{(k)}^T Y_c \right) \left( U_{(k)}^T Y_c \right)^T \right]^{-1}$$

其中 $P_1$ 是向全 1 向量的正交投影。

**步骤 III**：得到归一化后的数据：

$$Y_{(k)} = Y - \hat{W}_{(k)} \hat{\alpha}_{(k)}$$

> **直觉**：先用 $R_M$（伪重复残差）抓住非期望变异方向；用负对照基因校准这些方向的实际幅度；最后从每个基因的表达中扣掉这部分。

#### 1.3.3 PRPS（伪样本的伪重复）

RUV-III 本来需要**技术重复**（同一生物样本在不同批次各测一次）。但 TCGA 等大规模项目里没有这种重复。PRPS 用计算方法构造伪重复。

**算法流程**：

1. **识别非期望变异的来源**（如文库大小、肿瘤纯度、plate、年份、flow cell）
2. **识别相对同质的生物学亚群**：
   - READ/COAD：用 CMS 亚型 × MSI 状态 → 4 × 3 = 12 个亚群
   - BRCA：用 PAM50 亚型 → 5 个亚群
3. **构造伪样本（pseudo-samples）**：把每个亚群内、同一 plate 里的几个样本的表达**取平均**，得到一个 in silico 样本
4. **形成伪重复（pseudo-replicates）**：同一个生物学亚群、不同 plate 上得到的伪样本互为伪重复
5. 把这些伪样本 + 负对照基因送入 RUV-III

#### 1.3.4 负对照基因的选择

策略务实（pragmatic）：
- 从文献找（如 housekeeping 基因）
- 或从数据中找：在 FPKM.UQ 上做 ANOVA（以生物因素为因子），挑 F 统计量最低（约 1000 个基因），并要求它们对肿瘤纯度、文库大小相关性低
- 用 PCA 检验它们是否"能抓住 UV 而不抓住 biology"

#### 1.3.5 参数 k 的选择

在 1 到 $m - m_1$ 之间试多个 k 值，结合 PCA、R²、向量相关、silhouette、ARI 等指标和先验生物学知识选。RUV-III 对 k 的过估计比较稳健，但欠估计不行。

### 1.4 主要结果

应用在三个 TCGA 数据集：

| 数据集 | 主要非期望变异 | 关键发现 |
|---|---|---|
| READ（直肠腺癌，176 样本） | 文库大小 + plate | RUV-III 显著降低基因–文库大小相关性；CMS 亚型分得更清楚；校正了 TMF1/BCLAF1、MDH2/EIF4H 等基因对之间原本虚假的共表达；揭示 RAB18/FBXL14 与生存的关联 |
| COAD（结肠腺癌） | 文库大小 + plate + 时间 | 与 microarray 数据一致性大幅提升；CMS+MSI 亚型分离更清晰 |
| BRCA（乳腺癌，1180 样本） | 文库大小 + 肿瘤纯度 + plate + flow cell chemistry | 同时去除 4 种 UV；去除了 ZEB2/ETS1 因纯度造成的虚假相关；生存分析结果更可信 |

**额外的健壮性验证**：
- 即使随机打乱 20–80% 用于构造 PRPS 的样本，算法仍然表现良好
- 当生物亚群本身就受 UV 强烈影响时（如把 CMS4 本身当亚群），算法性能下降
- 可以跨研究整合——多个 RNA-seq 研究可通过 PRPS 同时归一化

### 1.5 与已有方法的关系

- RUVg、RUVs、SVAseq、ComBat-seq 等都假设"非期望变异与生物因素正交"——实际癌症数据里几乎不成立
- 已有"伪重复"思想（如 EB++）也用过，但本文是第一个把它和 RUV-III 严格连接、并在癌症 RNA-seq 上系统验证的工作

---

## Part II. 伪技术重复是怎么做的

### 2.1 核心思想

| 真实技术重复 | PRPS 的伪技术重复 |
|---|---|
| 同一生物样本，被分到不同批次处理 | 生物上同质（同一亚型）但在不同批次（plate/年份/高低文库大小）的若干**真实样本**，被**取平均**当作一个"伪样本" |
| 重复之间只差处理差异 | 伪样本之间理论上只差"批次/非期望变异" |
| 直接提供 RUV-III 需要的复制结构 | 提供同样的复制结构 |

**关键前提**：必须先定义"生物学亚群"（homogeneous biological subpopulation），让同一亚群内的样本在生物上是可比的；它们之间的差异主要来自非期望变异，而不是生物学。

### 2.2 构造流程（5 步）

#### Step 1：识别要消除的非期望变异来源

每个来源都要单独构造一组 PRPS。常见 UV 来源：
- 文库大小（library size）
- 肿瘤纯度（tumor purity）
- 测序板（plate）/ 时间（year）
- 流式细胞化学（flow cell chemistry）
- 组织来源站点（TSS）

#### Step 2：识别同质的生物学亚群

- **READ / COAD**：用 R 包 **CMScaller** 算 CMS 共识分子亚型 × **MSI 状态**（MSI-H / MSI-L / MSS）→ 4 × 3 = 12 个亚群（READ 实际只用到 11 个亚群，因为缺少 CMS4 + MSI-H）
- **BRCA**：用 R 包 **genefu** 算 **PAM50** 亚型（Basal / Her2 / Luminal A / Luminal B / Normal-like）→ 5 个亚群
- 在 FPKM 和 FPKM.UQ 两个数据集上取**共识亚型**（consensus calls），只保留两个方法分型一致的样本

#### Step 3：对每个 (亚群 × 批次) 组合构造伪样本

**方式 A —— 简单平均（用于去除 plate / 时间效应）**

- 在某个 plate 内、属于某亚群的样本 → 取**该亚群在该 plate 上所有样本**的表达平均，得到 1 个伪样本
- 论文要求：每个亚群在至少 2 个 plate 上至少有 ≥3 个样本（COAD 标准），或 ≥2 个样本（READ 标准）
- 例如：CMS2 + MSI-stable 亚群 + Plate 1 → 伪样本 $P_1$；CMS2 + MSI-stable 亚群 + Plate 2 → 伪样本 $P_2$

**方式 B —— 极端分层平均（用于去除文库大小 / 肿瘤纯度等连续型 UV）**

这是 BRCA 数据集论文实际采用的关键技巧，**PRPS 的精髓**：

**构造"文库大小版"伪样本**：
1. 在每块 plate 上，挑出某 PAM50 亚型的所有样本（要求 ≥12 个）
2. 把这些样本按文库大小排序
3. 取**最高的 3 个**样本的表达平均 → 高文库大小伪样本 $H_{plate,subtype}$
4. 取**最低的 3 个**样本的表达平均 → 低文库大小伪样本 $L_{plate,subtype}$
5. 这一对 ($H$, $L$) 形成一个伪重复对

**构造"肿瘤纯度版"伪样本**：同样逻辑，但用肿瘤纯度排序

这样伪重复对之间的差异就**主要反映**该 UV（文库大小或纯度），而不是其他来源的 UV。

#### Step 4：组装伪重复集

- 同一亚群 + 同一 UV 来源 + 不同 plate 产生的伪样本 → 构成一个**伪重复集（pseudo-replicate set）**
- 在 RUV-III 中，这个集合通过映射矩阵 $M$ 表示：哪几个伪样本对应同一个"独特样本 $h$"

#### Step 5：不同 UV 来源各自独立跑一次 RUV-III

论文 BRCA 数据集就同时建了 3 组 PRPS：
- (a) 文库大小 PRPS
- (b) 肿瘤纯度 PRPS
- (c) plate + flow cell PRPS（这两个完全 confounded，合并处理）

每次 RUV-III 跑完得到一组归一化数据，可以**串行应用**或**联合估计**多个非期望变异维度。

### 2.3 为什么这样能行？

从 RUV-III 的公式倒推：

$$R_M = I - M(M^TM)^{-1}M^T$$

$R_M$ 把每个伪样本替换成"同一独特伪样本上所有 assay 的均值"，残差就是这些伪样本之间的差异。

- 如果伪样本**生物学同质**（同亚型），那这些差异**不该是生物学的**
- 如果它们来自**不同 plate**，那这些差异**就是 plate 效应**（或其他 UV）
- 再用负对照基因校准幅度 → 精确估计 $W\alpha$ → 扣掉

> 直觉：**伪重复对之间的"差异信号" ≈ UV 信号**，把它减掉就得到干净数据。

### 2.4 关键参数与陷阱

| 参数 | 论文建议 | 说明 |
|---|---|---|
| 每个伪样本最少样本数 | ≥3（文库/纯度版）/ ≥3（plate 版） | 太少平均意义不大，太多又难满足"同质" |
| 每个亚群最少 plate 数 | ≥2 | 否则形不成伪重复对 |
| 亚群本身的纯度 | 亚群不能被要消除的 UV 强烈污染 | 论文特别提醒：若用 CMS4（本身就受文库大小影响大）做亚群，效果会下降 |
| 生物学亚群是否要做 consensus | 是 | FPKM 和 FPKM.UQ 都要算亚型，取交集 |
| 负对照基因是否要随亚群调整 | 可以 | BRCA 中在每个 flow cell chemistry 内做 ANOVA 找对照基因 |

### 2.5 具体例子（虚构，便于理解）

假设你做 3 个 plate 的 BRCA，每板 6 个 Luminal A 样本：

- **Plate 1**（LumA）：libsize 排序后 → L1A-1(高), L1A-2, L1A-3, L1A-4, L1A-5, L1A-6(低)
- **Plate 2**（LumA）：类似 L2A-1(高) … L2A-6(低)
- **Plate 3**（LumA）：类似 L3A-1(高) … L3A-6(低)

**构造文库大小 PRPS**：

- $H_1$ = mean(L1A-1, L1A-2, L1A-3) → 高文库 LumA 伪样本（plate 1）
- $L_1$ = mean(L1A-4, L1A-5, L1A-6) → 低文库 LumA 伪样本（plate 1）
- $H_2, L_2, H_3, L_3$ 类似

伪重复对：{($H_1, L_1$), ($H_2, L_2$), ($H_3, L_3$)}  
→ 同一 LumA 亚型、不同 plate、刻意挑了文库大小极端 → 差异主要就是文库大小本身

送入 RUV-III → 估计文库大小方向的 $W$ → 扣除。

### 2.6 实操建议

1. **生物学亚群**：用领域公认的分类（细胞类型亚群、临床亚型、组织区域），不要凭空聚类
2. **批次轴**：先用 PCA / RLE / 向量相关确定 UV 来源
3. **构造伪样本**：可以用 `dplyr` 或 `data.table` 按 (subtype, batch) 分组 `summarise_all(mean)`，但要确保每组样本数 ≥3
4. **构造 RUV-III 输入**：用 `RUVIII` 包，传入：
   - 原始表达矩阵 $Y$
   - 伪样本矩阵（作为 replicates）
   - 负对照基因列表
   - $k$ 值（用 `getK()` 在一系列候选值中按 RLE / silhouette 选最优）
5. **评估**：跑完用 RLE 看中位数是否归零、PCA 看是否还有批次聚类、silhouette 看亚型分离是否变好

---

## Part III. RUV + 变分自编码器的结合方案

### 3.1 为什么 RUV 和 VAE 是天生一对？

| 维度 | 经典 RUV-III | VAE 视角 |
|---|---|---|
| 数据生成假设 | $Y = MX\beta + W\alpha + \varepsilon$（线性） | $Y \sim p_\theta(y \mid z, w)$（可非线性） |
| UV 的载体 | 显式矩阵 $W$（m×k） | 潜变量 $w_i \in \mathbb{R}^k$（每样本一个） |
| 估计 UV 的依据 | 伪重复残差 $R_M Y$ + 负对照 | 重建损失 + 负对照约束 + KL |
| 优势 | 可解释、稳定 | 灵活、自动学潜空间 |
| 劣势 | 线性、需手工 PRPS | 需要解决"潜空间解耦" |

**关键洞察**：RUV 的非期望变异 $W\alpha$ 在 VAE 框架里等价于一个"技术噪声解码器"——每个样本的潜变量 $w_i$ 解码出该样本独有的"污染矩阵"。

### 3.2 推荐的架构：RUV-VAE

#### 3.2.1 模型图

```
输入 y_i (n_genes 维原始表达向量，raw count 或 log 归一)
        │
        ├── Encoder_E(y_i) ──→ μ_i, σ_i ─→ z_i ∈ R^d (生物学潜变量)
        │
        └── Encoder_W(y_i) ──→ w_i ∈ R^k (技术变异潜变量)
                    │
                    └── Decoder_W(w_i) ──→ Δ_i ∈ R^{n_genes} (技术污染贡献)
                                    │
                                    ↓
                          重建 x̂_i = Decoder_Bio(z_i) − Δ_i
```

**两个解码器**：
- $D_{bio}(z_i)$：从生物学潜变量重建"干净表达"
- $D_W(w_i)$：从技术潜变量重建"污染贡献"

**核心假设**：$\Delta_i$ 就是要去掉的非期望变异。

#### 3.2.2 数学形式

**编码**（每样本 i）：

$$z_i \sim q_\phi(z \mid y_i) = \mathcal{N}(\mu_z(y_i), \sigma_z^2(y_i) I)$$

$$w_i \sim q_\psi(w \mid y_i) = \mathcal{N}(\mu_w(y_i), \sigma_w^2(y_i) I)$$

**解码**（生物学部分，**用户想要的"内参隐变量"**）：

$$\hat{y}_i^{bio} = D_{bio}(z_i)$$

**解码**（技术部分，**RUV 的 $W\alpha$ 化身**）：

$$\Delta_i = D_W(w_i) = W \cdot \mathrm{softplus}(w_i)$$

其中 $W \in \mathbb{R}^{n\_genes \times k}$ 是可学习矩阵，每一列对应一种"变异方向"。

**最终输出**（推荐**减法 + 重建损失**的形式，更稳定）：

$$y_i^{norm} = \mathrm{softplus}\left( D_{bio}(z_i) - D_W(w_i) \right)$$

### 3.3 损失函数

#### 3.3.1 标准 VAE 损失

$$\mathcal{L}_{VAE} = \underbrace{-\mathbb{E}_{q}[\log p_\theta(y \mid z, w)]}_{\text{重建}} + \beta_1 \underbrace{KL(q(z) \| p(z))}_{\text{生物潜变量正则}} + \beta_2 \underbrace{KL(q(w) \| p(w))}_{\text{技术潜变量正则}}$$

对于 RNA-seq 计数数据，$p(y \mid z, w)$ 通常用 **负二项分布（NB）** 或 **零膨胀 NB（ZINB）**，参考 scVI 的做法。

#### 3.3.2 RUV 风格约束：负对照基因锚定

**这是把 RUV 的灵魂注入 VAE 的关键**。假设你有 $n_c$ 个负对照基因（housekeeping 或文献指定）：

$$\mathcal{L}_{NC} = \frac{1}{n_c} \sum_{g \in NC} \left\| \Delta_{i,g} - y_{i,g} \right\|^2$$

含义：负对照基因的"真实表达"应该等于"技术污染"的相反数（即应被完全扣除）。或更宽松的版本：

$$\mathcal{L}_{NC} = -\log p(y_{i,NC} \mid w_i)$$

即：**只用 $w$ 就能预测负对照基因的表达**——这正是 RUV "负对照基因不受生物学影响"假设的生成式翻译。

#### 3.3.3 解耦损失（防止 $w$ 偷藏生物学）

$$\mathcal{L}_{dis} = \underbrace{\text{MI}(w; \text{batch})}_{\text{最大化 } w \text{ 与批次相关}} - \underbrace{\text{MI}(z; \text{batch})}_{\text{最小化 } z \text{ 与批次相关}}$$

或者用对抗训练：

$$\mathcal{L}_{adv} = \max_\eta \mathbb{E}_i \log P_\eta(\text{batch}_i \mid w_i)$$

$$\mathcal{L}_{anti-adv} = \min_\phi \mathbb{E}_i \log P_\eta(\text{batch}_i \mid z_i)$$

#### 3.3.4 总损失

$$\mathcal{L} = \mathcal{L}_{VAE} + \lambda_1 \mathcal{L}_{NC} + \lambda_2 \mathcal{L}_{dis}$$

### 3.4 和已有工作的关系

| 工作 | 做了什么 | 与本想法的距离 |
|---|---|---|
| **scVI** (Lopez et al., 2018) | 条件 VAE，每个细胞有 size factor 潜变量 + batch one-hot | **最近！** 但 size factor 是标量；本方案 $w_i$ 是向量，能表达多种 UV |
| **scANVI** (Xu et al., 2021) | scVI + 半监督细胞类型标签 | 可借鉴的解耦机制 |
| **DCA** (Eraslan et al., 2019) | 计数自编码器 + ZINB 损失 | 通用去噪框架，没有显式 UV 潜变量 |
| **Harmony** (Korsunsky et al., 2019) | 迭代 PCA 软聚类去批次 | 不是生成式 |
| **scGen** (Lotfollahi et al., 2019) | VAE + 扰动预测 | 可借鉴的解耦机制 |
| **removeBatchEffect** (limma) | 线性回归扣批次 | 线性方法 |
| **PRPS + RUV-III** (Molania et al., 2023) | 刚读的论文 | **直接对照基线** |

**差异化卖点**：
1. PRPS 需要手工构造伪样本和亚群；RUV-VAE 完全数据驱动
2. RUV-III 是线性模型，RUV-VAE 能捕获非线性 UV（如 RNA 降解非线性效应）
3. "内参隐变量"概念比 scVI 的 scalar size factor 更通用，可推广到多源 UV（library size + purity + plate 同时学多个 $w_i$）

### 3.5 训练与使用细节

#### 3.5.1 输入预处理

- **不要做 FPKM/UQ**！输入 raw count 最好
- log1p 一下更稳：$y_i^{in} = \log(1 + y_i)$
- 文库大小作为**协变量**拼接进 encoder（也可以让它从 $w_i$ 学出来）

#### 3.5.2 负对照基因的选择

参考 PRPS 论文的两类策略：
- **文献**：housekeeping 基因（如 GAPDH, ACTB, RPL…）
- **数据驱动**：在 PCA 上对每个基因计算与 plate / library size 的相关性，挑高相关、低生物信号变异者

#### 3.5.3 输出解读

训练完：
- $z_i$：细胞的生物学潜表示 → 可用于聚类、UMAP、差异表达
- $w_i$：技术变异潜表示 → 可用作诊断工具（看哪些样本 $w$ 异常）
- $D_W$ 的权重矩阵 → 等价于 RUV 的 $W$，**可解释为"非期望变异方向"**

#### 3.5.4 参考实现骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RUVVAE(nn.Module):
    """
    RUV-VAE: 把 RUV-III 的负对照+伪重复思想与 VAE 结合的归一化框架
    """

    def __init__(self, n_genes, d=32, k=5):
        super().__init__()
        # 生物学编码器
        self.encoder_z = nn.Sequential(
            nn.Linear(n_genes, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.z_mu = nn.Linear(128, d)
        self.z_logvar = nn.Linear(128, d)

        # 技术变异编码器
        self.encoder_w = nn.Sequential(
            nn.Linear(n_genes, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.w_mu = nn.Linear(128, k)
        self.w_logvar = nn.Linear(128, k)

        # 生物学解码器
        self.decoder_bio = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, n_genes),
        )

        # RUV 风格的 UV 方向矩阵 W (n_genes × k)
        self.W = nn.Parameter(torch.randn(n_genes, k) * 0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, neg_control_mask):
        # 编码
        h_z = self.encoder_z(y)
        z_mu, z_logvar = self.z_mu(h_z), self.z_logvar(h_z)
        z = self.reparameterize(z_mu, z_logvar)

        h_w = self.encoder_w(y)
        w_mu, w_logvar = self.w_mu(h_w), self.w_logvar(h_w)
        w = self.reparameterize(w_mu, w_logvar)

        # 解码
        y_bio = self.decoder_bio(z)         # 生物学贡献
        delta = w @ self.W.T                 # 技术污染贡献 (batch, n_genes)

        # 干净表达（减法版）
        y_clean = F.softplus(y_bio - delta)

        # 负对照约束：只用 w 就能预测负对照基因的表达
        nc_loss = F.mse_loss(
            delta[:, neg_control_mask],
            y[:, neg_control_mask]
        )

        # KL 散度
        kl_z = -0.5 * torch.mean(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp()
        )
        kl_w = -0.5 * torch.mean(
            1 + w_logvar - w_mu.pow(2) - w_logvar.exp()
        )

        return {
            "y_clean": y_clean,
            "z": z_mu,          # 用 mu 而非采样值做下游分析
            "w": w_mu,
            "nc_loss": nc_loss,
            "kl_z": kl_z,
            "kl_w": kl_w,
        }
```

训练循环示意：

```python
model = RUVVAE(n_genes=Y.shape[1], d=32, k=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# NB / ZINB 重建损失（参考 scVI）
def nb_loss(y, mu, theta, eps=1e-8):
    """y ~ NB(mu, theta)"""
    log_mu = torch.log(mu + eps)
    log_y = torch.log(y + eps)
    return (theta * (log_mu - log_y)
            + y * (torch.log(theta + eps) - log_y)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1))

for epoch in range(num_epochs):
    for batch in dataloader:
        y = batch["x"]              # (batch_size, n_genes)
        mask_nc = batch["mask_nc"]  # (n_genes,) bool

        out = model(y, mask_nc)
        recon = nb_loss(y, out["y_clean"], theta=1.0).mean()

        loss = (
            recon
            + 1e-3 * out["kl_z"]
            + 1e-3 * out["kl_w"]
            + 1.0  * out["nc_loss"]   # 负对照约束权重 λ1
            + 1.0  * dis_loss         # 解耦损失 λ2（需额外实现）
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.6 实验设计建议

如果你要把这个写成论文，建议对比：

| 方法 | 类型 | 关键指标 |
|---|---|---|
| FPKM.UQ | 传统全局归一化 | baseline |
| RUV-III + PRPS | 线性 + 手工 PRPS | 直接对照基线 |
| scVI | VAE（无显式 UV） | 看条件 VAE 能否捕获 UV |
| Harmony | 迭代聚类 | 非生成式基线 |
| **RUV-VAE** | VAE + 显式 UV + NC 约束 | 应在所有指标上最优 |

**关键评测**：

1. **模拟数据**：人为注入已知的非线性 UV（plate × library size × tumor purity），看 $w_i$ 能否恢复
2. **真实数据**：用 TCGA-BRCA 看 $w_i$ 是否和 plate / flow cell / purity 强相关
3. **下游任务**：差异表达、共表达、生存分析的提升幅度

### 3.7 进一步研究方向

"RUV-as-a-Generative-Model"——把 RUV 系列方法（I/II/III/IV）都翻译成生成式潜变量框架：

- RUV-I：负对照基因均值归零 → 对应 $w$ 拟合负对照
- RUV-II：M 估计回归 → 对应 robust likelihood
- RUV-III：伪重复残差 → 对应 $w$ 拟合伪重复差
- RUV-IV：随机效应模型 → 对应 hierarchical VAE

---

## Part IV. Hybrid RUV-VAE：显式 W 协变量矩阵 + 隐内参向量

### 4.1 动机

把 RUV 拆成两部分同时建模：

- **隐变量部分**：捕获**未知 / 非线性**的 UV（PRPS 现在也搞不定的）
- **显式 W 协变量矩阵**：捕获**已知**的 UV（batch、library size、tumor purity），让模型有可解释的归纳偏置

这是把 scVI/scANVI 的"条件 VAE"思路 + RUV 的"负对照锚定" + 经典 RUV 的"$X\beta$ 显式因子"三股力量捏在一起。

### 4.2 架构：Hybrid RUV-VAE

```
输入:
  y_i (n_genes 维表达)
  c_i (n_cov 维协变量: one-hot batch + log library size + tumor purity + ...)

           ┌──────────────────────────────────────────────┐
           │                                              │
           ▼                                              ▼
   Encoder_E(y_i)                                  Encoder_W(y_i)
   → μ_z, σ_z → z_i ∈ R^d                      → μ_w, σ_w → w_i ∈ R^k
       (生物学)                                       (未知 UV 潜变量)
           │                                              │
           ▼                                              ▼
   D_bio(z_i)                                    Δ_latent = D_W(w_i)
   y_i^bio                                       (潜变量 UV 贡献)
           │                                              │
           │              显式 W 协变量分支                 │
           │   ┌─────────────────────────────────────┐   │
           │   │ c_i ∈ R^{n_cov}                     │   │
           │   │       ×                             │   │
           │   │ W_cov ∈ R^{n_cov × n_genes}  (学习) │   │
           │   │       ↓                             │   │
           │   │ Δ_cov,i = c_i · W_cov               │   │
           │   └─────────────────────────────────────┘   │
           │                                              │
           └──────────────────┬───────────────────────────┘
                              ▼
                y_clean = softplus(y_bio − Δ_latent − Δ_cov)
```

**核心公式**：

$$y_i^{clean} = \mathrm{softplus}\Big( D_{bio}(z_i) - D_W(w_i) - c_i \, W_{cov} \Big)$$

### 4.3 数学形式（完整版）

#### 4.3.1 编码

$$z_i \sim q_\phi(z \mid y_i), \quad w_i \sim q_\psi(w \mid y_i)$$

#### 4.3.2 三个生成项

| 项 | 含义 | 来源 |
|---|---|---|
| $D_{bio}(z_i)$ | 干净生物学信号 | 潜变量 $z$ 解码 |
| $D_W(w_i)$ | 未知 / 残差 UV | 潜变量 $w$ 解码 |
| $c_i W_{cov}$ | **已知 UV 的显式效应** | 协变量 × 可学习 W 矩阵 |

#### 4.3.3 $W_{cov}$ 的设计选择

**选项 A：单一 W 矩阵（最简单）**

$$W_{cov} \in \mathbb{R}^{n_{cov} \times n_{genes}}$$

$$\Delta_{cov,i} = c_i \, W_{cov} \in \mathbb{R}^{n_{genes}}$$

**选项 B：分块 W（推荐，更可解释）**

按协变量类型分块：

$$W_{cov} = \begin{bmatrix} W_{batch} \\ W_{libsize} \\ W_{purity} \end{bmatrix}$$

每块单独：
- $W_{batch} \in \mathbb{R}^{n_{batch} \times n_{genes}}$
- $W_{libsize} \in \mathbb{R}^{1 \times n_{genes}}$（连续型，可学一个基因向量）
- $W_{purity} \in \mathbb{R}^{1 \times n_{genes}}$

```python
Δ_cov = c_batch @ W_batch + c_libsize @ W_libsize + c_purity @ W_purity
```

**选项 C：低秩分解（参数更省）**

$$W_{cov} = U V, \quad U \in \mathbb{R}^{n_{cov} \times r}, \quad V \in \mathbb{R}^{r \times n_{genes}}$$

适合 $n_{cov}$ 很大的情况（比如 one-hot 编码后维度爆炸）。

**选项 D：稀疏 + 共享（最像 RUV 原版）**

- 让 $W_{cov}$ 只在**负对照基因子集**上学习 → 其他基因直接共享同一 $W_{cov}$
- 避免过拟合，保留 RUV 的"全局方向"思想

### 4.4 损失函数

#### 4.4.1 完整损失

$$\mathcal{L} = \mathcal{L}_{recon} + \beta_1 KL(z) + \beta_2 KL(w) + \lambda_1 \mathcal{L}_{NC}^{latent} + \lambda_2 \mathcal{L}_{NC}^{cov} + \lambda_3 \mathcal{L}_{dis}$$

其中：
- $\mathcal{L}_{NC}^{latent}$ = 负对照基因可由 $w$ 预测
- $\mathcal{L}_{NC}^{cov}$ = **新增**：负对照基因也可由 $c_i W_{cov}$ 预测

#### 4.4.2 负对照基因的双重锚定（核心创新点）

经典 RUV 只用：

$$\min_W \| y_{NC} - c W \|_{F}^2$$

我们的混合版本：

$$\mathcal{L}_{NC}^{cov} = \frac{1}{n_c} \sum_{g \in NC} \left\| y_{i,g} - (c_i W_{cov})_g \right\|^2$$

含义：**$W_{cov}$ 必须能解释负对照基因里所有协变量相关的变化**。这是 RUV 的"负对照 ↔ 协变量回归"在神经网络里的直接翻译。

#### 4.4.3 解耦正则（让 $W_{cov}$ 和 $w$ 不重复学同一件事）

加一个软正交：

$$\mathcal{L}_{orth} = \| (c_i W_{cov})^T \cdot D_W(w_i) \|_F^2$$

惩罚两个 UV 贡献之间的相关性，鼓励它们各管各的：
- $W_{cov}$：抓**已知的、可解释的** UV
- $w_i$：抓**未知的、残差的** UV

### 4.5 和已有工作的关系

| 工作 | 做了什么 | 距离 |
|---|---|---|
| **scVI** | batch one-hot 拼到 encoder / decoder | 只用 batch，**没有显式 W** |
| **scANVI** | scVI + 标签半监督 | 同上 |
| **Harmony** | 软聚类去批次 | 显式 batch 但无潜变量 |
| **RUV-III** | 显式 W = 伪重复残差 | **W 是隐式学的**，没有协变量 |
| **ComBat-seq** | 显式 batch 回归 | **纯线性**，无潜变量 |
| **scGen / CPA** | VAE + batch + 扰动 | batch 嵌入到潜空间，不显式 W |
| **Hybrid RUV-VAE** | 潜变量 $z, w$ + 显式 $W_{cov}$ + 双重 NC 锚定 | **三合一** |

**卖点**：第一次同时具备

1. 非线性（深度网络）
2. 可解释（$W_{cov}$ 矩阵可视化）
3. 已知 UV 显式建模（不受潜空间耦合干扰）
4. 未知 UV 仍可学（$w$ 兜底）

### 4.6 PyTorch 参考实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRUVVAE(nn.Module):
    """
    Hybrid RUV-VAE:
      - Latent biology z_i
      - Latent unknown UV w_i
      - Explicit W_cov: known covariates × learnable gene weights
      - Dual negative-control anchoring on both w and W_cov
    """

    def __init__(
        self,
        n_genes,
        n_cov,                # total covariate dim (after one-hot / encoding)
        d_bio=32,
        k_unk=5,
        cov_blocks=None,      # e.g. {"batch": 8, "libsize": 1, "purity": 1}
    ):
        super().__init__()
        self.n_genes = n_genes

        # ---- 生物学编码器 ----
        self.encoder_z = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
        )
        self.z_mu     = nn.Linear(256, d_bio)
        self.z_logvar = nn.Linear(256, d_bio)

        # ---- 未知 UV 编码器 ----
        self.encoder_w = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
        )
        self.w_mu     = nn.Linear(256, k_unk)
        self.w_logvar = nn.Linear(256, k_unk)

        # ---- 生物学解码器 ----
        self.decoder_bio = nn.Sequential(
            nn.Linear(d_bio, 256), nn.GELU(),
            nn.Linear(256, n_genes),
        )

        # ---- 未知 UV 解码器（潜变量 → 污染）----
        self.decoder_w = nn.Sequential(
            nn.Linear(k_unk, 256), nn.GELU(),
            nn.Linear(256, n_genes),
        )

        # ---- 显式 W 协变量矩阵（按块）----
        if cov_blocks is None:
            # 默认当作一个矩阵
            self.W_cov = nn.Parameter(torch.randn(n_cov, n_genes) * 0.01)
            self.cov_blocks = None
        else:
            # 分块，每块单独学习，参数更可解释
            self.cov_blocks = nn.ModuleDict()
            for name, dim in cov_blocks.items():
                self.cov_blocks[name] = nn.Parameter(
                    torch.randn(dim, n_genes) * 0.01
                )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def compute_delta_cov(self, c_dict):
        """
        c_dict: 例如 {"batch": (B, n_batch), "libsize": (B,1), "purity": (B,1)}
        """
        if self.cov_blocks is None:
            return c_dict["all"] @ self.W_cov
        delta = 0
        for name, W in self.cov_blocks.items():
            delta = delta + c_dict[name] @ W
        return delta

    def forward(self, y, c_dict, neg_control_mask):
        # ---- 编码 ----
        h_z = self.encoder_z(y)
        z_mu, z_logvar = self.z_mu(h_z), self.z_logvar(h_z)
        z = self.reparameterize(z_mu, z_logvar)

        h_w = self.encoder_w(y)
        w_mu, w_logvar = self.w_mu(h_w), self.w_logvar(h_w)
        w = self.reparameterize(w_mu, w_logvar)

        # ---- 三个 UV 贡献 ----
        y_bio     = self.decoder_bio(z)         # 生物学
        delta_lat = self.decoder_w(w)           # 未知 UV
        delta_cov = self.compute_delta_cov(c_dict)  # 已知 UV

        y_clean = F.softplus(y_bio - delta_lat - delta_cov)

        # ---- 损失项 ----
        # 1) 潜变量 NC 锚定
        nc_loss_lat = F.mse_loss(
            delta_lat[:, neg_control_mask],
            y[:, neg_control_mask]
        )
        # 2) 显式 W NC 锚定
        nc_loss_cov = F.mse_loss(
            delta_cov[:, neg_control_mask],
            y[:, neg_control_mask]
        )
        # 3) KL
        kl_z = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        kl_w = -0.5 * torch.mean(1 + w_logvar - w_mu.pow(2) - w_logvar.exp())
        # 4) 解耦：两个 UV 贡献正交
        orth = (delta_lat * delta_cov).mean()

        return {
            "y_clean": y_clean,
            "z": z_mu,
            "w": w_mu,
            "W_cov": {k: v for k, v in (self.cov_blocks or {"all": self.W_cov}).items()},
            "loss_recon_aux": (delta_lat, delta_cov),   # 给外部 NB 重建损失用
            "nc_lat": nc_loss_lat,
            "nc_cov": nc_loss_cov,
            "kl_z": kl_z,
            "kl_w": kl_w,
            "orth": orth,
        }


# ----------------- 训练循环示意 -----------------
model = HybridRUVVAE(
    n_genes=2000,
    n_cov=10,
    d_bio=32,
    k_unk=5,
    cov_blocks={"batch": 8, "libsize": 1, "purity": 1},
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        y       = batch["x"]            # (B, n_genes)
        c_dict  = batch["cov"]          # dict of tensors
        mask_nc = batch["mask_nc"]      # (n_genes,) bool

        out = model(y, c_dict, mask_nc)

        # NB 重建损失（用 clean 表达作为 μ）
        recon = nb_loss(y, out["y_clean"], theta=1.0).mean()

        loss = (
            recon
            + 1e-3 * out["kl_z"]
            + 1e-3 * out["kl_w"]
            + 1.0  * out["nc_lat"]   # λ1
            + 1.0  * out["nc_cov"]   # λ2  ← 新增：锚定 W_cov
            + 0.1  * out["orth"]     # λ3  ← 新增：解耦
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
```

### 4.7 $W_{cov}$ 的可视化与解读

训练完后可做：

```python
# 1. 看每个 batch 对每个 gene 的影响
W_batch = model.cov_blocks["batch"].detach().cpu()  # (n_batch, n_genes)

# 2. 画热图：行=batch，列=top variable genes
top_genes = W_batch.var(dim=0).topk(200).indices
plt.figure(figsize=(10, 4))
plt.imshow(W_batch[:, top_genes], aspect="auto", cmap="RdBu_r", center=0)
plt.xlabel("Top variable genes"); plt.ylabel("Batch")
plt.title("W_cov: batch effect on each gene")
plt.colorbar(); plt.show()

# 3. 看 library size 对每个基因的影响
W_ls = model.cov_blocks["libsize"].detach().cpu().squeeze()  # (n_genes,)
# 排序找最被 library size 影响的基因
top_ls_genes = W_ls.abs().topk(50).indices
```

**这些图直接对应 RUV 论文 Fig.5 的内容**，可以放在论文里当 "explicit W matrix provides interpretable batch effect estimation" 的卖点。

### 4.8 和 RUV 系列的对应关系（升级版）

| RUV 系列 | 显式 W | 潜变量 | 你的 Hybrid 模型对应 |
|---|---|---|---|
| RUV-I | 负对照均值为 0 | 无 | $W_{cov}$ 锚定 + KL 关闭 $w$ |
| RUV-II | 负对照均值回归 | 无 | $W_{cov}$ = 回归系数 + robust likelihood |
| RUV-III | **伪重复残差学出来** | 无 | $W_{cov}$ 来自协变量 + $w$ 兜底 |
| RUV-IV | 随机效应 | 隐式 | hierarchical VAE + $W_{cov}$ |
| **Hybrid RUV-VAE** | **协变量 × 可学习 W** | **$z, w$ 双潜变量** | **本方案** |

你的方案实际上是一个**统一框架**，把 RUV 四个版本用"显式 vs 潜变量"的二维空间覆盖了：

```
                   显式 W 已知            显式 W 未知
                ┌────────────────────┬────────────────────┐
   潜变量 无    │  RUV-I/II (PRPS)   │  RUV-III 显式版    │
                ├────────────────────┼────────────────────┤
   潜变量 有    │  Hybrid RUV-VAE ★  │  scVI/scANVI       │
                └────────────────────┴────────────────────┘
```

最右下角的 scVI/scANVI 没有显式 W，但有潜变量；最左上角的 RUV-I/II 没有潜变量，但有显式 W；**你正好把这两个对角的组合给填上了**。

### 4.9 训练小贴士

1. **先关 $w$，只训 $W_{cov}$**：前几个 epoch 把 `k_unk=5` 关掉（或者 `freeze encoder_w`），让 $W_{cov}$ 先收敛到 RUV 的解析解附近。
2. **再开 $w$ 兜底**：解冻 $w$，让它处理 $W_{cov}$ 解释不了的残差 UV。
3. **NC 权重衰减**：$\lambda_1, \lambda_2$ 可以从 1.0 慢慢升到 5.0。
4. **学习率分离**：$W_{cov}$ 用 `1e-3`，encoder/decoder 用 `5e-4`，避免 W 矩阵震荡。
5. **W 矩阵初始化稀疏**：用 `-lr * LogUniform(1e-4, 1e-2)` 这种初始化让大部分 gene-covariate 系数接近 0，相当于 RUV 的"只在 NC 上估"。

### 4.10 实验设计补充

在 Part III 的对比基础上，加入两个新基线：

| 方法 | 类型 | 关键差异 |
|---|---|---|
| **Hybrid RUV-VAE (no W_cov)** | 仅潜变量 | 退化为 Part III 的 RUV-VAE |
| **Hybrid RUV-VAE (no w)** | 仅显式 W | 退化为带 NC 约束的线性回归 + 生物学 VAE |
| **Hybrid RUV-VAE (full)** | 两者兼有 | 完整模型 |

**消融实验要点**：
- 拆掉 $W_{cov}$ → 看 $w$ 是否被迫吸收已知 UV（应该会，证明 $W_{cov}$ 必要）
- 拆掉 $w$ → 看 $W_{cov}$ 是否能解释所有 UV（应该不行，证明 $w$ 必要）
- 拆掉 NC 约束 → 看 $W_{cov}$ 是否还能正确识别 batch 效应

---

## Part V. RUV-VAE 作为 MAST DEG 的后处理器

### 5.1 动机：为什么做后处理而不是从头跑

如果已经用 MAST（DESeq2 / edgeR）跑出了 DEG 表格：

```
完整表达矩阵 Y (所有基因 × 所有样本)
       │
       ├─→ MAST ──→ DEG 基因列表 + log2FC_obs
       │
       └─→ RUV 分解 ──→ 内参隐变量 Ŵ
                              │
                              ▼
              对每个 DEG 基因 g：
                log2FC_obs(g) ≈ log2FC_true(g) + bias_UV(g)
                              │
                              ▼
                log2FC_corrected(g) = log2FC_obs(g) - bias_UV(g)
```

**核心假设**：DEG 基因的表达同时承载两件事——真实生物学差异 + 内参隐变量 $\hat{W}$（RUV 估计的 UV 方向）的污染。RUV 用负对照基因把 $\hat{W}$ 的结构估计出来了，对每个 DEG 基因只需把在这个 $\hat{W}$ 上的投影扣掉。

**优势**：
- 不破坏 MAST 的统计推断框架（hurdle model、自由度、p-value）
- 不需要构造伪重复
- 只针对已筛出的几百~几千个 DEG 基因做校正，计算量极小

### 5.2 数学分解

RUV-III 的数据分解：

$$Y = \underbrace{MX\beta}_{\text{生物信号}} + \underbrace{W\alpha}_{\text{非期望变异}} + \varepsilon$$

对任意基因 $g$（包括 DEG 基因），其表达向量：

$$Y_g = \beta_g \cdot X + \underbrace{\hat{W} \hat{\alpha}_g}_{\text{Ŵ 的贡献}} + \varepsilon_g$$

对两组比较（A vs B），log2FC 分解为：

$$\begin{aligned}
\log_2 FC_{obs}(g) &= \bar{Y}_g^A - \bar{Y}_g^B \\
&= \underbrace{(\beta_g^A - \beta_g^B)}_{\text{真实 log2FC}} 
+ \underbrace{\sum_{j=1}^{k} \hat{\alpha}_{g,j} \cdot \left(\bar{\hat{W}}_{:,j}^A - \bar{\hat{W}}_{:,j}^B\right)}_{\text{UV 偏差 } b_g}
\end{aligned}$$

第二项就是 UV 在两组分布不均时、通过基因 $g$ 的载荷 $\hat{\alpha}_g$ 泄漏进 logFC 的偏差。

### 5.3 负对照基因的选择（关键步骤）

如果 NC 基因里有真正的 DEG，RUV 假设 $\beta_c = 0$ 被违反，会导致校正过度（把真实生物信号也扣掉）。**采用三重过滤**：

#### 候选池构造

| 类别 | 优点 | 风险 | 建议 |
|---|---|---|---|
| Housekeeping | 经典的"不应变"基因 | 部分在癌症/激活态会变 | ✅ 用 |
| RPS/RPL（核糖体） | 高表达、技术噪声主导 | 增殖活跃细胞中会上调 | ⚠️ 用但检查 |
| MT 基因 | scRNA 早期工具箱常推荐 | scRNA 中 MT% 是应激标志，**不稳定** | ❌ **不推荐** |
| Top expressed | 高信噪比 | 高表达 ≠ 不 DEG | ⚠️ 候选池补充 |

```python
import re
import numpy as np


def build_candidate_nc_genes(gene_names, species="human"):
    """构造候选负对照基因池"""
    
    # 1) 文献 curated housekeeping（Eisenberg & Levanon 2013）
    eisenberg_hkg = {
        "ACTB", "B2M", "GAPDH", "HMBS", "HPRT1", "PGK1", "PPIA",
        "RPL13A", "RPLP0", "SDHA", "TBP", "TFRC", "UBC", "VCP",
        "YWHAZ", "C1orf43", "CHMP2A", "EMC7", "GPI", "PSMB2",
        "PSMB4", "RAB7A", "REEP5", "SNRPD3", "VPS29",
        "CLTC", "CSNK2B", "DDX5", "EEF1A1", "EIF4A2",
        "HNRNPK", "NONO", "RPL32", "RPS18", "SF3B1",
        "SRSF3", "SUMO1", "UBE2D2", "USP22", "XRN2",
    }
    
    # 2) 核糖体蛋白
    rp_genes = {g for g in gene_names if re.match(r'^RP[SL]\d', g)}
    
    # 3) 翻译延伸因子
    eef_genes = {g for g in gene_names if re.match(r'^EEF[12]', g)}
    
    # 4) 蛋白酶体亚基
    psm_genes = {g for g in gene_names if re.match(r'^PSM[A-D]\d', g)}
    
    candidate = (eisenberg_hkg & set(gene_names)) | rp_genes | eef_genes | psm_genes
    return np.array([g in candidate for g in gene_names])
```

#### 三重过滤

```python
from scipy import stats
from sklearn.covariance import MinCovDet


def select_clean_nc_genes(Y, group_labels, gene_names,
                          candidate_nc_mask, n_final=500,
                          p_threshold=0.3):
    """
    三重过滤：
    1. ANOVA F-test 排除有 DE 迹象的基因（p > 0.2）
    2. CV 稳定性评分
    3. 迭代稳健 Ŵ 估计 + 残差筛选
    """
    n_samples, n_genes = Y.shape
    groups = np.unique(group_labels)
    candidate_idx = np.where(candidate_nc_mask)[0]
    
    # ---- Filter 1: ANOVA F-test ----
    f_stats, p_values = [], []
    for g_idx in candidate_idx:
        group_data = [Y[group_labels == grp, g_idx] for grp in groups]
        f, p = stats.f_oneway(*group_data)
        f_stats.append(f); p_values.append(p)
    f_stats = np.array(f_stats); p_values = np.array(p_values)
    filter1_pass = p_values > p_threshold
    
    # ---- Filter 2: CV 评分 ----
    cvs = np.array([
        np.std(Y[:, candidate_idx[i]]) / (np.mean(Y[:, candidate_idx[i]]) + 1e-8)
        for i in range(len(candidate_idx))
    ])
    combined_score = (-np.log10(np.maximum(p_values, 1e-300)) + 
                      (cvs / cvs.median()))
    
    # ---- Filter 3: 迭代稳健估计 ----
    n_initial = min(2 * n_final, filter1_pass.sum())
    initial_order = np.argsort(combined_score)
    initial_order = [i for i in initial_order if filter1_pass[i]]
    current_genes = candidate_idx[initial_order[:n_initial]]
    
    for iteration in range(3):
        try:
            mcd = MinCovDet(random_state=42).fit(Y[:, current_genes])
            cov_robust = mcd.covariance_
            eigenvalues, eigenvectors = np.linalg.eigh(cov_robust)
            k = min(5, len(current_genes) - 1)
            W_hat_robust = eigenvectors[:, -k:]
        except Exception:
            Y_c = Y[:, current_genes] - Y[:, current_genes].mean(axis=0)
            U, S, Vt = np.linalg.svd(Y_c, full_matrices=False)
            k = min(5, len(current_genes) - 1)
            W_hat_robust = U[:, :k]
        
        # 残差过大的基因踢掉
        residuals = []
        for g_idx in current_genes:
            y_g = Y[:, g_idx]; y_g_c = y_g - y_g.mean()
            alpha_g = np.linalg.lstsq(W_hat_robust, y_g_c, rcond=None)[0]
            residuals.append(np.linalg.norm(y_g_c - W_hat_robust @ alpha_g))
        residuals = np.array(residuals)
        q75, q25 = np.percentile(residuals, [75, 25])
        upper_bound = q75 + 1.5 * (q75 - q25)
        current_genes = current_genes[residuals <= upper_bound]
        if len(current_genes) < n_final:
            break
    
    if len(current_genes) > n_final:
        final_scores = [combined_score[np.where(candidate_idx == g)[0][0]] 
                        if g in candidate_idx else np.inf 
                        for g in current_genes]
        final_order = np.argsort(final_scores)
        current_genes = current_genes[final_order[:n_final]]
    
    nc_mask = np.zeros(n_genes, dtype=bool)
    nc_mask[current_genes] = True
    return nc_mask, {
        "n_passed_filter1": int(filter1_pass.sum()),
        "n_final": int(nc_mask.sum()),
    }
```

### 5.4 PCA 版后校正（baseline）

```python
def correct_logfc_with_pca(Y, group_labels, gene_names, mast_degs,
                           neg_control_mask, k=5):
    """
    用 PCA 提取的 Ŵ 做后校正。
    适用：想快速看一眼 UV 偏差有多大。
    """
    n_samples, n_genes = Y.shape
    groups = np.unique(group_labels)
    A, B = groups[0], groups[1]
    mask_A = group_labels == A; mask_B = group_labels == B
    
    # ---- Step 1: 从 NC 基因估计 Ŵ ----
    Y_nc = Y[:, neg_control_mask]
    Y_nc_c = Y_nc - Y_nc.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Y_nc_c, full_matrices=False)
    W_hat = U[:, :k]
    
    # ---- Step 2: 对 DEG 回归到 Ŵ ----
    deg_idx = [list(gene_names).index(g) for g in mast_degs["gene"] if g in gene_names]
    Y_deg = Y[:, deg_idx]
    Y_deg_c = Y_deg - Y_deg.mean(axis=0, keepdims=True)
    alpha_hat = np.linalg.lstsq(W_hat, Y_deg_c, rcond=None)[0]  # (k, n_deg)
    
    # ---- Step 3: UV 偏差 ----
    W_diff = W_hat[mask_A].mean(axis=0) - W_hat[mask_B].mean(axis=0)
    b_g = alpha_hat.T @ W_diff
    
    mast_degs["log2FC_corrected"] = mast_degs["log2FC"] - b_g
    mast_degs["uv_bias"] = b_g
    mast_degs["uv_bias_abs"] = np.abs(b_g)
    return mast_degs, W_hat, alpha_hat
```

### 5.5 RUV-VAE 版后校正

用 VAE 的非线性 $\Delta_{total}$ 替代 PCA 的 $\hat{W}\hat{\alpha}$：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RUVVAE_log2(nn.Module):
    """log2 scale 的 RUV-VAE，输入输出与 MAST 对齐"""
    
    def __init__(self, n_genes, d_bio=32, k_unk=5):
        super().__init__()
        self.encoder_z = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.z_mu = nn.Linear(256, d_bio); self.z_logvar = nn.Linear(256, d_bio)
        
        self.encoder_w = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.w_mu = nn.Linear(256, k_unk); self.w_logvar = nn.Linear(256, k_unk)
        
        self.decoder_bio = nn.Sequential(nn.Linear(d_bio, 256), nn.GELU(),
                                         nn.Linear(256, n_genes))
        self.decoder_w = nn.Sequential(nn.Linear(k_unk, 256), nn.GELU(),
                                       nn.Linear(256, n_genes))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def forward(self, y, neg_control_mask):
        h_z = self.encoder_z(y)
        z_mu, z_logvar = self.z_mu(h_z), self.z_logvar(h_z)
        z = self.reparameterize(z_mu, z_logvar)
        
        h_w = self.encoder_w(y)
        w_mu, w_logvar = self.w_mu(h_w), self.w_logvar(h_w)
        w = self.reparameterize(w_mu, w_logvar)
        
        y_bio = self.decoder_bio(z)
        delta_lat = self.decoder_w(w)
        y_clean = y_bio - delta_lat
        
        losses = {
            'recon': F.mse_loss(y_clean, y),
            'kl_z': -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()),
            'kl_w': -0.5 * torch.mean(1 + w_logvar - w_mu.pow(2) - w_logvar.exp()),
            'nc_lat': F.mse_loss(delta_lat[:, neg_control_mask],
                                 y[:, neg_control_mask]),
        }
        return {'y_clean': y_clean, 'delta_lat': delta_lat,
                'w_mu': w_mu, 'z_mu': z_mu, 'losses': losses}


def train_ruv_vae(Y, neg_control_mask, n_epochs=100, batch_size=256,
                  lr=1e-3, lambda_nc=5.0, lambda_kl=1e-3):
    n_samples, n_genes = Y.shape
    model = RUVVAE_log2(n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Y_t = torch.FloatTensor(Y); nc_t = torch.BoolTensor(neg_control_mask)
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            out = model(Y_t[idx], nc_t)
            L = out['losses']
            loss = (L['recon'] + lambda_kl * L['kl_z'] + lambda_kl * L['kl_w']
                    + lambda_nc * L['nc_lat'])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


@torch.no_grad()
def correct_logfc_with_vae(model, Y, group_labels, gene_names,
                            mast_degs, neg_control_mask,
                            n_posterior_samples=100):
    """
    用 RUV-VAE 的 Δ_total 对 MAST log2FC 做后校正，并给出后验不确定性。
    """
    model.eval()
    Y_t = torch.FloatTensor(Y); nc_t = torch.BoolTensor(neg_control_mask)
    out = model(Y_t, nc_t)
    delta_total = out['delta_lat'].numpy()  # (n_samples, n_genes)
    
    deg_idx = [list(gene_names).index(g) for g in mast_degs["gene"]
               if g in gene_names]
    groups = np.unique(group_labels)
    mask_A = group_labels == groups[0]; mask_B = group_labels == groups[1]
    
    # 偏差 = Δ 在两组间的差
    delta_deg = delta_total[:, deg_idx]
    bias = delta_deg[mask_A].mean(axis=0) - delta_deg[mask_B].mean(axis=0)
    
    # 后验不确定性（VAE 独家）
    h_z = model.encoder_z(Y_t); z_mu = model.z_mu(h_z); z_logvar = model.z_logvar(h_z)
    h_w = model.encoder_w(Y_t); w_mu = model.w_mu(h_w); w_logvar = model.w_logvar(h_w)
    logFC_bias_samples = np.zeros((n_posterior_samples, len(deg_idx)))
    for s in range(n_posterior_samples):
        std = torch.exp(0.5 * w_logvar)
        w_sample = w_mu + std * torch.randn_like(std)
        delta_s = model.decoder_w(w_sample).numpy()
        logFC_bias_samples[s] = (delta_s[mask_A][:, deg_idx].mean(0) - 
                                  delta_s[mask_B][:, deg_idx].mean(0))
    
    mast_degs["log2FC_corrected"] = mast_degs["log2FC"] - bias
    mast_degs["uv_bias"] = bias
    mast_degs["log2FC_bias_ci_low"] = np.percentile(logFC_bias_samples, 2.5, axis=0)
    mast_degs["log2FC_bias_ci_high"] = np.percentile(logFC_bias_samples, 97.5, axis=0)
    return mast_degs, delta_total, logFC_bias_samples
```

### 5.6 PCA vs VAE 后处理对比

| | PCA 后校正 | RUV-VAE 后校正 |
|---|---|---|
| UV 表达能力 | 线性 | 非线性 |
| 后验不确定性 | 无 | **有**（后验采样） |
| 显式协变量支持 | 无 | 有（Hybrid 版） |
| 计算量 | 几秒 | 分钟~小时 |
| 适用场景 | 快速诊断 | 精细化校正 |

### 5.7 关键诊断

```python
def diagnose_post_correction(mast_degs, W_hat, w_mu, delta_total,
                            Y, gene_names, neg_control_mask, group_labels):
    """后校正的四个关键检查"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    groups = np.unique(group_labels)
    
    # Panel 1: 原始 vs 校正 logFC
    axes[0, 0].scatter(mast_degs["log2FC"], mast_degs["log2FC_corrected"],
                       alpha=0.4, s=10, c=mast_degs["uv_bias_abs"], cmap="Reds")
    lim = max(abs(mast_degs["log2FC"]).max(), 
              abs(mast_degs["log2FC_corrected"]).max()) * 1.1
    axes[0, 0].plot([-lim, lim], [-lim, lim], 'gray', linestyle='--')
    axes[0, 0].set_xlabel("MAST log2FC"); axes[0, 0].set_ylabel("Corrected")
    
    # Panel 2: Ŵ / w_i 在两组的分布
    W_use = w_mu if w_mu is not None else W_hat
    for j in range(min(W_use.shape[1], 3)):
        axes[0, 1].scatter(np.random.normal(0, 0.05, (group_labels == groups[0]).sum()),
                           W_use[group_labels == groups[0], j], alpha=0.4, s=8,
                           label=groups[0] if j == 0 else None)
        axes[0, 1].scatter(np.random.normal(1, 0.05, (group_labels == groups[1]).sum()),
                           W_use[group_labels == groups[1], j], alpha=0.4, s=8,
                           label=groups[1] if j == 0 else None)
    axes[0, 1].set_xticks([0, 1]); axes[0, 1].set_xticklabels(groups)
    axes[0, 1].set_title("UV latent per group\n(separated → confounding exists)")
    axes[0, 1].legend()
    
    # Panel 3: NC 基因 logFC 分布
    nc_idx = np.where(neg_control_mask)[0]
    nc_logFC = (Y[group_labels == groups[0]][:, nc_idx].mean(0) - 
                Y[group_labels == groups[1]][:, nc_idx].mean(0))
    axes[1, 0].hist(nc_logFC, bins=40, edgecolor='white')
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].set_title(f"NC logFC σ={nc_logFC.std():.3f}")
    
    # Panel 4: top 校正幅度
    top = mast_degs.nlargest(15, "uv_bias_abs")
    yp = range(len(top))
    axes[1, 1].hlines(yp, top["log2FC_corrected"], top["log2FC"],
                      color='gray', linewidth=2)
    axes[1, 1].scatter(top["log2FC"], yp, marker='o', label='Original', s=60, zorder=5)
    axes[1, 1].scatter(top["log2FC_corrected"], yp, marker='s', label='Corrected', s=60, zorder=5)
    axes[1, 1].set_yticks(yp); axes[1, 1].set_yticklabels(top["gene"], fontsize=8)
    axes[1, 1].axvline(0, color='gray', linestyle='--')
    axes[1, 1].legend()
    plt.tight_layout(); plt.show()
```

**Panel 3 最关键**：如果 NC 基因 logFC 的 $\sigma$ 很大（比如 > 0.3），说明 NC 选得不纯，整个 RUV 估计可信度下降。

---

## Part VI. RUV-VAE 作为 DEG 检测的主方法

### 6.1 动机

不再依赖 MAST，直接用 RUV-VAE 框架做 DEG：

```
y_ig = (group effect)_g           ← 你要检测的 DEG 信号
     + D_bio(z_i)_g              ← 其他生物学变化（细胞类型等）
     - D_W(w_i)_g                ← 非期望变异（潜变量嵌入）
     - (c_i W_cov)_g             ← 已知的协变量效应
     + ε_ig                      ← 噪声
```

### 6.2 模型架构

```python
class RUVVAE_DEG(nn.Module):
    """RUV-VAE 主方法做 DEG：group 作为显式协变量"""
    
    def __init__(self, n_genes, n_group=2, n_batch=0, d_bio=32, k_unk=5):
        super().__init__()
        self.n_genes = n_genes
        
        cov_blocks = {"group": n_group}
        if n_batch > 0:
            cov_blocks["batch"] = n_batch
        
        # 编码器
        self.encoder_z = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.z_mu = nn.Linear(256, d_bio); self.z_logvar = nn.Linear(256, d_bio)
        
        self.encoder_w = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.w_mu = nn.Linear(256, k_unk); self.w_logvar = nn.Linear(256, k_unk)
        
        # 解码器
        self.decoder_bio = nn.Sequential(nn.Linear(d_bio, 256), nn.GELU(),
                                         nn.Linear(256, n_genes))
        self.decoder_w = nn.Sequential(nn.Linear(k_unk, 256), nn.GELU(),
                                       nn.Linear(256, n_genes))
        # 显式协变量矩阵
        self.W_cov = nn.ParameterDict({
            name: nn.Parameter(torch.randn(dim, n_genes) * 0.01)
            for name, dim in cov_blocks.items()
        })
        self.bias = nn.Parameter(torch.zeros(n_genes))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def compute_delta_cov(self, c_dict):
        delta = 0
        for name, W in self.W_cov.items():
            delta = delta + c_dict[name] @ W
        return delta
    
    def forward(self, y, c_dict, neg_control_mask=None):
        h_z = self.encoder_z(y)
        z_mu, z_logvar = self.z_mu(h_z), self.z_logvar(h_z)
        z = self.reparameterize(z_mu, z_logvar)
        
        h_w = self.encoder_w(y)
        w_mu, w_logvar = self.w_mu(h_w), self.w_logvar(h_w)
        w = self.reparameterize(w_mu, w_logvar)
        
        y_bio = self.decoder_bio(z) + self.bias
        delta_lat = self.decoder_w(w)
        delta_cov = self.compute_delta_cov(c_dict)
        y_recon = y_bio - delta_lat - delta_cov
        
        losses = {
            'recon': F.mse_loss(y_recon, y),
            'kl_z': -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()),
            'kl_w': -0.5 * torch.mean(1 + w_logvar - w_mu.pow(2) - w_logvar.exp()),
        }
        if neg_control_mask is not None:
            delta_total = delta_lat + delta_cov
            losses['nc_total'] = F.mse_loss(
                delta_total[:, neg_control_mask],
                y[:, neg_control_mask]
            )
        return {'y_recon': y_recon, 'y_bio': y_bio,
                'delta_lat': delta_lat, 'delta_cov': delta_cov,
                'z_mu': z_mu, 'z_logvar': z_logvar,
                'w_mu': w_mu, 'w_logvar': w_logvar, 'losses': losses}
```

### 6.3 三种 DEG 计算方式

| logFC 类型 | 来源 | 含义 | 适用场景 |
|---|---|---|---|
| `logFC_linear` | $W_{cov}[\text{disease}] - W_{cov}[\text{control}]$ | 纯 group 效应 | 假设 disease 对所有细胞影响一致 |
| `logFC_bio` | $D_{bio}(z)$ 在两组的差异 | 纯生物学差异 | disease 通过改变细胞类型起作用 |
| `logFC_posterior_mean` | 两者之和 + 后验采样 | **综合效应** | **最终推荐** |

**关系**：

```
logFC_posterior = logFC_linear + logFC_bio
                  ↑                ↑
              |--- 显式 group ---|
                          |--- z 反映的细胞状态 ---|
```

### 6.4 DEG 计算（含后验贝叶斯 p-value）

```python
@torch.no_grad()
def compute_deg(model, Y, gene_names, groups_unique,
                neg_control_mask, batch_labels=None,
                n_posterior=200):
    """用 RUV-VAE 计算 DEG，返回贝叶斯 logFC + 可信区间 + p-value"""
    model.eval()
    n_samples, n_genes = Y.shape
    
    group_idx = np.array([groups_unique.index(g) for g in group_labels])
    c_dict = {"group": torch.FloatTensor(np.eye(len(groups_unique))[group_idx])}
    if batch_labels is not None:
        batches = sorted(np.unique(batch_labels))
        batch_idx = np.array([batches.index(b) for b in batch_labels])
        c_dict["batch"] = torch.FloatTensor(np.eye(len(batches))[batch_idx])
    
    Y_t = torch.FloatTensor(Y); nc_t = torch.BoolTensor(neg_control_mask)
    out = model(Y_t, c_dict, nc_t)
    
    # ---- 方法 A: 线性 logFC ----
    W_group = model.W_cov["group"].detach().numpy()
    logFC_linear = W_group[1] - W_group[0]
    
    # ---- 方法 B: D_bio(z) 差异 ----
    y_bio = out['y_bio'].numpy()
    mask_ctrl = group_labels == groups_unique[0]
    mask_disease = group_labels == groups_unique[1]
    logFC_bio = y_bio[mask_disease].mean(0) - y_bio[mask_ctrl].mean(0)
    
    # ---- 方法 C: 后验贝叶斯 ----
    z_mu = out['z_mu']; z_logvar = out['z_logvar']
    w_mu = out['w_mu']; w_logvar = out['w_logvar']
    
    logFC_posterior = np.zeros((n_posterior, n_genes))
    for s in range(n_posterior):
        z_sample = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
        w_sample = w_mu + torch.exp(0.5 * w_logvar) * torch.randn_like(w_mu)
        y_bio_s = (model.decoder_bio(z_sample) + model.bias).numpy()
        logFC_s = logFC_linear + (y_bio_s[mask_disease].mean(0) - 
                                   y_bio_s[mask_ctrl].mean(0))
        logFC_posterior[s] = logFC_s
    
    p_posterior = (logFC_posterior > 0).mean(0)
    p_value_bayes = 2 * np.minimum(p_posterior, 1 - p_posterior)
    
    return pd.DataFrame({
        'gene': gene_names,
        'logFC_linear': logFC_linear,
        'logFC_bio': logFC_bio,
        'logFC_posterior_mean': logFC_posterior.mean(0),
        'logFC_posterior_std': logFC_posterior.std(0),
        'logFC_ci_low': np.percentile(logFC_posterior, 2.5, axis=0),
        'logFC_ci_high': np.percentile(logFC_posterior, 97.5, axis=0),
        'p_value_bayes': p_value_bayes,
    }), {'W_group': W_group, 'y_bio': y_bio,
         'w_mu': w_mu.numpy(), 'z_mu': z_mu.numpy(),
         'posterior_samples': logFC_posterior}
```

### 6.5 RUV 公式的精确对应

| RUV 概念 | RUV-VAE 中的对应 |
|---|---|
| $M$ (assay→sample 映射) | identity（每样本一观测） |
| $X$ (设计矩阵) | c_dict 中的 group + batch |
| $\beta$ (基因 × 协变量效应) | $W_{cov}$ 矩阵 |
| $W$ (样本 × k 个 UV 方向) | $w_i$（潜变量）+ $c_i W_{cov}$ |
| $\alpha$ (基因 × k 个载荷) | $D_W(w)$ 解码器 |
| $\varepsilon$ | 重建残差 |

### 6.6 关键诊断

```python
def diagnose_deg_vae(model, vae_info, gene_names, group_labels,
                     deg_results, neg_control_mask, Y):
    """DEG-VAE 的六个关键检查"""
    import matplotlib.pyplot as plt
    W_group = vae_info['W_group']; w_mu = vae_info['w_mu']
    groups = np.unique(group_labels)
    mask_ctrl = group_labels == groups[0]; mask_disease = group_labels == groups[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: logFC_linear vs logFC_bio
    axes[0, 0].scatter(deg_results['logFC_linear'], deg_results['logFC_bio'],
                       alpha=0.3, s=10)
    axes[0, 0].axhline(0, color='gray', linestyle='--')
    axes[0, 0].axvline(0, color='gray', linestyle='--')
    lim = 3
    axes[0, 0].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.4)
    axes[0, 0].set_xlabel("logFC linear (W_cov)")
    axes[0, 0].set_ylabel("logFC bio (D_bio)")
    axes[0, 0].set_title("Linear vs nonlinear")
    
    # Panel 2: w_i 在两组分布
    for j in range(min(w_mu.shape[1], 3)):
        axes[0, 1].scatter(np.random.normal(0, 0.05, mask_ctrl.sum()),
                           w_mu[mask_ctrl, j], alpha=0.4, s=10,
                           label=groups[0] if j == 0 else None)
        axes[0, 1].scatter(np.random.normal(1, 0.05, mask_disease.sum()),
                           w_mu[mask_disease, j], alpha=0.4, s=10,
                           label=groups[1] if j == 0 else None)
    axes[0, 1].set_xticks([0, 1]); axes[0, 1].set_xticklabels(groups)
    axes[0, 1].set_title("w_i (UV) per group")
    axes[0, 1].legend()
    
    # Panel 3: NC 基因 logFC 分布
    nc_idx = np.where(neg_control_mask)[0]
    nc_logFC = (Y[mask_ctrl][:, nc_idx].mean(0) - Y[mask_disease][:, nc_idx].mean(0))
    axes[0, 2].hist(nc_logFC, bins=40, edgecolor='white')
    axes[0, 2].axvline(0, color='red', linestyle='--')
    axes[0, 2].set_title(f"NC genes σ={nc_logFC.std():.3f}")
    
    # Panel 4: 火山图
    axes[1, 0].scatter(deg_results['logFC_posterior_mean'],
                       -np.log10(deg_results['p_value_bayes']),
                       alpha=0.4, s=8)
    axes[1, 0].axhline(-np.log10(0.05), color='red', linestyle='--')
    axes[1, 0].set_xlabel("logFC posterior mean")
    axes[1, 0].set_ylabel("-log10(Bayes p)")
    axes[1, 0].set_title("Volcano")
    
    # Panel 5: top DEG 后验分布
    top = deg_results.nsmallest(5, 'p_value_bayes')
    for idx, (_, row) in enumerate(top.iterrows()):
        g_idx = list(gene_names).index(row['gene'])
        axes[1, 1].hist(vae_info['posterior_samples'][:, g_idx],
                        bins=40, alpha=0.5, label=row['gene'])
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].legend(fontsize=8)
    
    # Panel 6: 95% CI 宽度
    ci_width = deg_results['logFC_ci_high'] - deg_results['logFC_ci_low']
    axes[1, 2].hist(ci_width, bins=40, edgecolor='white')
    axes[1, 2].set_title(f"Posterior CI width\nmean={ci_width.mean():.3f}")
    plt.tight_layout(); plt.show()
```

**Panel 1 最关键**：
- 对角线上的基因：disease 效应是线性的、一致的
- 偏离对角线的基因：disease 效应是非线性的（比如某细胞类型在 disease 中消失）

### 6.7 相比 MAST 的优势

| | MAST | RUV-VAE DEG |
|---|---|---|
| 数据建模 | hurdle on counts | 深度生成模型 |
| UV 处理 | cngeneson 协变量（标量） | 潜变量 $w_i$（向量）+ 显式 W_cov |
| 不确定性 | Wald p-value | 后验分布 + 可信区间 |
| 非线性 UV | ❌ | ✅ |
| 可扩展性 | 千级细胞 | 百万级（mini-batch） |
| 解释性 | 系数 + 显著性 | $W_{cov}$ 矩阵 + 潜空间可视化 |

### 6.8 完整 pipeline

```python
def full_deg_pipeline(Y, group_labels, gene_names, neg_control_mask,
                      batch_labels=None, output_prefix="deg"):
    model, groups_unique = train_deg_vae(
        Y, group_labels, neg_control_mask, batch_labels,
        n_epochs=100, lambda_nc=5.0
    )
    deg_results, vae_info = compute_deg(
        model, Y, gene_names, groups_unique,
        neg_control_mask, batch_labels
    )
    from statsmodels.stats.multitest import multipletests
    deg_results['q_value'] = multipletests(
        deg_results['p_value_bayes'], method='fdr_bh'
    )[1]
    deg_results.to_csv(f"{output_prefix}_results.csv", index=False)
    
    nc_logFC = (Y[group_labels == groups_unique[0]][:, neg_control_mask].mean(0) - 
                Y[group_labels == groups_unique[1]][:, neg_control_mask].mean(0))
    diagnose_deg_vae(model, vae_info, gene_names, group_labels,
                     deg_results, neg_control_mask, Y)
    return deg_results, model, vae_info
```

---

## 附：参考文献

- Molania, R. et al. **Removing unwanted variation from large-scale RNA sequencing data with PRPS.** *Nature Biotechnology* 41, 82–95 (2023).
- Lopez, R. et al. **Deep generative modeling for single-cell transcriptomics.** *Nature Methods* 15, 1053–1058 (2018). (scVI)
- Xu, C. et al. **Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models.** *ICML* (2021). (scANVI)
- Eraslan, G. et al. **Single-cell RNA-seq denoising using a deep count autoencoder.** *Nature Communications* 10, 390 (2019). (DCA)
- Korsunsky, I. et al. **Fast, sensitive and accurate integration of single-cell data with Harmony.** *Nature Methods* 16, 1289–1296 (2019). (Harmony)
- Risso, D. et al. **Normalization of RNA-seq data using factor analysis of control genes or samples.** *Nature Biotechnology* 32, 896–902 (2014). (RUV-III 原始)
- Gagnon-Bartsch, J.A. et al. **Using control genes to correct for unwanted variation in microarray data.** *Biostatistics* 13, 539–552 (2012). (RUV-II)
- Finak, G. et al. **MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data.** *Genome Biology* 16, 278 (2015). (MAST 原文)
- Eisenberg, E. & Levanon, E.Y. **Human housekeeping genes, revisited.** *Trends in Genetics* 29, 569–574 (2013). (Housekeeping 基因集)
- Jacob, L. et al. **Accurate quantification of differentiation in 3D neural organoids through scVI.** *bioRxiv* (2020). (scVI DEG 方法)

---

*笔记由 Claude 协助整理，请自行核对原始论文细节。*
*Part V / Part VI 添加于 2026-07-24，对应 RUV-VAE 用作 MAST 后处理与 DEG 主方法的探索。*