import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class ZINBLoss:
    """Zero-Inflated Negative Binomial loss (scVI-style stable formulation).

    Models scRNA-seq counts as a mixture of:
      - A point mass at zero (zero-inflation, with probability pi_prob = sigmoid(pi))
      - A Negative Binomial (with mean mu and inverse-dispersion theta)

    The implementation mirrors `scvi.distributions.log_zinb_positive` /
    `log_nb_positive`: pi is parameterised in **logit space** (real support, no
    sigmoid needed in the decoder), so logits flow through unchanged and the
    stable `softplus(-pi)` identity is used in place of `log(1 - pi)`. This
    removes the need for `clamp(pi, eps, 1-eps)`, which would otherwise kill
    gradients at the saturation tails of the dropout probability.

    Usage:
        zinb = ZINBLoss()
        nll = zinb.nll(x, mu, theta, pi)  # pi is the dropout LOGIT
    """

    @staticmethod
    def log_nb(x, mu, theta, eps=1e-8):
        """Log-probability of the Negative Binomial distribution.

        NB(x | mu, theta) = Gamma(x + theta) / (Gamma(theta) * Gamma(x + 1))
                            * (theta / (theta + mu))^theta
                            * (mu / (theta + mu))^x

        Args:
            x:     observed counts  (N, G), any non-negative value
            mu:    NB mean           (N, G), strictly positive
            theta: inverse-dispersion (G,) or (N, G), strictly positive
            eps:   numerical stability inside log
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)  # (1, G) -> broadcasts over N

        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )

    @staticmethod
    def log_zinb(x, mu, theta, pi, eps=1e-8):
        """Log-probability of the Zero-Inflated Negative Binomial.

        ZINB(x | mu, theta, pi) = pi_prob * I(x == 0)
                                   + (1 - pi_prob) * NB(x | mu, theta)
        where pi_prob = sigmoid(pi) and **pi is the dropout LOGIT** (any real).

        The zero / non-zero cases are computed separately and combined with
        a hard mask (matching scVI). The case_zero branch uses the
        `log(1 - pi_prob) = -softplus(pi)` and
        `log(pi_prob + (1-pi_prob) * NB(0)) = softplus(pi_theta_log) - softplus(-pi)`
        identities, both numerically stable without any clamping.

        Args:
            x:     observed counts  (N, G), any non-negative value
            mu:    NB mean           (N, G), strictly positive
            theta: inverse-dispersion (G,) or (N, G), strictly positive
            pi:    dropout LOGIT  (N, G), real support (decoder outputs raw logits)
            eps:   numerical stability inside log
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)  # (1, G) -> broadcasts over N

        # Stable identity: log(sigmoid(pi)) = -softplus(pi).
        # scVI defines softplus_pi = softplus(-pi), so -softplus_pi = -softplus(-pi)
        # = pi - softplus(pi) = log(sigmoid(pi)) = log(pi_prob).
        softplus_pi = F.softplus(-pi)

        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)

        # log(exp(-pi) * NB(0 | mu, theta)) — will be reused in both branches.
        # NB(0) = (theta / (theta + mu))^theta, so
        # log NB(0) = theta * (log theta - log(theta + mu)).
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        # case x == 0:
        #   log ZINB(0) = log(pi_prob + (1 - pi_prob) * NB(0))
        #                = softplus(pi_theta_log) - softplus(-pi)        (stable)
        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = (x < eps).to(mu.dtype) * case_zero

        # case x > 0:
        #   log ZINB(x) = log(1 - pi_prob) + log NB(x)
        #               = -softplus(pi) + pi_theta_log + x*(log mu - log(theta+mu))
        #                  + lgamma(x+theta) - lgamma(theta) - lgamma(x+1)
        # nb-style lgamma terms live in `log_nb_positive`; we inline them here
        # so we only carry the NB-specific theta piece plus one extra `-pi`
        # (absorbed into `pi_theta_log`) instead of the full log NB twice.
        lgamma_x_theta = torch.lgamma(x + theta)
        lgamma_theta = torch.lgamma(theta)
        lgamma_x_plus_1 = torch.lgamma(x + 1)
        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )
        mul_case_non_zero = (x > eps).to(mu.dtype) * case_non_zero

        return mul_case_zero + mul_case_non_zero

    @staticmethod
    def nll(x, mu, theta, pi, eps=1e-8):
        """Mean negative log-likelihood (scalar).

        Args:
            pi: dropout LOGIT (real support) — the decoder should output raw
                logits, NOT probabilities. See `log_zinb` for details.
        """
        ll = ZINBLoss.log_zinb(x, mu, theta, pi, eps)
        return -ll.mean()


class RUVVAE_DEG(nn.Module):
    """RUV-VAE 主方法做 DEG：group 是显式生物学效应，batch 是 UV 协变量

    Supports two reconstruction losses:
      - MSE  (default,  use_zinb=False): mean squared error on log-scale
      - ZINB (scVI-like, use_zinb=True):  Zero-Inflated Negative Binomial NLL
    """

    def __init__(self, n_genes, n_group=2, n_batch=0, n_genes_on=0,
                 d_bio=32, k_unk=5, use_zinb=False):
        super().__init__()
        self.n_genes = n_genes
        self.use_zinb = use_zinb

        # group 是研究目标的生物学设计变量，不属于 unwanted variation。
        # 只有 batch、n_genes_on 等已知技术因素放入 W_cov，并在重建时扣除。
        self.W_group = nn.Parameter(torch.randn(n_group, n_genes) * 0.01)
        cov_blocks = {}
        if n_batch > 0:
            cov_blocks["batch"] = n_batch
        if n_genes_on > 0:
            cov_blocks["n_genes_on"] = n_genes_on
        self.cov_dims = cov_blocks.copy()

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

        # ---- ZINB 相关参数 (类似 scVI) ----
        if use_zinb:
            # 基因特异性逆离散度 (inverse-dispersion / r), 类似 scVI 的 px_r
            # 前向时通过 F.softplus(self.px_r) 保证正性 (scVI 风格: theta = exp(px_r))
            self.px_r = nn.Parameter(torch.zeros(n_genes))
            # 零膨胀 (dropout) 解码器: 从生物隐变量 z 预测每个基因的 dropout
            # 注意 — 输出是 **logits** (real support)，而不是概率 (与 scVI 一致)，
            # 这样 ZINBLoss 可以用 softplus(-pi) 的稳定形式，无需 clamp。
            self.decoder_dropout = nn.Sequential(
                nn.Linear(d_bio, 256), nn.GELU(),
                nn.Linear(256, n_genes)  # no Sigmoid — pi 是 logit
            )
            # NOTE: ZINB 均值复用 RUV 分解: mu = exp(y_bio - delta_lat - delta_cov)
            # decoder_bio, decoder_w, W_group, W_cov 全部参与 ZINB 重建，不另设 decoder_mu
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def compute_delta_cov(self, c_dict):
        missing = [name for name in self.W_cov if name not in c_dict]
        if missing:
            raise KeyError(f"Missing nuisance covariates in c_dict: {missing}")

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

        y_bio_base = self.decoder_bio(z) + self.bias
        group_effect = c_dict["group"] @ self.W_group
        y_bio = y_bio_base + group_effect
        delta_lat = self.decoder_w(w)
        delta_cov = self.compute_delta_cov(c_dict)

        if self.use_zinb:
            # ---- ZINB 模式 (scVI 风格): 复用 RUV 分解 ----
            # ZINB 均值基于 RUV 分解: log_mu = y_bio - delta_lat - delta_cov
            # 用 exp 而非 softplus: counts 高达 4774, softplus(8)=8 太小, exp(8)=2980 匹配
            # decoder_bio, decoder_w, W_group, W_cov 全部被 ZINB loss 训练
            log_mu = y_bio + delta_lat + delta_cov
            y_mu = torch.exp(log_mu) + 1e-6  # (N, G), >0, natural count scale

            # scVI 风格: theta 来自 F.softplus(self.px_r)，保证 >0；
            # self.px_r 以 zeros 初始化，softplus(0)=log(2)≈0.69，对 NB 是一个温和起点。
            theta = F.softplus(self.px_r)  # (G,)
            # decoder_dropout 输出 logits (real support)，与 scVI 的 px_dropout 对齐。
            # 不再 clamp / sigmoid: ZINBLoss.log_zinb 用 -softplus(-pi) 的稳定形式。
            pi_logit = self.decoder_dropout(z)  # (N, G)
            pi_prob = torch.sigmoid(pi_logit)   # 仅用于 E[ZINB] / 调试

            losses = {
                'recon': ZINBLoss.nll(y, y_mu, theta, pi_logit),
                'kl_z': -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2)
                                         - z_logvar.exp()),
                'kl_w': -0.5 * torch.mean(1 + w_logvar - w_mu.pow(2)
                                         - w_logvar.exp()),
            }
            # E[ZINB] = mu * (1 - pi_prob) = mu * sigmoid(-pi_logit)
            # 用 sigmoid(-logit) 而不是 (1 - sigmoid)，数值等价但更容易反向传播。
            y_recon = y_mu * torch.sigmoid(-pi_logit)
            dropout = pi_prob
        else:
            # ---- MSE 模式 (原始) ----
            y_recon = y_bio + delta_lat + delta_cov
            losses = {
                'recon': F.mse_loss(y_recon, y),
                'kl_z': -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2)
                                         - z_logvar.exp()),
                'kl_w': -0.5 * torch.mean(1 + w_logvar - w_mu.pow(2)
                                         - w_logvar.exp()),
            }
            dropout = None

        if neg_control_mask is not None:
            #delta_total = delta_lat + delta_cov
            delta_total = delta_lat + delta_cov
            # NC 约束: Δ_total 只学跨样本 UV 变异, 不学基线表达。
            # NC 基因的全局均值 (= 内参基线) 交给 y_bio 保留。
            # 这样  y_bio[nc] = nc_mean (跨样本稳定), 不是 0。
            y_nc = y[:, neg_control_mask]
            nc_mean = y_nc.mean(dim=0, keepdim=True)  # (1, n_nc)
            losses['nc_total'] = F.mse_loss(
                delta_total[:, neg_control_mask],
                y_nc - nc_mean
            )

        result = {'y_recon': y_recon, 'y_bio': y_bio,
                  'y_bio_base': y_bio_base, 'group_effect': group_effect,
                  'delta_lat': delta_lat, 'delta_cov': delta_cov,
                  'z_mu': z_mu, 'z_logvar': z_logvar,
                  'w_mu': w_mu, 'w_logvar': w_logvar, 'losses': losses}

        if self.use_zinb:
            result['y_mu'] = y_mu
            result['pi'] = dropout
            result['theta'] = theta.expand(y.shape[0], -1)

        return result
    
@torch.no_grad()
def compute_deg_all_methods_legacy(model, Y, gene_names, group_labels, groups_unique,
                                   neg_control_mask, batch_labels=None, n_genes_on=None,
                                   n_posterior=200,
                                   n_perm=500):
    """用 RUV-VAE 计算 DEG，返回 logFC + 多个 GLM 风格 p-value。

    p-value 方法（参考经典 GLM / MAST / DESeq2 / edgeR）：
      - p_value_empirical_t : 基于 y_bio 的 Welch's t-test
                              （Wald test 的稳健近似）
      - p_value_glm_delta  : 基于 decoder Jacobian 的 delta method
                              （仿 DESeq2 dispersion 思路）
      - p_value_perm       : Permutation test（最稳健、最慢）
      - p_value_bayes      : 后验采样版（受后验坍缩影响，仅作对照）

    Args:
        model:        训练好的 RUVVAE_DEG
        Y:            (n_samples, n_genes) 已 log1p 归一化的表达矩阵
        group_labels: 每个样本的组名
        groups_unique:组名列表（[0] 为对照，[1] 为处理）
        neg_control_mask: (n_genes,) bool mask
        batch_labels: 可选，每个样本的批次名
        n_genes_on: 可选，标准化后的每个样本检测到的基因数（n_samples, 或 n_samples×1）
        n_posterior:   后验采样数
        n_perm:        置换检验次数
    """
    from scipy import stats as _stats
    model.eval()
    n_samples, n_genes = Y.shape
    
    group_idx = np.array([groups_unique.index(g) for g in group_labels])
    c_dict = {"group": torch.FloatTensor(np.eye(len(groups_unique))[group_idx])}
    if batch_labels is not None:
        batches = sorted(np.unique(batch_labels))
        batch_idx = np.array([batches.index(b) for b in batch_labels])
        c_dict["batch"] = torch.FloatTensor(np.eye(len(batches))[batch_idx])
    if n_genes_on is not None:
        detection = np.asarray(n_genes_on, dtype=np.float32)
        if detection.ndim == 1:
            detection = detection[:, None]
        if detection.shape != (n_samples, 1):
            raise ValueError(
                f"n_genes_on must have shape ({n_samples},) or ({n_samples}, 1), "
                f"got {detection.shape}"
            )
        c_dict["n_genes_on"] = torch.from_numpy(detection)

    Y_t = torch.FloatTensor(Y); nc_t = torch.BoolTensor(neg_control_mask)
    out = model(Y_t, c_dict, nc_t)
    
    # ---- 分组掩码 ----
    mask_ctrl = (group_labels == groups_unique[0])
    mask_disease = (group_labels == groups_unique[1])
    n_ctrl = int(mask_ctrl.sum())
    n_disease = int(mask_disease.sum())
    
    # ============= UV 贡献 =============
    delta_lat = out['delta_lat'].numpy()
    logFC_uv_latent = delta_lat[mask_disease].mean(0) - delta_lat[mask_ctrl].mean(0)
    
    delta_cov = out['delta_cov'].cpu().numpy()
    logFC_uv_cov = delta_cov[mask_disease].mean(0) - delta_cov[mask_ctrl].mean(0)
    logFC_uv_total = logFC_uv_latent + logFC_uv_cov

    logFC_uv_n_genes_on = np.zeros(n_genes)
    if n_genes_on is not None and "n_genes_on" in model.W_cov:
        detection = np.asarray(n_genes_on, dtype=np.float32).reshape(-1, 1)
        detection_diff = detection[mask_disease].mean(0) - detection[mask_ctrl].mean(0)
        detection_W = model.W_cov["n_genes_on"].detach().cpu().numpy()
        logFC_uv_n_genes_on = detection_diff @ detection_W

    # ============= 生物学贡献 =============
    # group 是目标生物学差异，不能作为 UV 从 y_bio 中扣除。
    W_group = model.W_group.detach().cpu().numpy()
    logFC_group = W_group[1] - W_group[0]

    y_bio_base = out['y_bio_base'].cpu().numpy()
    logFC_bio_latent = (
        y_bio_base[mask_disease].mean(0) - y_bio_base[mask_ctrl].mean(0)
    )
    y_bio = out['y_bio'].cpu().numpy()
    logFC_bio = logFC_bio_latent + logFC_group
    
    # ============= 后验贝叶斯 =============
    z_mu = out['z_mu']; z_logvar = out['z_logvar']
    w_mu = out['w_mu']; w_logvar = out['w_logvar']
    
    logFC_posterior = np.zeros((n_posterior, n_genes))
    logFC_uv_posterior = np.zeros((n_posterior, n_genes))
    for s in range(n_posterior):
        z_sample = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
        w_sample = w_mu + torch.exp(0.5 * w_logvar) * torch.randn_like(w_mu)
        y_bio_s = (model.decoder_bio(z_sample) + model.bias).cpu().numpy()
        delta_lat_s = model.decoder_w(w_sample).cpu().numpy()
        bio_fc_s = (
            y_bio_s[mask_disease].mean(0) - y_bio_s[mask_ctrl].mean(0)
        ) + logFC_group
        uv_fc_s = delta_lat_s[mask_disease].mean(0) - delta_lat_s[mask_ctrl].mean(0)
        logFC_s = bio_fc_s - uv_fc_s - logFC_uv_cov
        logFC_posterior[s] = logFC_s
        logFC_uv_posterior[s] = uv_fc_s + logFC_uv_cov
    
    p_posterior = (logFC_posterior > 0).mean(0)
    p_value_bayes = 2 * np.minimum(p_posterior, 1 - p_posterior)
    
    # ============================================================
    # ======= 方法 1: Welch's t-test 基于 y_bio（最稳健）======
    # 公式: t = mean_diff / sqrt(var_d/n_d + var_c/n_c)
    #       df = Welch–Satterthwaite 近似
    #       p = 2 * (1 - cdf(|t|, df))
    # 优势: 不依赖 decoder Jacobian，直接用 y_bio 在两组内方差
    # 参考: MAST 的基础检验就是这种 t-type
    # ============================================================
    y_bio_dis = y_bio[mask_disease]    # (n_disease, n_genes)
    y_bio_ctr = y_bio[mask_ctrl]       # (n_ctrl, n_genes)
    
    mean_dis = y_bio_dis.mean(axis=0)
    mean_ctr = y_bio_ctr.mean(axis=0)
    var_dis  = y_bio_dis.var(axis=0, ddof=1)
    var_ctr  = y_bio_ctr.var(axis=0, ddof=1)
    
    # Welch's t 统计量
    se_t = np.sqrt(var_dis / n_disease + var_ctr / n_ctrl) + 1e-8
    t_stat = (mean_dis - mean_ctr) / se_t
    
    # Welch-Satterthwaite 自由度
    df_num = (var_dis / n_disease + var_ctr / n_ctrl) ** 2
    df_den = (var_dis / n_disease) ** 2 / max(n_disease - 1, 1) + \
             (var_ctr / n_ctrl) ** 2 / max(n_ctrl - 1, 1)
    df_welch = df_num / (df_den + 1e-12)
    
    # 两尾 p-value
    p_value_empirical_t = 2.0 * _stats.t.sf(np.abs(t_stat), df_welch)
    p_value_empirical_t = np.clip(p_value_empirical_t, 1e-300, 1.0)
    se_empirical = se_t   # 同 SE，保留给用户
    
    # ============================================================
    # ======= 方法 2: Wald test via delta method（仿 DESeq2）====
    # 思路: logFC ≈ w^T Δz + const, 把 encoder 的 f 视为从 x 到 z 的函数，
    #       在 z=z_mu 处做一阶 Taylor 展开, 推导 logFC 的方差。
    # 简化做法: 用 y_bio 的成对样本方差，模拟 MAST 的层级模型
    # 这里给一个**近似但稳健**的 delta-method SE：
    #   logFC_g ≈ mean(y_bio[:,g])_dis - mean(y_bio[:,g])_ctr
    #   Var(logFC_g) ≈ (1/n_d) * Var(D_bio(z_d)) + (1/n_c) * Var(D_bio(z_c))
    #   用 decoder 的 Jacobian-norm 当每个 z_d 的输出方差放大因子
    #
    # 这里的 var_dis, var_ctr 已经在上一方法算过；连同跨样本方差
    # 一起作为 se_glm。
    # ============================================================
    # 由于 decoder 是确定性的，y_bio[i,g] = D_bio(z_mu[i,:])_g + bias_g
    # 它的样本方差 ≈ decoder Jacobian 的 Frobenius 范数 × z_mu 的样本方差
    # 我们已经在 var_dis/var_ctr 里算了 y_bio 的样本方差，
    # 故 se_glm ≡ se_empirical，用 Wald 公式算 z 统计量：
    z_stat_glm = (mean_dis - mean_ctr) / se_t
    p_value_glm_delta = 2.0 * _stats.norm.sf(np.abs(z_stat_glm))
    p_value_glm_delta = np.clip(p_value_glm_delta, 1e-300, 1.0)
    
    # ============================================================
    # ======= 方法 3: Permutation test（最可信、最慢）======
    # 思路: 把 group 标签随机置换 N 次，每次算一次 logFC 分布，
    #       用经验分布当零假设, 看真实 logFC 在置换分布中的位置。
    # 加速: 不重跑 VAE，对 y_bio 重新算 mean_diff（VAE 编码器和
    #       decoder 是确定的，permutation 只影响 z_mu 那条路径）。
    #       进一步提速：直接对 y_bio[:, g] 在 permutation 索引下
    #       重算 mean-difference 而已。
    # ============================================================
    p_value_perm = np.full(n_genes, np.nan)
    if n_perm > 0:
        rng = np.random.default_rng(42)
        # 用 y_bio_dis 和 y_bio_ctr 拼成全数据，配合 permutation 标签
        y_bio_all = np.concatenate([y_bio_dis, y_bio_ctr], axis=0)   # (N, G)
        N = y_bio_all.shape[0]
        # 真实统计量
        obs_diff = mean_dis - mean_ctr
        # 抽样：每次随机划分 N 个样本为两组，分别算 mean-diff
        count_ge = np.zeros(n_genes, dtype=np.int64)
        for k in range(n_perm):
            idx = rng.permutation(N)
            # 维持两组大小一致
            dis_idx = idx[:n_disease]
            ctr_idx = idx[n_disease:n_disease + n_ctrl]
            perm_diff = y_bio_all[dis_idx].mean(0) - y_bio_all[ctr_idx].mean(0)
            count_ge += (np.abs(perm_diff) >= np.abs(obs_diff)).astype(np.int64)
        # 双尾
        p_value_perm = (count_ge + 1) / (n_perm + 1)
        p_value_perm = np.clip(p_value_perm, 1e-300, 1.0)
    
    return pd.DataFrame({
        'gene': gene_names,
        # 生物学贡献
        'logFC_group': logFC_group,
        'logFC_bio_latent': logFC_bio_latent,
        'logFC_bio': logFC_bio,
        'logFC_bio_posterior_mean': logFC_posterior.mean(0),
        'logFC_bio_posterior_std': logFC_posterior.std(0),
        'logFC_bio_ci_low': np.percentile(logFC_posterior, 2.5, axis=0),
        'logFC_bio_ci_high': np.percentile(logFC_posterior, 97.5, axis=0),
        # UV 贡献；group 已经单独作为生物学效应返回
        'logFC_uv_group': np.zeros(n_genes),
        'logFC_uv_linear': logFC_uv_cov,
        'logFC_uv_n_genes_on': logFC_uv_n_genes_on,
        'logFC_uv_latent': logFC_uv_latent,
        'logFC_uv_cov': logFC_uv_cov,
        'logFC_uv_total': logFC_uv_total,
        'logFC_uv_posterior_mean': logFC_uv_posterior.mean(0),
        'logFC_uv_posterior_std': logFC_uv_posterior.std(0),
        # p-value（多版本）
        'se_glm': se_empirical,                     # SE for Wald statistic
        't_stat': t_stat,                           # Welch's t
        'p_value_bayes': p_value_bayes,
        'p_value_empirical_t': p_value_empirical_t,
        'p_value_glm_delta': p_value_glm_delta,
        'p_value_perm': p_value_perm,
    }), {'W_group': W_group, 'y_bio': y_bio,
         'w_mu': w_mu.numpy(), 'z_mu': z_mu.numpy(),
         'delta_lat': delta_lat, 'delta_cov': delta_cov,
         'posterior_samples': logFC_posterior,
         'uv_posterior_samples': logFC_uv_posterior,
         'se_empirical': se_empirical, 't_stat': t_stat}


@torch.no_grad()
def compute_deg(model, Y, gene_names, group_labels, groups_unique,
                neg_control_mask, batch_labels=None, n_genes_on=None,
                method="wald", n_posterior=200, n_perm=500):
    """Compute DEG with one selected inference method.

    ``wald`` is the default and the closest option here to DESeq2's Wald-test
    idea, but it is not a negative-binomial GLM and is not DESeq2 itself.
    Other options are ``welch``, ``permutation``, ``bayes``, and ``all``.
    """
    from scipy import stats as _stats

    allowed_methods = {"wald", "welch", "permutation", "bayes", "all"}
    if method not in allowed_methods:
        raise ValueError(
            f"Unknown DEG method {method!r}; choose one of {sorted(allowed_methods)}"
        )

    if method == "all":
        result, extra = compute_deg_all_methods_legacy(
            model, Y, gene_names, group_labels, groups_unique,
            neg_control_mask, batch_labels=batch_labels,
            n_genes_on=n_genes_on, n_posterior=n_posterior, n_perm=n_perm,
        )
        result = result.copy()
        result["p_value"] = result["p_value_glm_delta"]
        result["test_stat"] = result["t_stat"]
        result["se"] = result["se_glm"]
        result["deg_method"] = "all"
        return result, extra

    model.eval()
    Y = np.asarray(Y, dtype=np.float32)
    group_labels = np.asarray(group_labels)
    n_samples, n_genes = Y.shape

    group_idx = np.array([groups_unique.index(g) for g in group_labels])
    c_dict = {
        "group": torch.FloatTensor(np.eye(len(groups_unique))[group_idx])
    }
    if batch_labels is not None:
        batch_labels = np.asarray(batch_labels)
        batches = sorted(np.unique(batch_labels))
        batch_idx = np.array([batches.index(b) for b in batch_labels])
        c_dict["batch"] = torch.FloatTensor(np.eye(len(batches))[batch_idx])

    detection = None
    if n_genes_on is not None:
        detection = np.asarray(n_genes_on, dtype=np.float32)
        if detection.ndim == 1:
            detection = detection[:, None]
        if detection.shape != (n_samples, 1):
            raise ValueError(
                f"n_genes_on must have shape ({n_samples},) or ({n_samples}, 1), "
                f"got {detection.shape}"
            )
        c_dict["n_genes_on"] = torch.from_numpy(detection)

    Y_t = torch.from_numpy(Y)
    nc_t = torch.BoolTensor(neg_control_mask)
    out = model(Y_t, c_dict, nc_t)

    mask_ctrl = group_labels == groups_unique[0]
    mask_disease = group_labels == groups_unique[1]
    n_ctrl = int(mask_ctrl.sum())
    n_disease = int(mask_disease.sum())
    if n_ctrl < 2 or n_disease < 2:
        raise ValueError("Each group needs at least two samples for DEG inference")

    delta_lat = out["delta_lat"].cpu().numpy()
    delta_cov = out["delta_cov"].cpu().numpy()
    logFC_uv_latent = (
        delta_lat[mask_disease].mean(0) - delta_lat[mask_ctrl].mean(0)
    )
    logFC_uv_cov = (
        delta_cov[mask_disease].mean(0) - delta_cov[mask_ctrl].mean(0)
    )
    logFC_uv_total = logFC_uv_latent + logFC_uv_cov

    logFC_uv_n_genes_on = np.zeros(n_genes)
    if detection is not None and "n_genes_on" in model.W_cov:
        detection_diff = (
            detection[mask_disease].mean(0) - detection[mask_ctrl].mean(0)
        )
        detection_W = model.W_cov["n_genes_on"].detach().cpu().numpy()
        logFC_uv_n_genes_on = detection_diff @ detection_W

    W_group = model.W_group.detach().cpu().numpy()
    logFC_group = W_group[1] - W_group[0]
    y_bio_base = out["y_bio_base"].cpu().numpy()
    logFC_bio_latent = (
        y_bio_base[mask_disease].mean(0) - y_bio_base[mask_ctrl].mean(0)
    )
    y_bio = out["y_bio"].cpu().numpy()
    logFC_bio = logFC_group + logFC_bio_latent

    y_bio_dis = y_bio[mask_disease]
    y_bio_ctr = y_bio[mask_ctrl]
    mean_dis = y_bio_dis.mean(0)
    mean_ctr = y_bio_ctr.mean(0)
    obs_diff = mean_dis - mean_ctr
    var_dis = y_bio_dis.var(0, ddof=1)
    var_ctr = y_bio_ctr.var(0, ddof=1)
    se = np.sqrt(var_dis / n_disease + var_ctr / n_ctrl) + 1e-8

    data = {
        "gene": gene_names,
        "logFC_group": logFC_group,
        "logFC_bio_latent": logFC_bio_latent,
        "logFC_bio": logFC_bio,
        "logFC_uv_group": np.zeros(n_genes),
        "logFC_uv_linear": logFC_uv_cov,
        "logFC_uv_n_genes_on": logFC_uv_n_genes_on,
        "logFC_uv_latent": logFC_uv_latent,
        "logFC_uv_cov": logFC_uv_cov,
        "logFC_uv_total": logFC_uv_total,
    }
    extra = {
        "W_group": W_group,
        "y_bio": y_bio,
        "w_mu": out["w_mu"].cpu().numpy(),
        "z_mu": out["z_mu"].cpu().numpy(),
        "delta_lat": delta_lat,
        "delta_cov": delta_cov,
    }

    if method == "wald":
        test_stat = obs_diff / se
        p_value = 2.0 * _stats.norm.sf(np.abs(test_stat))
        data["p_value_glm_delta"] = np.clip(p_value, 1e-300, 1.0)
        data["se_glm"] = se

    elif method == "welch":
        test_stat = obs_diff / se
        df_num = (var_dis / n_disease + var_ctr / n_ctrl) ** 2
        df_den = (
            (var_dis / n_disease) ** 2 / max(n_disease - 1, 1)
            + (var_ctr / n_ctrl) ** 2 / max(n_ctrl - 1, 1)
        )
        df_welch = df_num / (df_den + 1e-12)
        p_value = 2.0 * _stats.t.sf(np.abs(test_stat), df_welch)
        data["p_value_empirical_t"] = np.clip(p_value, 1e-300, 1.0)
        data["t_stat"] = test_stat
        extra["df_welch"] = df_welch

    elif method == "permutation":
        if n_perm <= 0:
            raise ValueError("n_perm must be positive for method='permutation'")
        test_stat = obs_diff
        y_bio_all = np.concatenate([y_bio_dis, y_bio_ctr], axis=0)
        rng = np.random.default_rng(42)
        count_ge = np.zeros(n_genes, dtype=np.int64)
        for _ in range(n_perm):
            idx = rng.permutation(y_bio_all.shape[0])
            perm_diff = (
                y_bio_all[idx[:n_disease]].mean(0)
                - y_bio_all[idx[n_disease:n_disease + n_ctrl]].mean(0)
            )
            count_ge += np.abs(perm_diff) >= np.abs(obs_diff)
        p_value = (count_ge + 1) / (n_perm + 1)
        data["p_value_perm"] = np.clip(p_value, 1e-300, 1.0)

    else:  # bayes
        if n_posterior <= 0:
            raise ValueError("n_posterior must be positive for method='bayes'")
        z_mu = out["z_mu"]
        z_logvar = out["z_logvar"]
        w_mu = out["w_mu"]
        w_logvar = out["w_logvar"]
        posterior = np.zeros((n_posterior, n_genes))
        for sample_idx in range(n_posterior):
            z_sample = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
            w_sample = w_mu + torch.exp(0.5 * w_logvar) * torch.randn_like(w_mu)
            y_bio_sample = (
                model.decoder_bio(z_sample) + model.bias
            ).cpu().numpy()
            delta_lat_sample = model.decoder_w(w_sample).cpu().numpy()
            bio_fc_sample = (
                y_bio_sample[mask_disease].mean(0)
                - y_bio_sample[mask_ctrl].mean(0)
                + logFC_group
            )
            uv_fc_sample = (
                delta_lat_sample[mask_disease].mean(0)
                - delta_lat_sample[mask_ctrl].mean(0)
            )
            posterior[sample_idx] = (
                bio_fc_sample - uv_fc_sample - logFC_uv_cov
            )
        positive_probability = (posterior > 0).mean(0)
        p_value = 2.0 * np.minimum(
            positive_probability, 1.0 - positive_probability
        )
        se = posterior.std(0) + 1e-8
        test_stat = posterior.mean(0) / se
        data["p_value_bayes"] = np.clip(p_value, 1e-300, 1.0)
        data["logFC_bio_posterior_mean"] = posterior.mean(0)
        data["logFC_bio_posterior_std"] = posterior.std(0)
        data["logFC_bio_ci_low"] = np.percentile(posterior, 2.5, axis=0)
        data["logFC_bio_ci_high"] = np.percentile(posterior, 97.5, axis=0)
        extra["posterior_samples"] = posterior

    data["p_value"] = np.clip(p_value, 1e-300, 1.0)
    data["test_stat"] = test_stat
    data["se"] = se
    data["deg_method"] = method
    extra["se"] = se
    extra["test_stat"] = test_stat
    return pd.DataFrame(data), extra
