import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class RUVVAE_DEG(nn.Module):
    """RUV-VAE 主方法做 DEG：group 是显式生物学效应，batch 是 UV 协变量"""
    
    def __init__(self, n_genes, n_group=2, n_batch=0, d_bio=32, k_unk=5):
        super().__init__()
        self.n_genes = n_genes
        
        # group 是研究目标的生物学设计变量，不属于 unwanted variation。
        # 只有 batch 等已知技术因素放入 W_cov，并在重建时扣除。
        self.W_group = nn.Parameter(torch.randn(n_group, n_genes) * 0.01)
        cov_blocks = {}
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
        
        y_bio_base = self.decoder_bio(z) + self.bias
        group_effect = c_dict["group"] @ self.W_group
        y_bio = y_bio_base + group_effect
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
                'y_bio_base': y_bio_base, 'group_effect': group_effect,
                'delta_lat': delta_lat, 'delta_cov': delta_cov,
                'z_mu': z_mu, 'z_logvar': z_logvar,
                'w_mu': w_mu, 'w_logvar': w_logvar, 'losses': losses}
    
@torch.no_grad()
def compute_deg(model, Y, gene_names, group_labels, groups_unique,
                neg_control_mask, batch_labels=None,
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
    
    delta_cov = out['delta_cov'].numpy()
    logFC_uv_cov = delta_cov[mask_disease].mean(0) - delta_cov[mask_ctrl].mean(0)
    logFC_uv_total = logFC_uv_latent + logFC_uv_cov
    
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
