import json
from pathlib import Path

nb_path = Path("/home/junyichen/code/RUVAEDEG/testRUVVAE_ZINB.ipynb")
nb = json.loads(nb_path.read_text())

new_cell = {
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": [
        "# ========== 13e. Extract company (batch) embedding vectors ==========\n",
        "from scvi.model.base._embedding_mixin import EmbeddingModuleMixin\n",
        "\n",
        "emb_layer = EmbeddingModuleMixin.get_embedding(model_disease.module, \"batch\")\n",
        "print(f\"Batch embedding layer: {emb_layer}\")\n",
        "print(f\"  num_embeddings (n_company): {emb_layer.num_embeddings}\")\n",
        "print(f\"  embedding_dim         : {emb_layer.embedding_dim}\")\n",
        "\n",
        "with torch.no_grad():\n",
        "    company_emb = emb_layer.weight.detach().cpu().numpy()\n",
        "\n",
        "company_categories = model_disease.adata_manager.get_state_registry(\n",
        "    \"batch\"\n",
        ").categorical_mapping\n",
        "company_to_idx = dict(zip(company_categories, range(len(company_categories))))\n",
        "\n",
        "df_company_emb = pd.DataFrame(\n",
        "    company_emb,\n",
        "    index=[f\"company_{i}\" for i in range(company_emb.shape[0])],\n",
        "    columns=[f\"emb_{j}\" for j in range(company_emb.shape[1])],\n",
        ")\n",
        "df_company_emb.insert(0, \"company\", company_categories)\n",
        "display(df_company_emb)\n",
        "\n",
        "# Per-cell embedding (one lookup per cell)\n",
        "batch_idx = adata_scvi.obs[batch_key].map(company_to_idx).to_numpy()\n",
        "batch_emb_per_cell = company_emb[batch_idx]\n",
        "print(f\"batch_emb_per_cell shape: {batch_emb_per_cell.shape}\")\n",
        "\n",
        "# Sanity check: rebuild the px-side contribution via the decoder projection,\n",
        "# which is exactly what scVI uses internally to inject the batch effect.\n",
        "px_cat_bias = model_disease.module.px_decoder.cat_bias\n",
        "px_cat_bias_weight = px_cat_bias.weight.detach().cpu().numpy()\n",
        "print(f\"px cat_bias weight shape: {px_cat_bias_weight.shape}\")\n",
        "reconstructed_bias = company_emb @ px_cat_bias_weight\n",
        "print(f\"company_emb @ px_cat_bias_weight shape: {reconstructed_bias.shape}\")\n"
    ],
    "id": "#VSC-extract-batch-emb",
}

nb["cells"].append(new_cell)
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("appended cell; total cells:", len(nb["cells"]))