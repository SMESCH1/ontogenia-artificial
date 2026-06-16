"""
Representational probes via Sparse Autoencoders (SAEs) on Pythia checkpoints.

Complementa el análisis conductual (BLiMP/Zorro) con análisis representacional:
qué features internos codifica Pythia para fenómenos sintácticos, y cómo evolucionan
a través de los checkpoints de entrenamiento.

Pre-requisito: pip install sparsify  (EleutherAI)

SAEs disponibles (checkpoint final de cada modelo):
  EleutherAI/sae-pythia-70m-deduped-32k
  EleutherAI/sae-pythia-160m-32k
  EleutherAI/sae-pythia-160m-deduped-32k
  EleutherAI/sae-pythia-410m-65k
  EleutherAI/sae-pythia-410m-deduped-65k

LIMITACIÓN: los SAEs están entrenados sobre el checkpoint FINAL de cada modelo.
Proyectar activaciones de checkpoints intermedios a través de ellos es una
aproximación — reportar como limitación en el paper.
Ver docs/06_sae_representational_probes.md para el workflow completo.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sparsify import Sae
    _SPARSIFY_AVAILABLE = True
except ImportError:
    _SPARSIFY_AVAILABLE = False

_SAE_REPO_MAP = {
    "pythia-70m-deduped":  "EleutherAI/sae-pythia-70m-deduped-32k",
    "pythia-160m":         "EleutherAI/sae-pythia-160m-32k",
    "pythia-160m-deduped": "EleutherAI/sae-pythia-160m-deduped-32k",
    "pythia-410m":         "EleutherAI/sae-pythia-410m-65k",
    "pythia-410m-deduped": "EleutherAI/sae-pythia-410m-deduped-65k",
}


def _require_sparsify() -> None:
    if not _SPARSIFY_AVAILABLE:
        raise ImportError(
            "Librería 'sparsify' no encontrada. Instalá con: pip install sparsify"
        )


def load_sae(model_name: str, layer: int) -> Sae:
    """
    Carga el SAE pre-entrenado de EleutherAI para un modelo Pythia y capa dados.

    Args:
        model_name: clave del modelo, p.ej. 'pythia-160m-deduped'
        layer: índice de capa (0-based). Pythia-160M tiene 12 capas.
               Para fenómenos sintácticos se recomiendan capas medias (4–8).
    """
    _require_sparsify()
    repo = _SAE_REPO_MAP.get(model_name)
    if repo is None:
        raise ValueError(
            f"Modelo '{model_name}' no tiene SAE disponible. "
            f"Opciones: {list(_SAE_REPO_MAP)}"
        )
    return Sae.load_from_hub(repo, hookpoint=f"layers.{layer}")


def extract_activations(
    sentences: list[str],
    model_hf_name: str,
    revision: str | None,
    layer: int,
    device: str = "cuda",
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Extrae hidden states en una capa dada para una lista de oraciones.

    Args:
        sentences:     lista de strings (gramaticales o agramaticales)
        model_hf_name: p.ej. 'EleutherAI/pythia-160m-deduped'
        revision:      p.ej. 'step10000' para checkpoint intermedio, None para final
        layer:         índice de capa (0-based)
        device:        'cuda' o 'cpu'
        batch_size:    oraciones por batch

    Returns:
        Tensor float32 de forma (N, d_model) — activación del último token por oración
    """
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_hf_name,
        revision=revision,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    ).to(device).eval()

    all_acts: list[torch.Tensor] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # hidden_states[0] = embeddings; [layer+1] = salida de la capa `layer`
        hs = out.hidden_states[layer + 1]
        last_pos = enc["attention_mask"].sum(1) - 1
        last_tok = hs[torch.arange(len(batch)), last_pos]
        all_acts.append(last_tok.cpu().float())

    del model
    torch.cuda.empty_cache()
    return torch.cat(all_acts, dim=0)


def get_feature_activations(
    activations: torch.Tensor,
    sae: Sae,
) -> torch.Tensor:
    """
    Proyecta activaciones a través del SAE y devuelve el vector de features sparse.

    Returns:
        Tensor (N, n_features) con activaciones de cada feature SAE
    """
    _require_sparsify()
    return sae.encode(activations).latent_acts  # (N, n_features)


def contrastive_features(
    grammatical_acts: torch.Tensor,
    ungrammatical_acts: torch.Tensor,
    sae: Sae,
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """
    Rankea features SAE por diferencia de activación media entre oraciones
    gramaticales y agramaticales. Los top features son candidatos a capturar
    el fenómeno sintáctico de interés.

    Returns:
        Lista de (feature_idx, diferencia_media) ordenada por |diferencia| desc.
    """
    feat_gram   = get_feature_activations(grammatical_acts, sae).mean(0).numpy()
    feat_ungram = get_feature_activations(ungrammatical_acts, sae).mean(0).numpy()
    diff = feat_gram - feat_ungram
    ranked = sorted(enumerate(diff.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_k]


def track_features_across_checkpoints(
    sentences_good: list[str],
    sentences_bad: list[str],
    model_hf_name: str,
    model_short_name: str,
    checkpoints: list[str],
    layer: int,
    top_features: list[int] | None = None,
    device: str = "cuda",
) -> dict:
    """
    Pipeline longitudinal: para cada checkpoint extrae activaciones, proyecta por SAE,
    y computa la diferencia de activación gramatical/agramatical para los features
    de interés.

    Args:
        sentences_good:   oraciones gramaticales del paradigma (BLiMP/Zorro)
        sentences_bad:    sus contrapartes agramaticales (mismo orden)
        model_hf_name:    p.ej. 'EleutherAI/pythia-160m-deduped'
        model_short_name: p.ej. 'pythia-160m-deduped'
        checkpoints:      lista de revisiones, p.ej. ['step512', 'step1000', ...]
        layer:            capa a analizar
        top_features:     índices de features a trackear (obtenidos de contrastive_features
                          sobre el checkpoint final). Si None, solo registra mean_diff.
        device:           'cuda' o 'cpu'

    Returns:
        {
          'checkpoints':   ['step512', ...],
          'mean_diff':     [float, ...],         # diferencia media por checkpoint
          'feature_diffs': {feature_idx: [float, ...]},  # solo si top_features != None
        }
    """
    sae = load_sae(model_short_name, layer)
    results: dict = {
        "checkpoints": checkpoints,
        "mean_diff": [],
        "feature_diffs": {fi: [] for fi in (top_features or [])},
    }

    for revision in checkpoints:
        acts_good = extract_activations(sentences_good, model_hf_name, revision, layer, device)
        acts_bad  = extract_activations(sentences_bad,  model_hf_name, revision, layer, device)

        feat_good = get_feature_activations(acts_good, sae).mean(0).numpy()
        feat_bad  = get_feature_activations(acts_bad,  sae).mean(0).numpy()
        diff = feat_good - feat_bad

        results["mean_diff"].append(float(diff.mean()))
        for fi in (top_features or []):
            results["feature_diffs"][fi].append(float(diff[fi]))

    return results
