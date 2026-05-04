"""Métricas SLLN-LP (Bunzeck et al. 2024 / ZhoBLiMP)."""

from __future__ import annotations


def slln_lp(log_prob: float, n_tokens: int, *, alpha: float = 0.5) -> float:
    """
    Sub-linear Length Normalized log-probability.

    SLLN-LP(s) = log P(s) / |s|^α con α ∈ (0, 1); por defecto α = 0.5.
    """
    if n_tokens <= 0:
        raise ValueError("n_tokens debe ser positivo")
    return log_prob / (n_tokens**alpha)


def pairwise_slln_lp_margin(
    log_prob_gram: float,
    n_tokens_gram: int,
    log_prob_ungram: float,
    n_tokens_ungram: int,
    *,
    alpha: float = 0.5,
) -> float:
    """
    Diferencia SLLN-LP(gramatical) − SLLN-LP(agramatical).

    Valores > 0 favorecen la oración gramatical bajo la métrica.
    """
    return slln_lp(log_prob_gram, n_tokens_gram, alpha=alpha) - slln_lp(
        log_prob_ungram, n_tokens_ungram, alpha=alpha
    )
