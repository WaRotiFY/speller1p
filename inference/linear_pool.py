"""Линейный пул: p = λ·prior + (1 − λ)·правдоподобие.

Не усиливает уверенность, когда источники согласны, зато устойчив к грубой ошибке
одного из них (MA-12, альтернатива для EXP-04).
"""
import numpy as np

PARAMS = {
    "lambda": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "доля БЯМ λ"},
}


def combine(prior, likelihood, params):
    lam = params["lambda"]
    return np.log(lam * np.exp(prior) + (1 - lam) * np.exp(likelihood) + 1e-300)
