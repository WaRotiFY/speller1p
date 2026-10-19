"""Байес с температурой и «полом» для БЯМ.

Сначала prior сглаживается: p ← (1 − ε)·p^(1/T) + ε/N, чтобы слишком уверенная БЯМ
не могла почти запретить правильную букву. Потом обычное произведение с правдоподобием.
"""
import numpy as np

PARAMS = {
    "temperature": {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1, "label": "температура БЯМ T"},
    "floor": {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01, "label": "доля равномерного ε"},
}


def combine(prior, likelihood, params):
    t, eps = params["temperature"], params["floor"]
    p = np.exp(prior / t)
    p /= p.sum()
    p = (1 - eps) * p + eps / len(p)
    return np.log(p) + likelihood
