"""Логарифмический пул (Байес): log p = α·log prior + β·log правдоподобие.

При α = β = 1 это точная формула Байеса, если физиология присылает правдоподобие
без собственной языковой модели (MA-03, MA-12).
"""

PARAMS = {
    "alpha": {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "label": "вес БЯМ α"},
    "beta": {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "label": "вес физиологии β"},
}


def combine(prior, likelihood, params):
    return params["alpha"] * prior + params["beta"] * likelihood
