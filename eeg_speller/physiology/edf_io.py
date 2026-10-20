"""Reading and writing EDF/EDF+ with ``edfio`` (pure Python, no compiler needed).

One small interface for the three places that touch EDF files: the EDF source of
the decoder, the synthetic recording generator and ``stimulus_kit/tools/edf_to_lsl.py``.
"""
from datetime import date
from pathlib import Path

import numpy as np


def _edfio():
    try:
        import edfio
    except ImportError:
        raise ValueError("EDF support requires the edfio package: uv pip install -e .") from None
    return edfio


def read_edf(path):
    """Return (labels, rates_hz, physical_dimensions, signals, annotations).

    ``signals`` are physical values (as stored, in their own units); annotations are
    (onset_s, duration_s, text) tuples. Annotation channels are not among the signals.
    """
    edf = _edfio().read_edf(str(path))
    signals = list(edf.signals)
    labels = [s.label.strip() for s in signals]
    rates = [float(s.sampling_frequency) for s in signals]
    dims = [s.physical_dimension.strip() for s in signals]
    data = [np.asarray(s.data, dtype=float) for s in signals]
    notes = [(float(a.onset), float(a.duration or 0.0), str(a.text)) for a in edf.annotations]
    return labels, rates, dims, data, notes


def write_edf(path, labels, fs, data, *, dimension="uV", physical_range=(-200.0, 200.0),
              annotations=(), startdate=date(2026, 1, 1), equipment="X", transducer="", prefilter=""):
    """Write an EDF+ file: ``data`` is channels × samples in ``dimension`` units at ``fs`` Hz."""
    edfio = _edfio()
    lo, hi = physical_range
    signals = [edfio.EdfSignal(np.clip(np.asarray(x, dtype=float), lo, hi), fs, label=name,
                               physical_dimension=dimension, physical_range=(lo, hi),
                               transducer_type=transducer, prefiltering=prefilter)
               for name, x in zip(labels, data)]
    edf = edfio.Edf(signals, recording=edfio.Recording(startdate=startdate, equipment_code=equipment),
                    annotations=[edfio.EdfAnnotation(onset, duration, text) for onset, duration, text in annotations])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    edf.write(str(path))
