"""P300 decoder example: synthetic EDF, calibration, decoding and the likelihood contract."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("pyedflib")

from eeg_speller.core.config import load_config
from eeg_speller.physiology.base import PhysiologyObservation, validate_observation
from eeg_speller.physiology.edf_demo import calibrate, generate_edf
from eeg_speller.physiology.p300 import CalibratedP300, P300Features
from eeg_speller.physiology.sources import EDFSource


@pytest.fixture
def recording(tmp_path):
    cfg = load_config("configs/p300_decoder.yaml")
    labels = generate_edf(cfg, "ажмтчя", tmp_path / "calibration.edf", signal_seed=521)
    calibrate(cfg, tmp_path / "calibration.edf", labels, tmp_path / "model.json")
    cfg.physiology["model_path"] = str(tmp_path / "model.json")
    return cfg, tmp_path


def decode(cfg, edf_path, model_path):
    """The same steps as stimulus_kit/decoders/p300_lsl_decoder.py, reading an EDF file."""
    features = P300Features(cfg.physiology)
    model = CalibratedP300(model_path, features)
    source = EDFSource({"path": edf_path}, features.channels)
    symbols = cfg.repertoire["symbols"]
    rows, cols = cfg.repertoire["layout"]
    out = []
    for eid in sorted(source.markers):
        logits = np.zeros(len(symbols))
        for marker in source.markers[eid]:
            x = features.extract(source.window(marker["source_time_s"], features.window), source.fs)
            if x is None:
                continue
            idx = (range(marker["index"] * cols, (marker["index"] + 1) * cols) if marker["kind"] == "row"
                   else range(marker["index"], len(symbols), cols))
            for i in idx:
                logits[i] += model.llr(x)
        out.append(logits)
    return out


def test_arbitrary_text_decoded_from_independent_edf_waveform(recording):
    cfg, directory = recording
    rng = np.random.default_rng(8402)
    text = "".join(rng.choice(cfg.repertoire["symbols"][:-1], 5))
    generate_edf(cfg, text, directory / "test.edf", signal_seed=9238)
    logits = decode(cfg, directory / "test.edf", directory / "model.json")
    assert "".join(cfg.repertoire["symbols"][int(np.argmax(v))] for v in logits) == text


def test_removing_p300_gives_flat_evidence(recording):
    cfg, directory = recording
    generate_edf(cfg, "п", directory / "test.edf", signal_seed=18, amplitude_uv=0)
    logits = decode(cfg, directory / "test.edf", directory / "model.json")[0]
    p = np.exp(logits - logits.max())
    assert (p / p.sum()).max() < 0.1


def test_device_channel_aliases_and_unknown_channels(recording):
    cfg, directory = recording
    generate_edf(cfg, "а", directory / "test.edf", signal_seed=2)
    source = EDFSource({"path": directory / "test.edf", "channel_aliases": {"posterior": "Pz"}}, ["posterior"])
    assert source.data.shape[0] == 1 and source.fs == 256
    with pytest.raises(ValueError, match="channels missing"):
        EDFSource({"path": directory / "test.edf"}, ["missing"])


def test_model_requires_matching_preprocessing(recording):
    cfg, directory = recording
    cfg.physiology["response_s"] = [0.4, 0.6]
    with pytest.raises(ValueError, match="recalibrate"):
        CalibratedP300(directory / "model.json", P300Features(cfg.physiology))


def test_contract_rejects_wrong_epoch_shape_and_embedded_prior():
    valid = PhysiologyObservation(3, np.full(36, -np.log(36)), "likelihood")
    validate_observation(valid, 3, 36)
    for invalid in (replace(valid, epoch_id=2), replace(valid, log_p=np.zeros(6)),
                    replace(valid, semantics="posterior_with_prior"), replace(valid, log_p=np.zeros(36))):
        with pytest.raises(ValueError):
            validate_observation(invalid, 3, 36)
