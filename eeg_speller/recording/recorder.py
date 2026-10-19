"""Sequential event recorder implementing DATA_MODEL section 5."""
import json
from pathlib import Path
from eeg_speller.core.events import EVENT_TYPES


def serializable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


class SessionRecorder:
    """Session Recorder; NEED-F-12/13; MA-07; Q-30."""
    def __init__(self, directory: Path, session_id: str, participant_id: str, clock):
        self.path = directory / "events.jsonl"
        self.session_id, self.participant_id, self.clock = session_id, participant_id, clock
        self.seq = 0
        self.file = self.path.open("w", encoding="utf-8")

    def emit(self, block: str, kind: str, payload: dict, epoch_id=None, t_ns=None):
        # IMPROV[Q-27]
        # IMPROV[Q-30]
        if kind not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {kind}")
        event = {"schema": "eeg-speller/events@1", "session_id": self.session_id,
                 "participant_id": self.participant_id, "seq": self.seq,
                 "t_ns": self.clock.t_ns() if t_ns is None else t_ns,
                 **self._lsl(),
                 "block": block, "type": kind, "epoch_id": epoch_id,
                 "payload": serializable(payload)}
        self.file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.seq += 1
        return event

    def _lsl(self) -> dict:
        """Live sessions also stamp every event with the LSL clock shared with EEG and markers."""
        source = getattr(self.clock, "lsl", None)
        return {"lsl_s": round(float(source()), 6)} if source is not None else {}

    def close(self):
        self.file.close()
