"""BirdNET v2.4 embeddings for every clip in data_raw/ (1,024 dims, mean of up to ten 3 s windows).

Needs: pip install birdnetlib ai-edge-litert soundfile librosa  (the model file ships inside birdnetlib).
Output: feat/birdnet_embeddings.npz with names = "<Species>/<year>/<file>.wav".
"""
from __future__ import annotations

import glob
import os

import birdnetlib
import librosa
import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter

MODEL = os.path.join(os.path.dirname(birdnetlib.__file__),
                     "models/analyzer/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
SR = 48000
WINDOW = 3 * SR
MAX_WINDOWS = 10


def load_windows(path: str) -> list[np.ndarray]:
    """Mono audio at 48 kHz cut into up to ten zero padded 3 s windows."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    windows = [y[i:i + WINDOW] for i in range(0, max(len(y), 1), WINDOW)][:MAX_WINDOWS]
    return [np.pad(w, (0, WINDOW - len(w))) if len(w) < WINDOW else w for w in windows]


class BirdNetEmbedder:
    """Wraps the BirdNET TFLite model and reads the layer before the classifier."""

    def __init__(self, model_path: str = MODEL):
        self.interpreter = Interpreter(model_path=model_path, num_threads=os.cpu_count(),
                                       experimental_preserve_all_tensors=True)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.embedding_index = self.interpreter.get_output_details()[0]["index"] - 1

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_index, window[None, :].astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.embedding_index)[0].copy()


def main() -> None:
    embed = BirdNetEmbedder()
    names, embeddings, n_windows = [], [], []
    for path in sorted(glob.glob("data_raw/*/*/*.wav")):
        windows = load_windows(path)
        names.append(os.path.relpath(path, "data_raw").replace(os.sep, "/"))
        embeddings.append(np.mean([embed(w) for w in windows], 0))
        n_windows.append(len(windows))
    os.makedirs("feat", exist_ok=True)
    np.savez_compressed("feat/birdnet_embeddings.npz", names=np.array(names),
                        emb=np.array(embeddings, np.float32), nwin=np.array(n_windows))
    print(len(names), "clips embedded")


if __name__ == "__main__":
    main()
