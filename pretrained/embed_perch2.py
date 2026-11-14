"""Perch 2.0 embeddings for every clip in data_raw/ (1,536 dims, mean of up to six 5 s windows).

Needs: pip install "tensorflow-cpu~=2.20.0" soundfile librosa, and the perch_v2_cpu SavedModel from
https://www.kaggle.com/models/google/bird-vocalization-classifier (TensorFlow2, perch_v2_cpu, v1) unpacked to PERCH_DIR.
Output: feat/perch2_embeddings.npz with names = "<Species>/<year>/<file>.wav".
"""
import glob
import os

import librosa
import numpy as np
import soundfile as sf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402  (log level must be set first)

PERCH_DIR = os.environ.get("PERCH_DIR", "perch_v2_cpu")
SR = 32000
WINDOW = 5 * SR
MAX_WINDOWS = 6
BATCH = 4


def load_windows(path):
    """Mono audio at 32 kHz (first 30 s only) cut into up to six zero padded 5 s windows."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)[:int(sr * MAX_WINDOWS * 5)]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    windows = [y[i:i + WINDOW] for i in range(0, max(len(y), 1), WINDOW)][:MAX_WINDOWS]
    return [(np.pad(w, (0, WINDOW - len(w))) if len(w) < WINDOW else w).astype(np.float32) for w in windows]


class BatchedEmbedder:
    """Queues windows from many clips and runs the model in fixed size batches.

    The SavedModel expects a fixed batch size, so the last batch is padded with
    silent windows whose outputs are discarded.
    """

    def __init__(self, model, batch_size=BATCH):
        self.model, self.batch_size = model, batch_size
        self.queue, self.owner, self.embeddings = [], [], {}

    def add(self, clip_index, windows):
        self.embeddings[clip_index] = []
        for window in windows:
            self.queue.append(window)
            self.owner.append(clip_index)
        self._run()

    def finish(self):
        self._run(force=True)

    def _run(self, force=False):
        while len(self.queue) >= self.batch_size or (force and self.queue):
            batch, owners = self.queue[:self.batch_size], self.owner[:self.batch_size]
            self.queue, self.owner = self.queue[self.batch_size:], self.owner[self.batch_size:]
            padding = [np.zeros(WINDOW, np.float32)] * (self.batch_size - len(batch))
            outputs = self.model(inputs=np.stack(batch + padding))["embedding"].numpy()[:len(batch)]
            for output, clip_index in zip(outputs, owners):
                self.embeddings[clip_index].append(output)

    def mean_embeddings(self, n_clips):
        return np.array([np.mean(self.embeddings[i], 0) for i in range(n_clips)], np.float32)


def main():
    model = tf.saved_model.load(PERCH_DIR).signatures["serving_default"]
    embedder = BatchedEmbedder(model)
    names, n_windows = [], []
    for path in sorted(glob.glob("data_raw/*/*/*.wav")):
        windows = load_windows(path)
        embedder.add(len(names), windows)
        names.append(os.path.relpath(path, "data_raw").replace(os.sep, "/"))
        n_windows.append(len(windows))
    embedder.finish()
    os.makedirs("feat", exist_ok=True)
    np.savez_compressed("feat/perch2_embeddings.npz", names=np.array(names),
                        emb=embedder.mean_embeddings(len(names)), nwin=np.array(n_windows))
    print(len(names), "clips embedded")


if __name__ == "__main__":
    main()
