"""Perch 2.0 embeddings for every clip in data_raw/ (1,536 dims, mean of up to six 5 s windows).

Needs: pip install "tensorflow-cpu~=2.20.0" soundfile librosa, and the perch_v2_cpu SavedModel from
https://www.kaggle.com/models/google/bird-vocalization-classifier (TensorFlow2, perch_v2_cpu, v1) unpacked to PERCH_DIR.
Output: feat/perch2_embeddings.npz with names = "<Species>/<year>/<file>.wav".
"""
import glob, os, numpy as np, soundfile as sf, librosa
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf

PERCH_DIR = os.environ.get('PERCH_DIR', 'perch_v2_cpu')
f = tf.saved_model.load(PERCH_DIR).signatures['serving_default']
SR, W, MAXW, B = 32000, 5 * 32000, 6, 4

names, nwin, acc, queue, owner = [], [], {}, [], []
def run(force=False):
    global queue, owner
    while len(queue) >= B or (force and queue):
        q, o = queue[:B], owner[:B]; queue, owner = queue[B:], owner[B:]
        x = np.stack(q + [np.zeros(W, np.float32)] * (B - len(q)))
        e = f(inputs=x)['embedding'].numpy()[:len(q)]
        for k, i in enumerate(o):
            acc[i].append(e[k])

for path in sorted(glob.glob('data_raw/*/*/*.wav')):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(1)[:int(sr * MAXW * 5)]          # only the first 30 s are used
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type='soxr_hq')
    windows = [y[i:i + W] for i in range(0, max(len(y), 1), W)][:MAXW]
    idx = len(names); names.append(os.path.relpath(path, 'data_raw').replace(os.sep, '/'))
    nwin.append(len(windows)); acc[idx] = []
    for c in windows:
        if len(c) < W:
            c = np.pad(c, (0, W - len(c)))
        queue.append(c.astype(np.float32)); owner.append(idx)
    run()
run(force=True)
E = np.array([np.mean(acc[i], 0) for i in range(len(names))], np.float32)
os.makedirs('feat', exist_ok=True)
np.savez_compressed('feat/perch2_embeddings.npz', names=np.array(names), emb=E, nwin=np.array(nwin))
print(len(names), 'clips embedded')
