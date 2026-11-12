"""BirdNET v2.4 embeddings for every clip in data_raw/ (1,024 dims, mean of up to ten 3 s windows).

Needs: pip install birdnetlib ai-edge-litert soundfile librosa  (the model file ships inside birdnetlib).
Output: feat/birdnet_embeddings.npz with names = "<Species>/<year>/<file>.wav".
"""
import glob, os, numpy as np, soundfile as sf, librosa, birdnetlib
from ai_edge_litert.interpreter import Interpreter

MODEL = os.path.join(os.path.dirname(birdnetlib.__file__), 'models/analyzer/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite')
SR, W, MAXW = 48000, 3 * 48000, 10
it = Interpreter(model_path=MODEL, num_threads=os.cpu_count(), experimental_preserve_all_tensors=True)
it.allocate_tensors()
inp = it.get_input_details()[0]['index']
emb_idx = it.get_output_details()[0]['index'] - 1   # layer before the classifier = embedding

names, E, nwin = [], [], []
for path in sorted(glob.glob('data_raw/*/*/*.wav')):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type='soxr_hq')
    windows = [y[i:i + W] for i in range(0, max(len(y), 1), W)][:MAXW]
    em = []
    for c in windows:
        if len(c) < W:
            c = np.pad(c, (0, W - len(c)))
        it.set_tensor(inp, c[None, :].astype(np.float32)); it.invoke()
        em.append(it.get_tensor(emb_idx)[0].copy())
    names.append(os.path.relpath(path, 'data_raw').replace(os.sep, '/'))
    E.append(np.mean(em, 0)); nwin.append(len(windows))
os.makedirs('feat', exist_ok=True)
np.savez_compressed('feat/birdnet_embeddings.npz', names=np.array(names), emb=np.array(E, np.float32), nwin=np.array(nwin))
print(len(names), 'clips embedded')
