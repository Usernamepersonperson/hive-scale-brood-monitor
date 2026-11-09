"""Queen-loss hint from a simple spectral feature on public annotated hive audio.

Features are computed per window from Welch band powers (no learned embeddings). The
classifier is a one-feature threshold chosen on one hive and tested on the other hive,
so the reported accuracy is hive-held-out, not window-shuffled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

BANDS = ((100, 200), (200, 300), (300, 450), (450, 700), (700, 1200))
WINDOW_S = 10.0
FEATURES = [f"b{lo}_{hi}" for lo, hi in BANDS] + ["centroid", "peak_hz"]


def window_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Band-power fractions (100-1200 Hz), spectral centroid and dominant peak of one window."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    f, p = signal.welch(x, fs=sr, nperseg=min(len(x), 8192))
    sel = (f >= 100) & (f < 1200)
    tot = p[sel].sum()
    if tot <= 0:
        return {k: float("nan") for k in FEATURES}
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"b{lo}_{hi}"] = float(p[m].sum() / tot)
    out["centroid"] = float((f[sel] * p[sel]).sum() / tot)
    pk = (f >= 100) & (f < 700)
    out["peak_hz"] = float(f[pk][np.argmax(p[pk])])
    return out


def parse_name(name: str) -> dict:
    """Hive id, date tag and queen label from a file name of the annotated audio set."""
    stem = Path(name).stem
    parts = stem.split("_")
    hive = next((p for p in parts if p.startswith("H") and p[1:].isdigit()), parts[0])
    queenless = "NO_QueenBee" in stem
    return {"hive": hive, "day": "_".join(parts[1:4]) if len(parts) > 4 else "", "queenless": int(queenless)}


def read_labels(lab: Path) -> list[tuple[float, float]]:
    """Segments annotated as bee sound (start, end) in seconds from a .lab file."""
    segs = []
    for line in lab.read_text().splitlines():
        t = line.split()
        if len(t) >= 3 and t[2] == "bee":
            segs.append((float(t[0]), float(t[1])))
    return segs


def extract_file(wav: Path, lab: Path | None = None, max_windows: int = 40) -> pd.DataFrame:
    sr, x = wavfile.read(wav)
    x = x.mean(axis=1) if x.ndim > 1 else x
    n = int(WINDOW_S * sr)
    spans = []
    if lab is not None and lab.exists():
        for a, b in read_labels(lab):
            s = a
            while s + WINDOW_S <= b:
                spans.append(s)
                s += WINDOW_S
    else:
        spans = list(np.arange(0, len(x) / sr - WINDOW_S, WINDOW_S))
    spans = spans[:: max(1, len(spans) // max_windows)][:max_windows]
    meta = parse_name(wav.name)
    rows = []
    for s in spans:
        i = int(s * sr)
        rows.append({**meta, "file": wav.stem, "t0": s, **window_features(x[i:i + n], sr)})
    return pd.DataFrame(rows)


def extract_dir(folder: Path, max_windows: int = 40) -> pd.DataFrame:
    frames = [extract_file(w, w.with_suffix(".lab"), max_windows) for w in sorted(folder.glob("*.wav"))]
    return pd.concat(frames, ignore_index=True)


def best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    """Threshold and sign maximising balanced accuracy for y (1 = queenless). Returns (thr, sign, bacc)."""
    best = (float(np.median(x)), 1, 0.0)
    for thr in np.unique(np.quantile(x, np.linspace(0.02, 0.98, 97))):
        for sign in (1, -1):
            pred = (sign * (x - thr) > 0).astype(int)
            tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
            tnr = 1 - pred[y == 0].mean() if (y == 0).any() else 0.0
            bacc = (tpr + tnr) / 2
            if bacc > best[2]:
                best = (float(thr), sign, float(bacc))
    return best


def evaluate_holdout(feat: pd.DataFrame) -> dict:
    """Pick feature + threshold on one hive only, test on the other hive (both directions)."""
    feat = feat.dropna(subset=FEATURES)
    hives = sorted(feat["hive"].unique())
    if len(hives) < 2:
        raise ValueError("need at least two hives for a hive-held-out test")
    out = []
    for tr_h in hives:
        tr = feat[feat.hive == tr_h]
        te = feat[feat.hive != tr_h]
        if tr.queenless.nunique() < 2 or te.queenless.nunique() < 2:
            raise ValueError("each hive needs queen-present and queen-absent recordings")
        best = max(((n, *best_threshold(tr[n].to_numpy(), tr.queenless.to_numpy())) for n in FEATURES),
                   key=lambda r: r[3])
        name, thr, sign, tr_bacc = best
        pred = (sign * (te[name].to_numpy() - thr) > 0).astype(int)
        y = te.queenless.to_numpy()
        tpr, tnr = pred[y == 1].mean(), 1 - pred[y == 0].mean()
        file_pred = pd.Series(pred, index=te.index).groupby(te.file).mean() > 0.5
        file_y = te.groupby("file").queenless.first().astype(bool)
        out.append({"train_hive": tr_h, "test_hive": ",".join(sorted(te.hive.unique())),
                    "feature": name, "threshold": thr, "sign": sign, "train_bacc": tr_bacc,
                    "window_bacc": float((tpr + tnr) / 2), "tpr": float(tpr), "tnr": float(tnr),
                    "file_acc": float((file_pred == file_y).mean()),
                    "n_windows": int(len(te)), "n_files": int(len(file_y))})
    return {"mean_window_bacc": float(np.mean([r["window_bacc"] for r in out])),
            "mean_file_acc": float(np.mean([r["file_acc"] for r in out])), "per_direction": out}
