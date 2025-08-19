import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def safe_array(x, name="array"):
    arr = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(arr)):
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        print(f"경고: {name}에 NaN 또는 Inf 값 {n_bad}개를 0으로 치환했습니다.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def ensure_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def dtw_similarity(seq1, seq2, k=10):
    if not isinstance(seq1, np.ndarray): seq1 = np.array(seq1, dtype=float)
    if not isinstance(seq2, np.ndarray): seq2 = np.array(seq2, dtype=float)
    if seq1.size == 0 or seq2.size == 0: return 0.0
    seq1 = safe_array(seq1, name="dtw_seq1")
    seq2 = safe_array(seq2, name="dtw_seq2")
    if seq1.ndim == 1:
        dist_func = lambda x, y: float(abs(x - y))
    elif seq1.shape[1] == 4:
        def quat_dist(q1, q2):
            d = float(abs(np.dot(q1, q2)))
            d = np.clip(d, -1.0, 1.0)
            return 1.0 - d
        dist_func = quat_dist
    else:
        dist_func = euclidean
    distance, _ = fastdtw(seq1, seq2, dist=dist_func)
    normalized_distance = distance / (len(seq1) + len(seq2))
    return float(np.exp(-k * normalized_distance))


