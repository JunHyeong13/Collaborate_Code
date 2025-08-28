import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------
# Model definition (must match training)
# ------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ------------------------
# Utilities
# ------------------------

def load_artifacts(base_dir: Path):
    model_path = base_dir / 'autoencoder.pt'
    scaler_path = base_dir / 'scaler.joblib'
    columns_path = base_dir / 'used_columns.json'

    if not (model_path.exists() and scaler_path.exists() and columns_path.exists()):
        raise FileNotFoundError('Required artifacts not found: autoencoder.pt, scaler.joblib, used_columns.json')

    with open(columns_path, 'r', encoding='utf-8') as f:
        cols = json.load(f)['columns']
    scaler = joblib.load(scaler_path)

    model = Autoencoder(input_dim=len(cols))
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    return model, scaler, cols


def preprocess_csv(csv_path: Path, cols: list[str], scaler) -> torch.Tensor:
    df_raw = pd.read_csv(csv_path)
    # Align columns and coerce to numeric
    df = df_raw.reindex(columns=cols)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    X = scaler.transform(df.values)
    X_t = torch.tensor(X, dtype=torch.float32)
    return X_t


def pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute DTW distance between two sequences of vectors.
    Returns average path cost (path cost divided by path length) for length normalization.
    seq_a: (Na, D), seq_b: (Nb, D)
    """
    Na, Nb = seq_a.shape[0], seq_b.shape[0]
    # Use a large matrix; for long sequences, consider downsampling/windowing
    cost = np.full((Na + 1, Nb + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    def local(i, j):
        return np.linalg.norm(seq_a[i] - seq_b[j])

    for i in range(1, Na + 1):
        # vectorize over j could be memory-heavy; keep simple for now
        for j in range(1, Nb + 1):
            c = local(i - 1, j - 1)
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Reconstruct path length for normalization
    i, j = Na, Nb
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        choices = [
            (cost[i - 1, j] if i > 0 else np.inf, i - 1, j),
            (cost[i, j - 1] if j > 0 else np.inf, i, j - 1),
            (cost[i - 1, j - 1] if (i > 0 and j > 0) else np.inf, i - 1, j - 1),
        ]
        _, i, j = min(choices, key=lambda t: t[0])
    total_cost = cost[Na, Nb]
    return float(total_cost / max(path_len, 1))


def similarity_from_distance(dist: float, scale: float = 1.0) -> tuple[float, float]:
    """Return (reciprocal similarity, exp similarity)."""
    sim_recip = 1.0 / (1.0 + dist)
    sim_exp = float(np.exp(-dist / max(scale, 1e-8)))
    return sim_recip, sim_exp


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description='Latent-DTW similarity using trained autoencoder')
    parser.add_argument('--test', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--train_glob', type=str, default='hook_left_*.csv', help='Glob of reference (정답) CSVs')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale for exp(-dist/scale) similarity')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    model, scaler, cols = load_artifacts(base_dir)

    # Test latent sequence
    X_test = preprocess_csv(Path(args.test), cols, scaler)
    with torch.no_grad():
        Z_test = model.encoder(X_test).cpu().numpy()

    # Reference files
    ref_files = sorted(base_dir.glob(args.train_glob))
    if not ref_files:
        raise FileNotFoundError(f'No reference files matching {args.train_glob}')

    dists = []
    print(f'비교 대상(정답) 파일 수: {len(ref_files)}')
    for rf in ref_files:
        X_ref = preprocess_csv(rf, cols, scaler)
        with torch.no_grad():
            Z_ref = model.encoder(X_ref).cpu().numpy()
        dist = dtw_distance(Z_ref, Z_test)
        dists.append(dist)
        sim_r, sim_e = similarity_from_distance(dist, args.scale)
        print(f' - {rf.name}: DTW(dist)={dist:.6f}, sim_recip={sim_r:.6f}, sim_exp={sim_e:.6f}')

    dists_np = np.array(dists, dtype=np.float64)
    avg_dist = float(dists_np.mean())
    sim_r_avg, sim_e_avg = similarity_from_distance(avg_dist, args.scale)
    print('\n결과 요약:')
    print(f' - 평균 DTW 거리: {avg_dist:.6f}')
    print(f' - 평균 유사도(recip): {sim_r_avg:.6f}')
    print(f' - 평균 유사도(exp, scale={args.scale}): {sim_e_avg:.6f}')


if __name__ == '__main__':
    main()
