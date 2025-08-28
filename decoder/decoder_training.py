import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import joblib

# 1. 데이터 로드 및 전처리
script_dir = Path(__file__).parent
csv_paths = sorted(script_dir.glob('hook_left_*.csv'))
if not csv_paths:
    raise FileNotFoundError("decoder/ 디렉터리 내에 'hook_left_*.csv' 파일을 찾을 수 없습니다. 파일명을 확인하세요.")

print(f"발견된 학습 CSV 수: {len(csv_paths)}")
for p in csv_paths:
    print(f" - {p.name}")

# 공통 숫자 컬럼 찾기
numeric_cols_per_file = []
for p in csv_paths:
    df_head = pd.read_csv(p, nrows=5)
    numeric_cols = set(df_head.select_dtypes(include=[np.number]).columns.tolist())
    numeric_cols_per_file.append(numeric_cols)

common_numeric_cols = set.intersection(*numeric_cols_per_file)
if not common_numeric_cols:
    raise ValueError("모든 파일에 공통으로 존재하는 수치형 컬럼이 없습니다.")

common_cols_sorted = sorted(common_numeric_cols)
print(f"공통 사용 컬럼 수: {len(common_cols_sorted)}")

# 데이터 읽기
all_frames = []
for p in csv_paths:
    df = pd.read_csv(p, usecols=lambda c: c in common_cols_sorted)
    df = df.apply(pd.to_numeric, errors='coerce')
    all_frames.append(df[common_cols_sorted])

df_combined = pd.concat(all_frames, ignore_index=True)

# NaN/Inf 처리 및 분산0 컬럼 제거
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.fillna(0.0, inplace=True)
std_series = df_combined.std(axis=0, ddof=0)
zero_var_cols = std_series[std_series == 0].index.tolist()
if zero_var_cols:
    print(f"분산 0 컬럼 제거: {len(zero_var_cols)}개")
    df_combined.drop(columns=zero_var_cols, inplace=True)

print(f"최종 사용 컬럼 수: {df_combined.shape[1]}")

# 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_combined.values)

X = torch.tensor(scaled_data, dtype=torch.float32)
if not torch.isfinite(X).all():
    raise ValueError("전처리 후에도 NaN/Inf가 존재합니다.")

# 아티팩트 저장 경로
artifacts_dir = Path(__file__).parent
model_path = artifacts_dir / 'autoencoder.pt'
scaler_path = artifacts_dir / 'scaler.joblib'
columns_path = artifacts_dir / 'used_columns.json'

# Dataset & DataLoader
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. 오토인코더 모델
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_dim = X.shape[1]
model = Autoencoder(input_dim)
print("모델 구조:\n", model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("학습 완료!")

# 4. 학습 데이터 재구성 오류
model.eval()
with torch.no_grad():
    reconstructed_data = model(X)
    mse_per_sample = torch.mean((reconstructed_data - X)**2, dim=1)
    train_mse_mean = mse_per_sample.mean().item()
    train_mse_std = mse_per_sample.std().item()
    threshold = train_mse_mean + 2 * train_mse_std

print("\n학습 데이터셋 평균 MSE:", train_mse_mean)
print("임계값(평균+2표준편차):", threshold)

# 5. 모델/스케일러/컬럼 저장
torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)
used_columns = df_combined.columns.tolist()
with open(columns_path, 'w', encoding='utf-8') as f:
    json.dump({"columns": used_columns}, f, ensure_ascii=False, indent=2)
print(f"모델 저장: {model_path.name}, 스케일러 저장: {scaler_path.name}, 컬럼 저장: {columns_path.name}")

# 6. 테스트 평가
test_path = Path(__file__).parent / 'test_main.csv'
if test_path.exists():
    print(f"\n테스트 파일 발견: {test_path.name}")
    df_test_raw = pd.read_csv(test_path)
    df_test = df_test_raw.reindex(columns=used_columns)
    df_test = df_test.apply(pd.to_numeric, errors='coerce')
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0.0, inplace=True)

    test_scaled = scaler.transform(df_test.values)
    X_test = torch.tensor(test_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        recon_test = model(X_test)
        test_mse = torch.mean((recon_test - X_test)**2, dim=1)
        test_mean = test_mse.mean().item()
        test_std = test_mse.std().item()

        # ✅ 퍼센티지 스코어 계산
        score = max(0.0, 100.0 * (1 - (test_mean / threshold)))

        print(f"테스트 데이터셋 평균 MSE: {test_mean:.6f}")
        print(f"훈련 임계값: {threshold:.6f}")
        print(f"정상도 점수: {score:.2f}%")
        verdict = "정상" if score >= 50 else "비정상"
        print(f"판정: {verdict}")
else:
    print("테스트 파일(test_main.csv)을 찾지 못했습니다. decoder/ 폴더에 두고 다시 실행하세요.")
