from pathlib import Path
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mocap.visualization import visualize_results as viz_results
from mocap.visualization import animate_3d_segments as viz_animate
from mocap.visualization import export_joint_map_figure as viz_export_joint_map

warnings.filterwarnings('ignore')


class MocapMotionAnalyzer:
    """
    모션 캡처 데이터를 사용하여 두 복싱 동작의 유사도를 정교하게 분석하는 클래스.

    v2.14 (안정화/정확도 강화)
    - CSV 로드 시 비수치→NaN→0 치환
    - 쿼터니언 이중 안전망(정규화/zero-norm 교정/부호 연속성 유지)
    - 좌표계 정렬 시 프레임별 quaternion 보정 + 위치/회전 모두 안전 처리
    - 속도/가속도 계산 시 길이/유효값 보장
    - 세그먼트 검출: 빈/짧은 구간 robust, (start, end) inclusive 반환
    - 스케일링: 'standard' | 'minmax' | None (공통 fit)
    - DTW: 입력 전처리(NaN/Inf→0), 1D/ND 자동, quaternion sign-invariant distance
    - 디버그 로그: NaN/Inf/zero-norm/길이 불일치 등 즉시 리포트
    """

    def __init__(self, scaling: str = 'standard', feature_weights: dict | None = None,
                 normalize_scale: bool = True, scale_mode: str = 'combined'):
        assert scaling in ('standard', 'minmax', None)
        self.scaling = scaling

        default_weights = {
            'position': 0.20,
            'rotation': 0.25,
            'velocity': 0.25,
            'acceleration': 0.10,
            'joint_angles': 0.20
        }
        allowed_keys = set(default_weights.keys())
        if feature_weights is not None:
            filtered = {k: float(v) for k, v in feature_weights.items() if k in allowed_keys}
            self.feature_weights = {**default_weights, **filtered}
        else:
            self.feature_weights = default_weights
        s = sum(self.feature_weights.values())
        if s <= 0:
            self.feature_weights = {k: 1.0 for k in self.feature_weights}
            s = sum(self.feature_weights.values())
        self.feature_weights = {k: v / s for k, v in self.feature_weights.items()}

        # 각도 키(누락되면 zero로 채움)
        self.angle_keys = [
            'torso_twist',
            'l_shoulder_angle', 'l_elbow_angle',
            'r_shoulder_angle', 'r_elbow_angle',
            'l_knee_angle', 'r_knee_angle',
            'l_ankle_angle', 'r_ankle_angle',
            'neck_flexion',
        ]

        # 스켈레톤 스케일 정규화 설정
        assert scale_mode in ('shoulder', 'torso', 'combined')
        self.normalize_scale = bool(normalize_scale)
        self.scale_mode = scale_mode

    def set_feature_weights(self, feature_weights: dict, normalize: bool = True):
        """특성 가중치를 동적으로 설정합니다."""
        allowed = {'position', 'rotation', 'velocity', 'acceleration', 'joint_angles'}
        if not feature_weights:
            return
        clean = {k: float(v) for k, v in feature_weights.items() if k in allowed}
        self.feature_weights.update(clean)
        if normalize:
            s = sum(self.feature_weights.values())
            if s <= 0:
                self.feature_weights = {k: 1.0 for k in self.feature_weights}
                s = sum(self.feature_weights.values())
            self.feature_weights = {k: v / s for k, v in self.feature_weights.items()}

    # ============================== I/O ==============================

    def load_mocap_data(self, file_path: str) -> pd.DataFrame | None:
        """모션캡처 CSV 데이터를 로드합니다. 비수치 → NaN → 0 처리."""
        try:
            df = pd.read_csv(file_path)
            # 공백 컬럼명 정리
            df = df.rename(columns=str.strip)
            # 전부 float로 시도, 실패값은 NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            # NaN은 0으로 치환
            n_nans = int(df.isna().sum().sum())
            if n_nans > 0:
                print(f"경고: '{file_path}' 내 비수치 또는 결측값 {n_nans}개를 0으로 대체했습니다.")
            df = df.fillna(0.0)
            print(f"파일 로드를 완료했습니다. 프레임 수: {df.shape[0]}, 컬럼 수: {df.shape[1]}")
            return df
        except Exception as e:
            print(f"데이터 로드 오류가 발생했습니다: {e}")
            return None

    # ======================= Finite/Shape Guards =====================

    def _safe_array(self, x, name="array"):
        """NaN/Inf → 0 치환 + 경고."""
        arr = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(arr)):
            n_bad = int(np.size(arr) - np.isfinite(arr).sum())
            print(f"경고: {name}에 NaN 또는 Inf 값 {n_bad}개를 0으로 치환했습니다.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _ensure_2d(self, x):
        """DTW 입력용: 1D → (T,1)로 변환, 그 외는 원형 유지."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    # ======================= Quaternion Utilities ====================

    def _clean_quaternions(self, quats, joint_name="Unknown"):
        """
        잘못된 쿼터니언(0-노름/NaN/Inf) → 단위 쿼터니언으로 대체,
        정규화 + 프레임 간 부호 연속성 유지.
        입력/출력: (T, 4) [x, y, z, w]
        """
        if quats is None or len(quats) == 0:
            return quats

        q = self._safe_array(quats, name=f"{joint_name}.rotation").astype(float)

        # 1) zero-norm → 단위쿼터니언 대체
        norms = np.linalg.norm(q, axis=1)
        zero_idx = np.where(norms < 1e-8)[0]
        if len(zero_idx) > 0:
            print(f"경고: '{joint_name}' 쿼터니언 zero-norm {len(zero_idx)}개를 [0,0,0,1]로 대체했습니다.")
            q[zero_idx] = np.array([0, 0, 0, 1], dtype=float)

        # 2) 정규화
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norms, 1e-8, None)

        # 3) 부호 연속성 (프레임 간 dot < 0 → 부호 반전)
        for i in range(1, len(q)):
            if float(np.dot(q[i - 1], q[i])) < 0.0:
                q[i] = -q[i]

        return q

    def _normalize_quat_sequence(self, quats, name="quat_seq"):
        """정규화만 수행(부호연속성은 외부에서 수행)."""
        if quats is None or len(quats) == 0:
            return quats
        q = self._safe_array(quats, name=name)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norms, 1e-8, None)
        return q

    # ===================== Coordinate Alignment =====================

    def _align_coordinate_system(self, features):
        """
        시작 방향이 달라도 동일한 조건에서 비교할 수 있도록 좌표계를 정렬합니다.
        - Hip/Chest 필요
        - 위치는 일괄 회전
        - 회전은 프레임별로 align_rotation과 합성
        """
        if 'Hip' not in features or 'Chest' not in features:
            return features

        hip_pos_initial = features['Hip'].get('position')
        chest_pos_initial = features['Chest'].get('position')
        if hip_pos_initial is None or chest_pos_initial is None:
            return features

        hip0 = np.asarray(hip_pos_initial[0], dtype=float)
        chest0 = np.asarray(chest_pos_initial[0], dtype=float)
        forward_vec = chest0 - hip0
        # 수평면 투영
        if len(forward_vec) >= 2:
            forward_vec[1] = 0.0

        norm = np.linalg.norm(forward_vec)
        if norm < 1e-6:
            # 정렬 불가 → 원본 유지
            return features
        forward_vec = forward_vec / norm

        target_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        # 회전축/각
        rotation_axis = np.cross(forward_vec, target_vec)
        axis_norm = np.linalg.norm(rotation_axis)
        if axis_norm < 1e-8:
            align_rotation = R.identity()
        else:
            rotation_axis = rotation_axis / axis_norm
            cosang = float(np.clip(np.dot(forward_vec, target_vec), -1.0, 1.0))
            rotation_angle = float(np.arccos(cosang))
            align_rotation = R.from_rotvec(rotation_angle * rotation_axis)

        # 적용
        for joint, jfeat in features.items():
            # 위치
            if 'position' in jfeat and jfeat['position'] is not None:
                pos = self._safe_array(jfeat['position'], name=f"{joint}.position")
                try:
                    jfeat['position'] = self._safe_array(
                        align_rotation.apply(pos),
                        name=f"{joint}.position_aligned"
                    )
                except Exception:
                    # shape 불일치 등 예외 발생 시 원본 유지
                    jfeat['position'] = pos

            # 회전 (프레임별 합성)
            if 'rotation' in jfeat and jfeat['rotation'] is not None:
                quats = self._clean_quaternions(jfeat['rotation'], joint_name=joint)
                fixed_quats = []
                for t in range(len(quats)):
                    q = quats[t]
                    if np.linalg.norm(q) < 1e-8:
                        q = np.array([0, 0, 0, 1], dtype=float)
                    try:
                        r = R.from_quat(q)
                        aligned_q = (align_rotation * r).as_quat()
                    except Exception:
                        aligned_q = np.array([0, 0, 0, 1], dtype=float)
                    fixed_quats.append(aligned_q)
                jfeat['rotation'] = self._safe_array(np.array(fixed_quats), name=f"{joint}.rotation_aligned")

        return features

    # =================== Skeleton Scale Normalization ===================

    def _estimate_body_scale(self, positions: dict[str, np.ndarray], mode: str = 'combined') -> float:
        """프레임 전반의 대표 스케일을 추정합니다. 반환값이 0이면 정규화를 수행하지 않습니다.

        mode:
          - 'shoulder': LShoulder–RShoulder 거리의 중앙값
          - 'torso': Hip–Chest 거리의 중앙값
          - 'combined': 사용 가능한 항목의 기하평균
        """
        def median_dist(a: str, b: str) -> float | None:
            if a in positions and b in positions and positions[a] is not None and positions[b] is not None:
                pa = np.asarray(positions[a], dtype=float)
                pb = np.asarray(positions[b], dtype=float)
                T = min(len(pa), len(pb))
                if T == 0:
                    return None
                d = np.linalg.norm(pb[:T] - pa[:T], axis=1)
                if d.size == 0:
                    return None
                med = float(np.median(d))
                return med if np.isfinite(med) and med > 1e-8 else None
            return None

        shoulder = median_dist('LShoulder', 'RShoulder')
        torso = median_dist('Hip', 'Chest')

        if mode == 'shoulder':
            return shoulder or 0.0
        if mode == 'torso':
            return torso or 0.0

        vals = [v for v in (shoulder, torso) if v is not None and v > 0]
        if not vals:
            return 0.0
        if len(vals) == 1:
            return vals[0]
        # 기하평균이 스케일 추정에 안정적
        logv = np.log(vals)
        return float(np.exp(np.mean(logv)))

    def _apply_scale_normalization(self, raw: dict, mode: str = 'combined') -> dict:
        """스켈레톤 위치(및 파생 속성에 앞서)를 스케일로 나눠 정규화합니다."""
        positions = {j: d.get('position') for j, d in raw.items() if 'position' in d}
        scale = self._estimate_body_scale(positions, mode=mode)
        if scale <= 0.0:
            return raw
        for j, d in raw.items():
            if 'position' in d and d['position'] is not None:
                d['position'] = self._safe_array(d['position'] / scale, name=f"{j}.pos_scaled")
        return raw

    def _rescale_target_to_reference(self, f_ref: dict, f_tgt: dict, mode: str = 'combined') -> tuple[dict, float]:
        """motion2(타깃)의 위치를 motion1(레퍼런스)의 체형 스케일에 맞춥니다.

        반환: (스케일 적용된 position 딕셔너리, 적용 배율 factor)
        """
        pos_ref = f_ref.get('position', {})
        pos_tgt = f_tgt.get('position', {})
        ref_positions = {j: arr for j, arr in pos_ref.items() if arr is not None}
        tgt_positions = {j: arr for j, arr in pos_tgt.items() if arr is not None}
        s_ref = self._estimate_body_scale(ref_positions, mode=mode)
        s_tgt = self._estimate_body_scale(tgt_positions, mode=mode)
        if s_ref <= 0.0 or s_tgt <= 0.0:
            return pos_tgt, 1.0
        factor = float(s_ref / s_tgt)
        scaled = {}
        for j, arr in pos_tgt.items():
            if arr is None:
                continue
            scaled[j] = self._safe_array(arr * factor, name=f"{j}.pos_refscaled")
        return scaled, factor

    def _compute_forward_from_features(self, f: dict) -> np.ndarray | None:
        pos = f.get('position', {})
        if 'Hip' not in pos or 'Chest' not in pos:
            return None
        hip = np.asarray(pos['Hip'][0], dtype=float)
        chest = np.asarray(pos['Chest'][0], dtype=float)
        v = chest - hip
        if v.shape[0] >= 2:
            v[1] = 0.0
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        return v / n

    def _rotation_from_to(self, src: np.ndarray, dst: np.ndarray) -> R:
        src = np.asarray(src, dtype=float)
        dst = np.asarray(dst, dtype=float)
        src = src / (np.linalg.norm(src) + 1e-8)
        dst = dst / (np.linalg.norm(dst) + 1e-8)
        axis = np.cross(src, dst)
        axis_n = np.linalg.norm(axis)
        if axis_n < 1e-8:
            return R.identity()
        axis = axis / axis_n
        cosang = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        return R.from_rotvec(angle * axis)

    def _apply_rotation_to_features(self, feats: dict, rot: R):
        # 위치/속도/가속도 회전
        for ftype in ('position', 'velocity', 'acceleration'):
            if ftype in feats:
                for j, arr in feats[ftype].items():
                    if arr is None:
                        continue
                    try:
                        feats[ftype][j] = self._safe_array(rot.apply(arr), name=f"{j}.{ftype}_rot")
                    except Exception:
                        feats[ftype][j] = arr
        # 회전(quaternion) 합성
        if 'rotation' in feats:
            for j, quats in feats['rotation'].items():
                if quats is None:
                    continue
                new_qs = []
                for q in quats:
                    try:
                        r = R.from_quat(q)
                        nq = (rot * r).as_quat()
                    except Exception:
                        nq = q
                    new_qs.append(nq)
                feats['rotation'][j] = self._safe_array(np.array(new_qs), name=f"{j}.rotation_rot")

    def _scale_vel_acc_features(self, feats: dict, factor: float):
        if abs(factor - 1.0) < 1e-12:
            return
        for ftype in ('velocity', 'acceleration'):
            if ftype in feats:
                for j, arr in feats[ftype].items():
                    if arr is None:
                        continue
                    feats[ftype][j] = self._safe_array(arr * factor, name=f"{j}.{ftype}_scaled")

    # =================== Per-Joint Feature Similarities ===================

    def _compute_per_joint_feature_similarities(self, f1: dict, f2: dict,
                                                s1: int, e1: int, s2: int, e2: int) -> dict:
        """각 특성별로 조인트(또는 각도 키) 단위의 DTW 유사도를 계산합니다.

        반환 형태: { feature_type: { key: similarity, ... }, ... }
        - feature_type ∈ {'position','rotation','velocity','acceleration','joint_angles'}
        - key: 조인트명 또는 각도 키명
        """
        result: dict[str, dict[str, float]] = {}
        for f_type in self.feature_weights.keys():
            d1 = f1.get(f_type, {})
            d2 = f2.get(f_type, {})
            per_keys: dict[str, float] = {}

            if f_type == 'joint_angles':
                keys = set(d1.keys()) & set(d2.keys())
                for key in keys:
                    seq1 = d1[key][s1:e1 + 1]
                    seq2 = d2[key][s2:e2 + 1]
                    if seq1.size == 0 or seq2.size == 0:
                        continue
                    per_keys[key] = self._dtw_similarity(seq1, seq2)
            else:
                keys = set(d1.keys()) & set(d2.keys())
                for key in keys:
                    seq1 = d1[key][s1:e1 + 1]
                    seq2 = d2[key][s2:e2 + 1]
                    if seq1.size == 0 or seq2.size == 0:
                        continue
                    if f_type == 'rotation':
                        seq1 = self._normalize_quat_sequence(seq1, name=f"{key}.rot1")
                        seq2 = self._normalize_quat_sequence(seq2, name=f"{key}.rot2")
                        for qarr in (seq1, seq2):
                            for i in range(1, len(qarr)):
                                if float(np.dot(qarr[i - 1], qarr[i])) < 0.0:
                                    qarr[i] = -qarr[i]
                        scaled_seq1, scaled_seq2 = seq1, seq2
                    else:
                        scaled_seq1, scaled_seq2 = (seq1, seq2) if self.scaling is None else self._fit_scaler(seq1, seq2)
                    per_keys[key] = self._dtw_similarity(scaled_seq1, scaled_seq2)

            result[f_type] = per_keys
        return result

    def export_per_joint_similarities(self, per_joint: dict, output_path: str):
        """per_joint 유사도 결과를 간단한 CSV로 저장합니다.

        CSV 컬럼: feature_type,key,similarity
        """
        import csv
        rows = []
        for f_type, mapping in per_joint.items():
            for key, sim in mapping.items():
                rows.append((f_type, key, float(sim)))
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['feature_type', 'key', 'similarity'])
                writer.writerows(rows)
            print(f"per-joint 유사도 결과를 저장했습니다: {output_path}")
        except Exception as e:
            print(f"per-joint 유사도 저장 중 오류가 발생했습니다: {e}")

    # =========================== Features ===========================

    def extract_features(self, df: pd.DataFrame):
        """종합적인 특성을 추출합니다."""
        print("종합 특성 추출을 시작합니다.")

        # joint 목록
        joints = sorted(set([c.split('.')[0] for c in df.columns if '.' in c]))

        raw = {}
        for joint in joints:
            pos_cols = [f'{joint}.posX', f'{joint}.posY', f'{joint}.posZ']
            rot_cols = [f'{joint}.rotX', f'{joint}.rotY', f'{joint}.rotZ', f'{joint}.rotW']

            jfeat = {}

            if all(c in df.columns for c in pos_cols):
                pos = df[pos_cols].to_numpy(dtype=float)
                jfeat['position'] = self._safe_array(pos, name=f"{joint}.pos")

            if all(c in df.columns for c in rot_cols):
                raw_quats = df[rot_cols].to_numpy(dtype=float)
                jfeat['rotation'] = self._clean_quaternions(raw_quats, joint_name=joint)

            if jfeat:
                raw[joint] = jfeat

        if 'Hip' not in raw or 'position' not in raw['Hip']:
            raise ValueError("'Hip' 조인트 position 데이터가 필요합니다.")

        # Hip 기준 중심화
        hip = raw['Hip']['position'].copy()
        for jfeat in raw.values():
            if 'position' in jfeat:
                jfeat['position'] = self._safe_array(jfeat['position'] - hip, name="centered_pos")

        # 좌표계 정렬
        raw = self._align_coordinate_system(raw)

        # 스켈레톤 스케일 정규화(옵션)
        if self.normalize_scale:
            raw = self._apply_scale_normalization(raw, mode=self.scale_mode)

        # 속도/가속도 계산
        for j, jfeat in raw.items():
            if 'position' in jfeat and jfeat['position'] is not None and len(jfeat['position']) > 0:
                pos = self._safe_array(jfeat['position'], name=f"{j}.pos_for_vel")
                vel = np.diff(pos, axis=0, prepend=pos[0:1])
                acc = np.diff(vel, axis=0, prepend=vel[0:1])
                jfeat['velocity'] = self._safe_array(vel, name=f"{j}.vel")
                jfeat['acceleration'] = self._safe_array(acc, name=f"{j}.acc")

        # 벡터/각도
        joint_positions = {j: d['position'] for j, d in raw.items() if 'position' in d}
        pairs = [
            ('LShoulder', 'RShoulder'), ('LThigh', 'RThigh'),
            ('Chest', 'Ab'),
            ('LShoulder', 'LUArm'), ('RShoulder', 'RUArm'),
            ('LUArm', 'LFArm'), ('RUArm', 'RFArm'),
            # Legs
            ('LThigh', 'LShin'), ('LShin', 'LFoot'), ('LFoot', 'LToe'),
            ('RThigh', 'RShin'), ('RShin', 'RFoot'), ('RFoot', 'RToe'),
            # Neck/Head
            ('Chest', 'Neck'), ('Neck', 'Head'),
        ]
        vectors = {}
        for a, b in pairs:
            vec = self._calculate_vector(joint_positions, a, b)
            if vec is not None:
                vectors[f"{a}_{b}"] = vec

        T = len(next(iter(joint_positions.values()))) if len(joint_positions) else len(df)

        def safe_angle(vkey1, vkey2):
            if vkey1 in vectors and vkey2 in vectors:
                return self._calculate_angle(vectors[vkey1], vectors[vkey2])
            return np.zeros(T, dtype=float)

        def safe_signed_angle(vkey1, vkey2, normal=(0.0, 1.0, 0.0)):
            if vkey1 in vectors and vkey2 in vectors:
                return self._signed_angle(vectors[vkey1], vectors[vkey2], normal=normal)
            return np.zeros(T, dtype=float)

        joint_angles = {
            # 수평면(Y 법선) 기준으로 좌/우 비틀림 부호를 가진 각도
            'torso_twist': safe_signed_angle('LShoulder_RShoulder', 'LThigh_RThigh', normal=(0.0, 1.0, 0.0)),
            'l_shoulder_angle': safe_angle('Chest_Ab', 'LShoulder_LUArm'),
            'l_elbow_angle': safe_angle('LShoulder_LUArm', 'LUArm_LFArm'),
            'r_shoulder_angle': safe_angle('Chest_Ab', 'RShoulder_RUArm'),
            'r_elbow_angle': safe_angle('RShoulder_RUArm', 'RUArm_RFArm'),
            # Legs
            'l_knee_angle': safe_angle('LThigh_LShin', 'LShin_LFoot'),
            'r_knee_angle': safe_angle('RThigh_RShin', 'RShin_RFoot'),
            'l_ankle_angle': safe_angle('LShin_LFoot', 'LFoot_LToe'),
            'r_ankle_angle': safe_angle('RShin_RFoot', 'RFoot_RToe'),
            # Head/neck
            'neck_flexion': safe_angle('Chest_Neck', 'Neck_Head'),
        }

        # 최종 딕셔너리 구성
        feats = {'position': {}, 'rotation': {}, 'velocity': {}, 'acceleration': {}, 'joint_angles': {}}
        for joint, jfeat in raw.items():
            for ftype in ('position', 'rotation', 'velocity', 'acceleration'):
                if ftype in jfeat and jfeat[ftype] is not None:
                    feats[ftype][joint] = self._safe_array(jfeat[ftype], name=f"{joint}.{ftype}")

        feats['joint_angles'] = {k: self._safe_array(joint_angles.get(k, np.zeros(T)), name=f"angle.{k}") for k in self.angle_keys}

        print("특성 추출을 완료했습니다.")
        return feats

    # ======================= Motion Segmentation ====================

    def segment_action(self, features, threshold_ratio=0.15, min_length=30):
        """
        속도를 기반으로 동작의 핵심 구간을 탐지합니다.
        반환: (start_idx, end_idx)  (end_idx 포함)
        """
        vlist = [np.linalg.norm(v, axis=1) for v in features['velocity'].values() if v is not None and len(v) > 0]
        if not vlist:
            any_len = len(next(iter(features['position'].values()))) if features['position'] else 0
            return 0, max(any_len - 1, 0)

        total_velocity = np.sum(vlist, axis=0)
        total_velocity = self._safe_array(total_velocity, name="total_velocity")

        if total_velocity.size == 0:
            return 0, 0

        thr = float(np.max(total_velocity)) * float(threshold_ratio)
        active = np.where(total_velocity > thr)[0]

        if active.size < min_length:
            return 0, len(total_velocity) - 1

        start_idx, end_idx = int(active[0]), int(active[-1])
        print(f"동작 구간을 탐지했습니다. 시작 프레임: {start_idx}, 종료 프레임: {end_idx}, 총 길이: {end_idx - start_idx + 1} 프레임")
        return start_idx, end_idx

    # ============================= Math =============================

    def _calculate_angle(self, v1, v2):
        v1 = self._safe_array(v1, name="angle_v1")
        v2 = self._safe_array(v2, name="angle_v2")
        v1_u = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        dot = np.einsum('ij,ij->i', v1_u, v2_u)
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)

    def _signed_angle(self, v1, v2, normal=(0.0, 1.0, 0.0)):
        """평면 법선 normal에 대한 서명 각도(라디안, -pi~pi).
        v1, v2는 (T,3). 수평면(XZ) 서명 각도의 경우 normal=(0,1,0).
        """
        v1 = self._safe_array(v1, name="signed_angle_v1")
        v2 = self._safe_array(v2, name="signed_angle_v2")
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-8)
        # 평면 투영
        v1p = v1 - np.einsum('ij,j->i', v1, n)[:, None] * n[None, :]
        v2p = v2 - np.einsum('ij,j->i', v2, n)[:, None] * n[None, :]
        # 정규화
        v1p = v1p / (np.linalg.norm(v1p, axis=1, keepdims=True) + 1e-8)
        v2p = v2p / (np.linalg.norm(v2p, axis=1, keepdims=True) + 1e-8)
        # dot, cross
        dot = np.einsum('ij,ij->i', v1p, v2p)
        dot = np.clip(dot, -1.0, 1.0)
        cross = np.cross(v1p, v2p)
        s = np.einsum('ij,j->i', cross, n)
        return np.arctan2(s, dot)

    def _calculate_vector(self, positions, start_joint, end_joint):
        if start_joint in positions and end_joint in positions:
            a = positions[start_joint]
            b = positions[end_joint]
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                return None
            T = min(len(a), len(b))
            a = np.asarray(a[:T], dtype=float)
            b = np.asarray(b[:T], dtype=float)
            return self._safe_array(b - a, name=f"vec.{start_joint}_{end_joint}")
        return None

    # ======================= Scaling & DTW ==========================

    def _fit_scaler(self, seq1, seq2):
        """선택된 스케일러를 두 시퀀스에 공통으로 피팅/적용."""
        if self.scaling is None:
            return seq1, seq2

        _s1 = self._ensure_2d(seq1)
        _s2 = self._ensure_2d(seq2)
        if _s1.shape[1] != _s2.shape[1]:
            return seq1, seq2  # 차원 불일치 시 스킵

        if self.scaling == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        both = np.vstack([_s1, _s2])
        both = self._safe_array(both, name="scaler_fit_input")
        scaler.fit(both)

        s1 = scaler.transform(_s1)
        s2 = scaler.transform(_s2)
        # 원래 차원 유지(1D였다면 다시 1D로)
        if seq1.ndim == 1: s1 = s1.flatten()
        if seq2.ndim == 1: s2 = s2.flatten()
        return s1, s2

    def _dtw_similarity(self, seq1, seq2, k=10):
        """DTW 거리를 정규화하고 지수 함수로 유사도로 변환."""
        if not isinstance(seq1, np.ndarray): seq1 = np.array(seq1, dtype=float)
        if not isinstance(seq2, np.ndarray): seq2 = np.array(seq2, dtype=float)
        if seq1.size == 0 or seq2.size == 0: return 0.0

        # 최종 안전화
        seq1 = self._safe_array(seq1, name="dtw_seq1")
        seq2 = self._safe_array(seq2, name="dtw_seq2")

        # 거리 함수 선택
        if seq1.ndim == 1:
            dist_func = lambda x, y: float(abs(x - y))
        elif seq1.shape[1] == 4:
            # 쿼터니언 sign-invariant
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

    # ============================ Compare ===========================

    def compare_motions(self, motion1_data: pd.DataFrame, motion2_data: pd.DataFrame):
        """두 동작을 종합적으로 비교합니다."""
        print("\n두 동작의 비교를 시작합니다.")
        try:
            f1 = self.extract_features(motion1_data)
            f2 = self.extract_features(motion2_data)

            # 1) motion1 기준으로 motion2의 체형 스케일을 일치
            if 'position' in f1 and 'position' in f2:
                rescaled_pos2, factor = self._rescale_target_to_reference(f1, f2, mode=self.scale_mode)
                if rescaled_pos2:
                    f2['position'].update(rescaled_pos2)
                    # 속도/가속도도 동일 배율 반영(위치 차분 기반이므로 동일 스케일 필요)
                    self._scale_vel_acc_features(f2, factor)

            # 2) motion1의 전방을 기준으로 motion2 방향을 일치(좌표계 회전)
            fwd1 = self._compute_forward_from_features(f1)
            fwd2 = self._compute_forward_from_features(f2)
            if fwd1 is not None and fwd2 is not None:
                rot = self._rotation_from_to(fwd2, fwd1)  # motion2 -> motion1 방향으로 회전
                self._apply_rotation_to_features(f2, rot)

            s1, e1 = self.segment_action(f1)
            s2, e2 = self.segment_action(f2)

            feature_similarities = {}

            for f_type in self.feature_weights.keys():
                print(f"특성 '{f_type}' 유사도 계산을 수행합니다.")
                sims = []

                d1 = f1[f_type]
                d2 = f2[f_type]

                if f_type == 'joint_angles':
                    keys = set(d1.keys()) & set(d2.keys())
                    for key in keys:
                        seq1 = d1[key][s1:e1 + 1]
                        seq2 = d2[key][s2:e2 + 1]
                        if seq1.size == 0 or seq2.size == 0:
                            continue
                        # 각도는 일반적으로 스케일링 불필요
                        sim = self._dtw_similarity(seq1, seq2)
                        sims.append(sim)
                else:
                    keys = set(d1.keys()) & set(d2.keys())
                    for key in keys:
                        seq1 = d1[key][s1:e1 + 1]
                        seq2 = d2[key][s2:e2 + 1]
                        if seq1.size == 0 or seq2.size == 0:
                            continue

                        if f_type == 'rotation':
                            # 정규화 + 부호연속성
                            seq1 = self._normalize_quat_sequence(seq1, name=f"{key}.rot1")
                            seq2 = self._normalize_quat_sequence(seq2, name=f"{key}.rot2")
                            for qarr in (seq1, seq2):
                                for i in range(1, len(qarr)):
                                    if float(np.dot(qarr[i - 1], qarr[i])) < 0.0:
                                        qarr[i] = -qarr[i]
                            scaled_seq1, scaled_seq2 = seq1, seq2
                        else:
                            # 위치/속도/가속도
                            scaled_seq1, scaled_seq2 = (seq1, seq2) if self.scaling is None else self._fit_scaler(seq1, seq2)

                        sim = self._dtw_similarity(scaled_seq1, scaled_seq2)
                        sims.append(sim)

                feature_similarities[f_type] = float(np.mean(sims)) if sims else 0.0

            # 파트(왼팔/오른팔/왼다리/오른다리/코어/머리) 단위 DTW 산출
            parts = {
                'left_arm': ['LShoulder', 'LUArm', 'LFArm', 'LHand'],
                'right_arm': ['RShoulder', 'RUArm', 'RFArm', 'RHand'],
                'left_leg': ['LThigh', 'LShin', 'LFoot', 'LToe'],
                'right_leg': ['RThigh', 'RShin', 'RFoot', 'RToe'],
                'core': ['LThigh', 'RThigh', 'Hip', 'Ab', 'Chest'],
                'head': ['LShoulder', 'RShoulder', 'Neck', 'Head'],
            }

            # 파트별 각도 키 매핑(해당 파트에 의미 있는 각도만 사용)
            angle_keys_by_part: dict[str, set[str]] = {
                'left_arm': {'l_shoulder_angle', 'l_elbow_angle'},
                'right_arm': {'r_shoulder_angle', 'r_elbow_angle'},
                'left_leg': {'l_knee_angle', 'l_ankle_angle'},
                'right_leg': {'r_knee_angle', 'r_ankle_angle'},
                'core': {'torso_twist'},
                'head': {'neck_flexion'},
            }

            part_scores = {}
            part_breakdown = {}
            # 피처별 파트 스코어 계산 함수
            def part_feature_score(feature_type: str, joints_set: set[str], part_name: str) -> float:
                d1 = f1.get(feature_type, {})
                d2 = f2.get(feature_type, {})
                sims_local = []
                if feature_type == 'joint_angles':
                    # 파트에 해당하는 각도 키만 사용
                    allowed = angle_keys_by_part.get(part_name, set())
                    keys = (set(d1.keys()) & set(d2.keys())) & allowed
                    for key in keys:
                        seq1 = d1[key][s1:e1 + 1]
                        seq2 = d2[key][s2:e2 + 1]
                        if seq1.size == 0 or seq2.size == 0:
                            continue
                        sims_local.append(self._dtw_similarity(seq1, seq2))
                else:
                    keys = (set(d1.keys()) & set(d2.keys())) & joints_set
                    for key in keys:
                        seq1 = d1[key][s1:e1 + 1]
                        seq2 = d2[key][s2:e2 + 1]
                        if seq1.size == 0 or seq2.size == 0:
                            continue
                        if feature_type == 'rotation':
                            seq1 = self._normalize_quat_sequence(seq1, name=f"{key}.rot1")
                            seq2 = self._normalize_quat_sequence(seq2, name=f"{key}.rot2")
                            for qarr in (seq1, seq2):
                                for i in range(1, len(qarr)):
                                    if float(np.dot(qarr[i - 1], qarr[i])) < 0.0:
                                        qarr[i] = -qarr[i]
                            scaled_seq1, scaled_seq2 = seq1, seq2
                        else:
                            scaled_seq1, scaled_seq2 = (seq1, seq2) if self.scaling is None else self._fit_scaler(seq1, seq2)
                        sims_local.append(self._dtw_similarity(scaled_seq1, scaled_seq2))
                return float(np.mean(sims_local)) if sims_local else 0.0

            for part_name, part_joints in parts.items():
                jset = set(part_joints)
                # 모든 피처를 고려한 파트 스코어(동일 가중 평균)
                per_feature_scores = {}
                for f_type in ('position', 'rotation', 'velocity', 'acceleration', 'joint_angles'):
                    per_feature_scores[f_type] = part_feature_score(f_type, jset, part_name)
                part_breakdown[part_name] = per_feature_scores
                vals = list(per_feature_scores.values())
                part_scores[part_name] = float(np.mean(vals)) if vals else 0.0

            # 조인트 단위 유사도(전 특성) 산출 및 내보내기 용도로 함께 반환
            per_joint = self._compute_per_joint_feature_similarities(f1, f2, s1, e1, s2, e2)

            # 가중 합 (전체)
            final_similarity = 0.0
            for f_type, sim in feature_similarities.items():
                final_similarity += sim * self.feature_weights.get(f_type, 0.0)

            return float(final_similarity), {
                **feature_similarities,
                **{f"part_{k}": v for k, v in part_scores.items()},
                'per_joint': per_joint,
                'part_breakdown': part_breakdown,
            }

        except Exception as e:
            import traceback
            print(f"동작 비교 중 오류가 발생했습니다: {e}")
            traceback.print_exc()
            return 0.0, {}

    # =========================== Visualization ======================

    def visualize_results(self, similarity, feature_similarities):
        viz_results(similarity, feature_similarities, self.feature_weights)

    def animate_3d_segments(self, motion1_data: pd.DataFrame, motion2_data: pd.DataFrame,
                            overlay: bool = True, interval: int = 40,
                            save_path: str | None = None,
                            joints_to_show: list[str] | None = None,
                            skeleton_edges: list[tuple[str, str]] | None = None):
        viz_animate(self, motion1_data, motion2_data, overlay, interval, save_path, joints_to_show, skeleton_edges)

    def export_joint_map_figure(self, output_path: str, dpi: int = 200, language: str = 'ko'):
        viz_export_joint_map(output_path, dpi, language)


# 유사도를 한번에 여러 파일과 비교하는 배치 함수

def print_similarity_batch(file1_path: str, file2_dir: str, analyzer: MocapMotionAnalyzer,
                           keyword: str | None = None, limit: int | None = None):
    """
    file2_dir 안의 모든 CSV(선택적으로 keyword 필터)를 file1과 비교하고,
    전체 유사도와 주요 항목을 콘솔에 깔끔하게 출력합니다.
    """
    file1 = Path(file1_path)
    dir2 = Path(file2_dir)

    if not file1.exists():
        print(f"[오류] file1이 존재하지 않습니다: {file1}")
        return
    if not dir2.exists() or not dir2.is_dir():
        print(f"[오류] file2 디렉터리가 존재하지 않습니다: {dir2}")
        return

    # file1은 한 번만 로드
    motion1 = analyzer.load_mocap_data(str(file1))
    if motion1 is None:
        print("[오류] file1 로드 실패로 배치 비교를 종료합니다.")
        return

    # 대상 파일 수집
    candidates = sorted([p for p in dir2.glob("*.csv") if p.is_file()])
    if keyword:
        candidates = [p for p in candidates if keyword.lower() in p.name.lower()]
    # file1이 같은 디렉토리에 있어도 제외
    candidates = [p for p in candidates if p.resolve() != file1.resolve()]
    if limit is not None:
        candidates = candidates[:limit]

    if not candidates:
        msg = f"'{dir2}'에서 비교할 CSV가 없습니다."
        if keyword:
            msg += f" (키워드='{keyword}')"
        print(msg)
        return

    print("\n" + "=" * 72)
    print(f"[배치 비교 시작] 기준 파일: {file1.name}  |  대상 디렉터리: {dir2}")
    if keyword:
        print(f"키워드 필터: {keyword}")
    print(f"총 대상 파일 수: {len(candidates)}")
    print("=" * 72)

    for idx, p in enumerate(candidates, start=1):
        print(f"\n[{idx}/{len(candidates)}] 비교 대상: {p.name}")
        motion2 = analyzer.load_mocap_data(str(p))
        if motion2 is None:
            print(" → 로드 실패, 건너뜁니다.")
            continue

        # 비교
        similarity, details = analyzer.compare_motions(motion1, motion2)

        # 출력(간단/명료)
        print(f" → 전체 유사도: {similarity:.4f}")
        # 주요 피처별 유사도만 골라 간단 표기
        for key in ('rotation', 'joint_angles', 'position', 'velocity', 'acceleration'):
            if key in details:
                print(f"    - {key:13s}: {details[key]:.4f}")
        # 파트 요약(원하면 주석 해제)
        for part in ('part_left_arm','part_right_arm','part_left_leg','part_right_leg','part_core','part_head'):
            if part in details:
                print(f"    - {part:13s}: {details[part]:.4f}")

    print("\n" + "=" * 72)
    print("[배치 비교 완료]")
    print("=" * 72)
    
    # utils/save_similarity_matrix.py  (새 파일로 두거나, 기존 파일 하단에 추가해도 됩니다)

import csv
from pathlib import Path

# === 새 함수: 배치 결과를 CSV로 저장 ===
def save_similarity_matrix(
    file1_path: str,
    file2_dir: str,
    analyzer: MocapMotionAnalyzer,
    keyword: str | None = None,
    limit: int | None = None,
    title: str = "Uppercut(L)",
    output_csv_path: str = "C:\\Users\\harry\\OneDrive\\Desktop\\DTW_Method\\Collaborate_Code\\similarity_matrix.csv"
) -> pd.DataFrame:
    """
    file2_dir의 모든 CSV(선택적으로 keyword 필터)를 file1과 비교해
    '부위별+피처별 유사도'를 한 번에 CSV로 저장합니다.

    열 순서(고정):
    Head, Core, Right_Leg, Left_Leg, Right_Arm, Left_Arm,
    Acceleration, Velocity, Position, Joint Angle, rotation
    """
    # 고정 열 이름(이미지 순서 그대로)
    col_order = [
        "Head", "Core", "Right_Leg", "Left_Leg", "Right_Arm", "Left_Arm",
        "Acceleration", "Velocity", "Position", "Joint Angle", "rotation", "Overall"
    ]

    # 내부 키 매핑( compare_motions details → 표의 열 )
    key_map = {
        "Head":         "part_head",
        "Core":         "part_core",
        "Right_Leg":    "part_right_leg",
        "Left_Leg":     "part_left_leg",
        "Right_Arm":    "part_right_arm",
        "Left_Arm":     "part_left_arm",
        "Acceleration": "acceleration",
        "Velocity":     "velocity",
        "Position":     "position",
        "Joint Angle":  "joint_angles",
        "rotation":     "rotation",
    }

    file1 = Path(file1_path)
    dir2  = Path(file2_dir)

    if not file1.exists():
        print(f"[오류] 기준 파일이 존재하지 않습니다: {file1}")
        return pd.DataFrame()

    if not dir2.exists() or not dir2.is_dir():
        print(f"[오류] 대상 디렉터리가 없습니다: {dir2}")
        return pd.DataFrame()

    # 기준 모션 1회 로드
    motion1 = analyzer.load_mocap_data(str(file1))
    if motion1 is None:
        print("[오류] 기준 파일 로드 실패")
        return pd.DataFrame()

    # 후보 수집
    candidates = sorted([p for p in dir2.glob("*.csv") if p.is_file()])
    if keyword:
        candidates = [p for p in candidates if keyword.lower() in p.name.lower()]
    candidates = [p for p in candidates if p.resolve() != file1.resolve()]
    if limit is not None:
        candidates = candidates[:limit]

    if not candidates:
        print(f"[안내] 비교할 CSV가 없습니다. dir={dir2}, keyword={keyword}")
        return pd.DataFrame()

    # 결과 누적
    rows = []
    index_labels = []

    for idx, p in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] 비교: {p.name}")  # 존댓말 로그
        motion2 = analyzer.load_mocap_data(str(p))
        if motion2 is None:
            print(" → 로드 실패, 건너뜁니다.")
            continue

        similarity, details = analyzer.compare_motions(motion1, motion2)

        # 한 행 구성(없으면 0.0)
        row = []
        for col in col_order:
            if col == "Overall":
                v = float(similarity)
            else:   
                v = float(details.get(key_map[col], 0.0))
            row.append(v)

        rows.append(row)
        # 행 라벨: 파일명에서 확장자 제거
        index_labels.append(p.stem)

    # DataFrame 구성(제목 열을 맨 앞에 추가)
    df = pd.DataFrame(rows, index=index_labels, columns=col_order)
    df.insert(0, title, index_labels)

    # 평균 행 추가(제목 칸은 'AVG')
    if len(df) > 0:
        avg_vals = df[col_order].mean(axis=0).to_list()
        avg_row = pd.DataFrame([[ "AVG", *avg_vals ]], columns=[title, *col_order])
        df = pd.concat([df, avg_row], ignore_index=True)

    # 저장
    try:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[완료] 유사도 매트릭스를 CSV로 저장했습니다: {output_csv_path}")
    except Exception as e:
        print(f"[오류] CSV 저장 중 문제 발생: {e}")

    return df


# =============================== Main ==============================

if __name__ == "__main__":
    print("복싱 동작 DTW 분석기 (v2.14 - 안정성/정확도 강화)")
    print("=" * 64)

    # uppercut_left_001.csv
    # uppercut_right_002.csv
    # hook_left_002.csv
    # hook_right_001.csv
    # jap_001.csv
    # straight_003.csv

    #file1 = "/Users/jonabi/Downloads/TEPA/mocap_test/uppercut_right_004.csv"
    #file2 = "/Users/jonabi/Downloads/TEPA/p24_Global"
    
    # 윈도우 기준 
    file1 = "C:\\Users\\PC\\Documents\\GitHub\\Collaborate_Code\\mocap_test\\uppercut_left_001.csv"
    file2 = "C:\\Users\\PC\\Documents\\GitHub\\Collaborate_Code\\p24_Global"
 
    #file1 = "C:\\Users\\user\\Downloads\\TEPA\\Collaborate_Code\\mocap_test\\jap_005.csv"
    #file2 = "C:\\Users\\user\\Downloads\\TEPA\\Collaborate_Code\\p26_Global"
 
    # 실행 중 어떤 파일을 비교하는지 표시
    # print(f"분석 대상 파일 1: {file1}")
    # print(f"분석 대상 파일 2: {file2}")
 
    # 가중치 사용자 정의 예시 (필요 시 수정)
    custom_feature_weights = {
        'position': 0.0,
        'rotation': 0.7,
        'velocity': 0.0,
        'acceleration': 0.0,
        'joint_angles': 0.3,
    }

    analyzer = MocapMotionAnalyzer(scaling='standard', feature_weights=custom_feature_weights)  
    
    ### ====> 추가한 부분.
    
    # 👉 배치 비교 실행 (출력만)
    #  - keyword: 특정 단어가 파일명에 포함된 것만 비교하고 싶으면 넣기 (예: "post" 또는 "hook_left")
    #  - limit: 상위 N개만 테스트하고 싶으면 숫자 지정
    # print_similarity_batch(
    #     file1_path=file1,
    #     file2_dir=file2,
    #     analyzer=analyzer,
    #     keyword="uppercut_left",   # 예: "post" 또는 None
    #     limit=None      # 예: 10 또는 None
    # )
    
    _ = save_similarity_matrix(
        file1_path=file1,
        file2_dir=file2,
        analyzer=analyzer,
        keyword="uppercut_right",      # 필요 시 수정
        limit=None,                   # 필요 시 숫자
        title="uppercut_left",          # 시트 좌측 첫 열 제목
        output_csv_path="p23_uppercut_left_004_similarity_matrix.csv"      # 시트 좌측 첫 열 제목
    )
    
    # 'standard' | 'minmax' | None
    # motion1 = analyzer.load_mocap_data(file1)
    # motion2 = analyzer.load_mocap_data(file2)
    # if motion1 is not None and motion2 is not None:
    #     similarity, details = analyzer.compare_motions(motion1, motion2)
    #     analyzer.visualize_results(similarity, details)
    #     analyzer.animate_3d_segments(motion1, motion2, save_path="output.gif")
    # else:
    #     print("파일 로드에 실패하여 분석을 진행할 수 없습니다.")