# motion_analyzer.py
# This script contains the core functions for boxing motion analysis.
# It now includes segmentation to compare only core actions (punches).

from typing import Tuple, Literal, Callable, Dict, List
import numpy as np
import pandas as pd

Plane = Literal['xy', 'xz', 'yz', 'xyz']

# -----------------------------
# Segmentation
# -----------------------------

def segment_core_action(df: pd.DataFrame, hand: Literal['L', 'R'] = 'L', velocity_threshold: float = 0.02, min_duration: int = 15) -> pd.DataFrame:
    """
    Detects and extracts core action segments (punches) from the motion data.
    A punch is identified by the forward velocity of the hand.

    Args:
        df (pd.DataFrame): The input motion dataframe.
        hand (Literal['L', 'R']): The hand to track for punches ('L' for left, 'R' for right).
        velocity_threshold (float): The forward velocity (Z-axis) threshold to trigger punch detection.
        min_duration (int): The minimum number of frames for a segment to be considered a valid punch.

    Returns:
        pd.DataFrame: A concatenated dataframe containing only the detected punch segments.
    """
    pos_z_col = f'{hand}Hand.posZ'
    if pos_z_col not in df.columns:
        raise ValueError(f"Column '{pos_z_col}' not found for segmentation.")

    # Calculate forward velocity (change in Z position)
    velocity = df[pos_z_col].diff().rolling(window=3, min_periods=1).mean()

    # Identify frames where the hand is moving forward above the threshold
    is_punching = (velocity > velocity_threshold)
    
    # --- FIX: Implemented a more stable method to find segment edges that avoids FutureWarnings ---
    # Find start points (where False changes to True)
    prev_is_punching = is_punching.shift(1)
    prev_is_punching.iloc[0] = False
    start_mask = is_punching & ~prev_is_punching.astype(bool)
    start_indices = df.index[start_mask]

    # Find end points (where True changes to False)
    next_is_punching = is_punching.shift(-1)
    next_is_punching.iloc[-1] = False
    end_mask = is_punching & ~next_is_punching.astype(bool)
    end_indices = df.index[end_mask]

    segments = []
    for start in start_indices:
        # Find the corresponding end for this start
        end = next((e for e in end_indices if e >= start), None)
        if end is not None:
            # Extend the segment slightly to capture the full motion
            extended_start = max(0, start - 5)
            extended_end = min(len(df) - 1, end + 5)
            segment = df.iloc[extended_start:extended_end]
            
            if len(segment) >= min_duration:
                segments.append(segment)

    if not segments:
        # Return an empty DataFrame with the same columns if no punches are detected
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(segments)


# -----------------------------
# Utilities (DTW, Validation)
# -----------------------------

def _dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Na, Nb = len(a), len(b)
    if Na == 0 or Nb == 0: return np.inf
    D = np.full((Na + 1, Nb + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Na + 1):
        for j in range(1, Nb + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = Na, Nb
    L = 0
    while i > 0 or j > 0:
        L += 1
        choices = [
            (D[i - 1, j] if i > 0 else np.inf, i - 1, j),
            (D[i, j - 1] if j > 0 else np.inf, i, j - 1),
            (D[i - 1, j - 1] if (i > 0 and j > 0) else np.inf, i - 1, j - 1),
        ]
        _, i, j = min(choices, key=lambda t: t[0])
    return float(D[Na, Nb] / max(L, 1))

def _get_validated_data(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:6]}")
    return df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# --- Metrics (Unchanged, but will now operate on segmented data) ---

def foot_to_shoulder_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    required = ['LFoot.posX', 'LFoot.posZ', 'RFoot.posX', 'RFoot.posZ', 'LShoulder.posX', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posZ']
    df_local = _get_validated_data(df, required)
    LFoot = df_local[['LFoot.posX', 'LFoot.posZ']].to_numpy()
    RFoot = df_local[['RFoot.posX', 'RFoot.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posZ']].to_numpy()
    d_feet = np.linalg.norm(LFoot - RFoot, axis=1)
    d_shoulders = np.linalg.norm(LShoulder - RShoulder, axis=1)
    return np.divide(d_feet, d_shoulders, out=np.zeros_like(d_feet), where=d_shoulders > 1e-8)

def stance_depth_width_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    required = ['LFoot.posX', 'LFoot.posZ', 'RFoot.posX', 'RFoot.posZ']
    df_local = _get_validated_data(df, required)
    d_width = np.abs(df_local['LFoot.posX'] - df_local['RFoot.posX'])
    d_depth = np.abs(df_local['LFoot.posZ'] - df_local['RFoot.posZ'])
    return np.divide(d_depth, d_width, out=np.zeros_like(d_depth), where=d_width > 1e-8)

def left_guard_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    required = ['LHand.posY', 'LShoulder.posY', 'Head.posY']
    df_local = _get_validated_data(df, required)
    dist_head_hand = np.abs(df_local['Head.posY'] - df_local['LHand.posY'])
    dist_head_shoulder = np.abs(df_local['Head.posY'] - df_local['LShoulder.posY'])
    return np.divide(dist_head_hand, dist_head_shoulder, out=np.zeros_like(dist_head_hand), where=dist_head_shoulder > 1e-8)

def right_guard_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    required = ['RHand.posY', 'RShoulder.posY', 'Head.posY']
    df_local = _get_validated_data(df, required)
    dist_head_hand = np.abs(df_local['Head.posY'] - df_local['RHand.posY'])
    dist_head_shoulder = np.abs(df_local['Head.posY'] - df_local['RShoulder.posY'])
    return np.divide(dist_head_hand, dist_head_shoulder, out=np.zeros_like(dist_head_hand), where=dist_head_shoulder > 1e-8)

def left_hand_reach_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    required = ['LHand.posX', 'LHand.posY', 'LHand.posZ', 'LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']
    df_local = _get_validated_data(df, required)
    Hand = df_local[['LHand.posX', 'LHand.posY', 'LHand.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']].to_numpy()
    dist_shoulder_hand = np.linalg.norm(LShoulder - Hand, axis=1)
    dist_shoulders = np.linalg.norm(LShoulder - RShoulder, axis=1)
    return np.divide(dist_shoulder_hand, dist_shoulders, out=np.zeros_like(dist_shoulder_hand), where=dist_shoulders > 1e-8)

def right_hand_reach_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    required = ['RHand.posX', 'RHand.posY', 'RHand.posZ', 'LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']
    df_local = _get_validated_data(df, required)
    Hand = df_local[['RHand.posX', 'RHand.posY', 'RHand.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posY', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posY', 'RShoulder.posZ']].to_numpy()
    dist_shoulder_hand = np.linalg.norm(RShoulder - Hand, axis=1)
    dist_shoulders = np.linalg.norm(LShoulder - RShoulder, axis=1)
    return np.divide(dist_shoulder_hand, dist_shoulders, out=np.zeros_like(dist_shoulder_hand), where=dist_shoulders > 1e-8)

def torso_rotation_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    required = ['LShoulder.posX', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posZ', 'LThigh.posX', 'LThigh.posZ', 'RThigh.posX', 'RThigh.posZ']
    df_local = _get_validated_data(df, required)
    shoulder_vec = df_local[['RShoulder.posX', 'RShoulder.posZ']].to_numpy() - df_local[['LShoulder.posX', 'LShoulder.posZ']].to_numpy()
    hip_vec = df_local[['RThigh.posX', 'RThigh.posZ']].to_numpy() - df_local[['LThigh.posX', 'LThigh.posZ']].to_numpy()
    dot_product = np.einsum('ij,ij->i', shoulder_vec, hip_vec)
    norm_shoulder = np.linalg.norm(shoulder_vec, axis=1)
    norm_hip = np.linalg.norm(hip_vec, axis=1)
    cos_angle = np.clip(dot_product / (norm_shoulder * norm_hip + 1e-8), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad) / 90.0

def com_shift_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    required = ['Hip.posX', 'Hip.posZ', 'LFoot.posX', 'LFoot.posZ', 'RFoot.posX', 'RFoot.posZ']
    df_local = _get_validated_data(df, required)
    Hips = df_local[['Hip.posX', 'Hip.posZ']].to_numpy()
    LFoot = df_local[['LFoot.posX', 'LFoot.posZ']].to_numpy()
    RFoot = df_local[['RFoot.posX', 'RFoot.posZ']].to_numpy()
    foot_midpoint = (LFoot + RFoot) / 2
    offset_dist = np.linalg.norm(Hips - foot_midpoint, axis=1)
    stance_width = np.linalg.norm(LFoot - RFoot, axis=1)
    return np.divide(offset_dist, stance_width, out=np.zeros_like(offset_dist), where=stance_width > 1e-8)

def knee_bend_ratio(df: pd.DataFrame, plane: Plane = 'xyz') -> np.ndarray:
    required = ['Hip.posY', 'LFoot.posY', 'RFoot.posY', 'Head.posY']
    df_local = _get_validated_data(df, required)
    avg_foot_height = (df_local['LFoot.posY'] + df_local['RFoot.posY']) / 2
    total_height = df_local['Head.posY'] - avg_foot_height
    hip_height = df_local['Hip.posY'] - avg_foot_height
    return np.divide(hip_height, total_height, out=np.zeros_like(hip_height), where=total_height > 1e-8)

def head_movement_ratio(df: pd.DataFrame, plane: Plane = 'xz') -> np.ndarray:
    required = ['Head.posX', 'Head.posZ', 'Hip.posX', 'Hip.posZ', 'LShoulder.posX', 'LShoulder.posZ', 'RShoulder.posX', 'RShoulder.posZ']
    df_local = _get_validated_data(df, required)
    Head = df_local[['Head.posX', 'Head.posZ']].to_numpy()
    Hips = df_local[['Hip.posX', 'Hip.posZ']].to_numpy()
    LShoulder = df_local[['LShoulder.posX', 'LShoulder.posZ']].to_numpy()
    RShoulder = df_local[['RShoulder.posX', 'RShoulder.posZ']].to_numpy()
    head_offset = np.linalg.norm(Head - Hips, axis=1)
    shoulder_width = np.linalg.norm(LShoulder - RShoulder, axis=1)
    return np.divide(head_offset, shoulder_width, out=np.zeros_like(head_offset), where=shoulder_width > 1e-8)

# -----------------------------
# Main Analysis Function
# -----------------------------

def analyze_motion_similarity(df_a: pd.DataFrame, df_b: pd.DataFrame, plane: Plane = 'xz', velocity_threshold: float = 0.02) -> Dict:
    
    # --- Step 1: Segment core actions from both dataframes ---
    # Since filenames are 'hook_left', we segment based on the left hand.
    df_a_segmented = segment_core_action(df_a, hand='L', velocity_threshold=velocity_threshold)
    df_b_segmented = segment_core_action(df_b, hand='L', velocity_threshold=velocity_threshold)

    if df_a_segmented.empty or df_b_segmented.empty:
        # Return a specific message if segmentation fails
        return {"individual_scores": {}, "final_score": 0.0, "segmented_frames": (len(df_a_segmented), len(df_b_segmented))}

    segmented_frames_info = (len(df_a_segmented), len(df_b_segmented))

    metrics_to_run: Dict[str, Callable] = {
        "자세: 발 너비/어깨너비": foot_to_shoulder_ratio,
        "자세: 스탠스 깊이/너비": stance_depth_width_ratio,
        "가드: 왼손 높이": left_guard_ratio,
        "가드: 오른손 높이": right_guard_ratio,
        "가드: 왼손 리치": left_hand_reach_ratio,
        "가드: 오른손 리치": right_hand_reach_ratio,
        "역동성: 몸통 회전": torso_rotation_ratio,
        "역동성: 무게중심 이동": com_shift_ratio,
        "역동성: 무릎 굽힘": knee_bend_ratio,
        "역동성: 헤드 무브먼트": head_movement_ratio,
    }

    results = {"individual_scores": {}}
    all_scores = []

    for name, func in metrics_to_run.items():
        try:
            # --- Step 2: Calculate metrics on segmented data ---
            r_a = func(df_a_segmented, plane=plane)
            r_b = func(df_b_segmented, plane=plane)
            
            dist = _dtw_distance_1d(r_a, r_b)
            sim_score = 1.0 / (1.0 + dist)
            
            results["individual_scores"][name] = sim_score
            all_scores.append(sim_score)
        except Exception as e:
            results["individual_scores"][name] = None
            print(f"Warning: Could not calculate metric '{name}'. Error: {e}")

    results["final_score"] = np.mean(all_scores) if all_scores else 0.0
    results["segmented_frames"] = segmented_frames_info
    return results
