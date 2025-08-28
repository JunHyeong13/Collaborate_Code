import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fastdtw import fastdtw


def visualize_results(similarity: float, feature_similarities: dict, feature_weights: dict):
    print("\n" + "=" * 64)
    print("동작 유사도 분석 결과 (v2.14)")
    print("=" * 64)

    print("\n특성별 유사도 (가중치 포함):")
    per_joint = feature_similarities.get('per_joint') if isinstance(feature_similarities.get('per_joint'), dict) else None
    part_weights = feature_similarities.get('part_weights') if isinstance(feature_similarities.get('part_weights'), dict) else {}
    for feature_name, sim in feature_similarities.items():
        if feature_name == 'per_joint':
            continue
        if feature_name == 'part_breakdown' or feature_name == 'part_weights':
            continue
        if isinstance(sim, (int, float, np.floating)):
            percentage = float(sim) * 100.0
            bar_len = int(min(max(percentage, 0.0) / 2.0, 50))
            bar = "█" * bar_len
            # 파트 키는 part_weights 사용, 그 외는 feature_weights 사용
            if feature_name.startswith('part_'):
                weight = float(part_weights.get(feature_name, 0.0))
            else:
                weight = float(feature_weights.get(feature_name, 0.0))
            print(f"  {feature_name:<15}: {percentage:6.2f}%  w={weight:.2f} |{bar}")

    print(f"\n최종 종합 유사도: {similarity * 100.0:.2f}%")

    if per_joint:
        try:
            print("\n조인트 단위 유사도 개요:")
            joint_scores = {}
            for f_type, mapping in per_joint.items():
                for key, val in mapping.items():
                    if isinstance(val, (int, float, np.floating)):
                        joint_scores.setdefault(key, []).append(float(val))
            joint_mean = [(k, float(np.mean(v))) for k, v in joint_scores.items() if v]
            joint_mean.sort(key=lambda x: x[1], reverse=True)
            top = joint_mean[:5]
            for name, sc in top:
                print(f"  {name:<12}: {sc*100.0:6.2f}%")
        except Exception:
            pass

    # 파트 상세 점수 출력
    part_breakdown = feature_similarities.get('part_breakdown') if isinstance(feature_similarities.get('part_breakdown'), dict) else None
    if part_breakdown:
        print("\n파트별 상세 유사도 (가중치 적용):")
        ordered_features = ['position', 'rotation', 'velocity', 'acceleration', 'joint_angles']
        for part_name, mapping in part_breakdown.items():
            cells = []
            for f in ordered_features:
                v = float(mapping.get(f, 0.0))
                w = float(feature_weights.get(f, 0.0))
                vw = v * w
                cells.append((vw, w))
            # 출력: 각 항목의 가중 적용 퍼센트와 weight 표시
            print(
                "  {}: pos {:6.2f}% (w={:.2f}) | rot {:6.2f}% (w={:.2f}) | vel {:6.2f}% (w={:.2f}) | acc {:6.2f}% (w={:.2f}) | ang {:6.2f}% (w={:.2f})".format(
                    f"{part_name:<12}",
                    cells[0][0]*100.0, cells[0][1],
                    cells[1][0]*100.0, cells[1][1],
                    cells[2][0]*100.0, cells[2][1],
                    cells[3][0]*100.0, cells[3][1],
                    cells[4][0]*100.0, cells[4][1],
                )
            )


def animate_3d_segments(analyzer,
                        motion1_data,
                        motion2_data,
                        overlay: bool = True,
                        interval: int = 40,
                        save_path: str | None = None,
                        joints_to_show: list[str] | None = None,
                        skeleton_edges: list[tuple[str, str]] | None = None,
                        sync_mode: str = 'linear'):
    if motion1_data is None or motion2_data is None:
        print("입력 데이터가 없어 3D 애니메이션을 생성할 수 없습니다.")
        return

    f1 = analyzer.extract_features(motion1_data)
    f2 = analyzer.extract_features(motion2_data)

    s1, e1 = analyzer.segment_action(f1)
    s2, e2 = analyzer.segment_action(f2)

    pos1 = f1.get('position', {})
    pos2 = f2.get('position', {})
    if not pos1 or not pos2:
        print("시각화할 위치 데이터가 부족합니다.")
        return

    joints_all_1 = set(pos1.keys())
    joints_all_2 = set(pos2.keys())
    joints_common = sorted(joints_all_1 & joints_all_2)
    if joints_to_show:
        joints = [j for j in joints_to_show if j in joints_common]
    else:
        joints = joints_common
    if not joints:
        print("공통 조인트가 없어 시각화를 중단합니다.")
        return

    seg1 = {j: pos1[j][s1:e1 + 1] for j in joints}
    seg2 = {j: pos2[j][s2:e2 + 1] for j in joints}

    len1 = len(next(iter(seg1.values())))
    len2 = len(next(iter(seg2.values())))

    def total_velocity_trace(seg):
        # 합계 속도 크기(프레임별): 모든 조인트 위치 차분의 노름 합
        traces = []
        for arr in seg.values():
            v = np.diff(arr, axis=0, prepend=arr[0:1])
            traces.append(np.linalg.norm(v, axis=1))
        if not traces:
            return np.zeros(0, dtype=float)
        return np.sum(traces, axis=0)

    if sync_mode not in ('linear', 'lag', 'dtw'):
        sync_mode = 'linear'

    if len1 <= 1 or len2 <= 1:
        print("세그먼트 길이가 짧아 애니메이션을 생성할 수 없습니다.")
        return

    if sync_mode == 'linear':
        num_frames = max(len1, len2)
        idx_map1 = np.clip(np.round(np.linspace(0, len1 - 1, num_frames)).astype(int), 0, len1 - 1)
        idx_map2 = np.clip(np.round(np.linspace(0, len2 - 1, num_frames)).astype(int), 0, len2 - 1)
    elif sync_mode == 'lag':
        tv1 = total_velocity_trace(seg1)
        tv2 = total_velocity_trace(seg2)
        # 교차상관 기반 래그 추정 (full)
        a = (tv1 - tv1.mean()) if tv1.size else tv1
        b = (tv2 - tv2.mean()) if tv2.size else tv2
        corr = np.correlate(a, b, mode='full')
        lag = int(np.argmax(corr) - (len(b) - 1))
        # 공통 길이로 매핑
        if lag >= 0:
            start1, start2 = lag, 0
        else:
            start1, start2 = 0, -lag
        end1 = len1
        end2 = len2
        valid_len = max(0, min(end1 - start1, end2 - start2))
        if valid_len <= 1:
            num_frames = max(len1, len2)
            idx_map1 = np.clip(np.round(np.linspace(0, len1 - 1, num_frames)).astype(int), 0, len1 - 1)
            idx_map2 = np.clip(np.round(np.linspace(0, len2 - 1, num_frames)).astype(int), 0, len2 - 1)
        else:
            num_frames = valid_len
            idx_map1 = np.arange(start1, start1 + valid_len, dtype=int)
            idx_map2 = np.arange(start2, start2 + valid_len, dtype=int)
    else:  # 'dtw'
        tv1 = total_velocity_trace(seg1)
        tv2 = total_velocity_trace(seg2)
        if tv1.size == 0 or tv2.size == 0:
            num_frames = max(len1, len2)
            idx_map1 = np.clip(np.round(np.linspace(0, len1 - 1, num_frames)).astype(int), 0, len1 - 1)
            idx_map2 = np.clip(np.round(np.linspace(0, len2 - 1, num_frames)).astype(int), 0, len2 - 1)
        else:
            _, path = fastdtw(tv1, tv2, dist=lambda x, y: abs(float(x - y)))
            # path: list of (i,j)
            if not path:
                num_frames = max(len1, len2)
                idx_map1 = np.clip(np.round(np.linspace(0, len1 - 1, num_frames)).astype(int), 0, len1 - 1)
                idx_map2 = np.clip(np.round(np.linspace(0, len2 - 1, num_frames)).astype(int), 0, len2 - 1)
            else:
                i_list = np.array([p[0] for p in path], dtype=int)
                j_list = np.array([p[1] for p in path], dtype=int)
                # 중복 완화 위해 균등 샘플링
                num_frames = len(path)
                # 너무 길면 제한하여 성능 보호
                max_frames = 2000
                if num_frames > max_frames:
                    sel = np.linspace(0, num_frames - 1, max_frames).astype(int)
                    i_list = i_list[sel]
                    j_list = j_list[sel]
                    num_frames = len(i_list)
                idx_map1 = np.clip(i_list, 0, len1 - 1)
                idx_map2 = np.clip(j_list, 0, len2 - 1)

    if skeleton_edges is not None:
        edges = [e for e in skeleton_edges if e[0] in joints and e[1] in joints]
    else:
        chains = [
            ['Hip', 'Ab', 'Chest', 'Neck', 'Head'],
            ['Chest', 'LShoulder', 'LUArm', 'LFArm', 'LHand'],
            ['Chest', 'RShoulder', 'RUArm', 'RFArm', 'RHand'],
            ['Hip', 'LThigh', 'LShin', 'LFoot', 'LToe'],
            ['Hip', 'RThigh', 'RShin', 'RFoot', 'RToe'],
        ]
        built = []
        for chain in chains:
            available = [n for n in chain if n in joints]
            for i in range(len(available) - 1):
                built.append((available[i], available[i + 1]))
        seen = set()
        edges = []
        for a, b in built:
            if (a, b) not in seen:
                edges.append((a, b))
                seen.add((a, b))

    def compute_limits():
        pts = []
        for j in joints:
            pts.append(seg1[j])
            pts.append(seg2[j])
        all_pts = np.vstack(pts)
        xmin, ymin, zmin = np.min(all_pts, axis=0)
        xmax, ymax, zmax = np.max(all_pts, axis=0)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        max_range = float(max(xmax - xmin, ymax - ymin, zmax - zmin))
        half = 0.5 * (max_range if max_range > 0 else 1.0)
        return (cx - half, cx + half), (cy - half, cy + half), (cz - half, cz + half)

    xlim, ylim, zlim = compute_limits()

    fig = plt.figure(figsize=(10, 5) if not overlay else (6, 6))
    if overlay:
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
    else:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        axes = [ax1, ax2]

    for a in axes:
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.set_zlim(*zlim)
        a.set_xlabel('X')
        a.set_ylabel('Y')
        a.set_zlabel('Z')
        a.view_init(elev=20, azim=-60)

    color1, color2 = '#1f77b4', '#d62728'

    def get_frame_pts(segment_dict, frame_idx):
        xs = [segment_dict[j][frame_idx, 0] for j in joints]
        ys = [segment_dict[j][frame_idx, 1] for j in joints]
        zs = [segment_dict[j][frame_idx, 2] for j in joints]
        return xs, ys, zs

    if overlay:
        ax = axes[0]
        xs1, ys1, zs1 = get_frame_pts(seg1, idx_map1[0])
        xs2, ys2, zs2 = get_frame_pts(seg2, idx_map2[0])
        scat1 = ax.scatter(xs1, ys1, zs1, c=color1, s=20, label='motion1')
        scat2 = ax.scatter(xs2, ys2, zs2, c=color2, s=20, label='motion2')
        lines1 = {}
        lines2 = {}
        for a, b in edges:
            i = joints.index(a)
            j = joints.index(b)
            l1, = ax.plot([xs1[i], xs1[j]], [ys1[i], ys1[j]], [zs1[i], zs1[j]], c=color1, lw=2)
            l2, = ax.plot([xs2[i], xs2[j]], [ys2[i], ys2[j]], [zs2[i], zs2[j]], c=color2, lw=2)
            lines1[(a, b)] = l1
            lines2[(a, b)] = l2
        title = ax.set_title('3D segment replay (overlay)')
    else:
        ax1, ax2 = axes
        xs1, ys1, zs1 = get_frame_pts(seg1, idx_map1[0])
        xs2, ys2, zs2 = get_frame_pts(seg2, idx_map2[0])
        scat1 = ax1.scatter(xs1, ys1, zs1, c=color1, s=20)
        scat2 = ax2.scatter(xs2, ys2, zs2, c=color2, s=20)
        lines1 = {}
        lines2 = {}
        for a, b in edges:
            i = joints.index(a)
            j = joints.index(b)
            l1, = ax1.plot([xs1[i], xs1[j]], [ys1[i], ys1[j]], [zs1[i], zs1[j]], c=color1, lw=2)
            l2, = ax2.plot([xs2[i], xs2[j]], [ys2[i], ys2[j]], [zs2[i], zs2[j]], c=color2, lw=2)
            lines1[(a, b)] = l1
            lines2[(a, b)] = l2
        title = fig.suptitle('3D segment replay (side-by-side)')

    def update(frame):
        i1 = int(idx_map1[frame])
        i2 = int(idx_map2[frame])
        xs1, ys1, zs1 = get_frame_pts(seg1, i1)
        xs2, ys2, zs2 = get_frame_pts(seg2, i2)
        scat1._offsets3d = (xs1, ys1, zs1)
        scat2._offsets3d = (xs2, ys2, zs2)
        for (a, b), line in lines1.items():
            ii = joints.index(a)
            jj = joints.index(b)
            line.set_data([xs1[ii], xs1[jj]], [ys1[ii], ys1[jj]])
            line.set_3d_properties([zs1[ii], zs1[jj]])
        for (a, b), line in lines2.items():
            ii = joints.index(a)
            jj = joints.index(b)
            line.set_data([xs2[ii], xs2[jj]], [ys2[ii], ys2[jj]])
            line.set_3d_properties([zs2[ii], zs2[jj]])
        title.set_text(f'3D segment replay (frame {frame + 1}/{num_frames})')
        return []

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    if save_path:
        try:
            writer = PillowWriter(fps=max(1, int(1000 / max(1, interval))))
            anim.save(save_path, writer=writer)
            print(f"애니메이션을 저장했습니다: {save_path}")
        except Exception as e:
            print(f"애니메이션 저장 중 오류가 발생했습니다: {e}")

    plt.show()


def export_joint_map_figure(output_path: str, dpi: int = 200, language: str = 'ko'):
    node_pos = {
        'Hip': (0.0, 0.0), 'Ab': (0.0, 1.0), 'Chest': (0.0, 2.0), 'Neck': (0.0, 3.0), 'Head': (0.0, 4.0),
        'LShoulder': (-1.0, 2.8), 'LUArm': (-1.8, 2.2), 'LFArm': (-2.2, 1.5), 'LHand': (-2.3, 1.0),
        'RShoulder': (1.0, 2.8), 'RUArm': (1.8, 2.2), 'RFArm': (2.2, 1.5), 'RHand': (2.3, 1.0),
        'LThigh': (-0.5, -1.0), 'LShin': (-0.5, -2.0), 'LFoot': (-0.5, -2.5), 'LToe': (-0.5, -2.8),
        'RThigh': (0.5, -1.0), 'RShin': (0.5, -2.0), 'RFoot': (0.5, -2.5), 'RToe': (0.5, -2.8),
    }

    chains = [
        ['Hip', 'Ab', 'Chest', 'Neck', 'Head'],
        ['Chest', 'LShoulder', 'LUArm', 'LFArm', 'LHand'],
        ['Chest', 'RShoulder', 'RUArm', 'RFArm', 'RHand'],
        ['Hip', 'LThigh', 'LShin', 'LFoot', 'LToe'],
        ['Hip', 'RThigh', 'RShin', 'RFoot', 'RToe'],
    ]

    labels_ko = {
        'Hip': '골반', 'Ab': '복부', 'Chest': '흉부', 'Neck': '목', 'Head': '머리',
        'LShoulder': '왼쪽 어깨', 'LUArm': '왼쪽 상완', 'LFArm': '왼쪽 하완', 'LHand': '왼손',
        'RShoulder': '오른쪽 어깨', 'RUArm': '오른쪽 상완', 'RFArm': '오른쪽 하완', 'RHand': '오른손',
        'LThigh': '왼쪽 허벅지', 'LShin': '왼쪽 종아리', 'LFoot': '왼발', 'LToe': '왼쪽 발끝',
        'RThigh': '오른쪽 허벅지', 'RShin': '오른쪽 종아리', 'RFoot': '오른발', 'RToe': '오른쪽 발끝',
    }
    labels_en = {
        'Hip': 'Pelvis', 'Ab': 'Abdomen', 'Chest': 'Chest', 'Neck': 'Neck', 'Head': 'Head',
        'LShoulder': 'Left Shoulder', 'LUArm': 'Left Upper Arm', 'LFArm': 'Left Forearm', 'LHand': 'Left Hand',
        'RShoulder': 'Right Shoulder', 'RUArm': 'Right Upper Arm', 'RFArm': 'Right Forearm', 'RHand': 'Right Hand',
        'LThigh': 'Left Thigh', 'LShin': 'Left Shin', 'LFoot': 'Left Foot', 'LToe': 'Left Toe',
        'RThigh': 'Right Thigh', 'RShin': 'Right Shin', 'RFoot': 'Right Foot', 'RToe': 'Right Toe',
    }
    labels = labels_ko if language == 'ko' else labels_en

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Joint Map', pad=12)

    for chain in chains:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if a in node_pos and b in node_pos:
                x1, y1 = node_pos[a]
                x2, y2 = node_pos[b]
                ax.plot([x1, x2], [y1, y2], color='#444', linewidth=2)

    for j, (x, y) in node_pos.items():
        ax.scatter([x], [y], s=60, color='#1f77b4', zorder=3)
        label = labels.get(j, j)
        ha = 'right' if j.startswith('L') else ('left' if j.startswith('R') else 'center')
        dx = -0.12 if ha == 'right' else (0.12 if ha == 'left' else 0)
        dy = 0.06
        ax.text(x + dx, y + dy, f"{j} ({label})", fontsize=9, ha=ha, va='bottom')

    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    xmin, xmax = min(xs) - 0.8, max(xs) + 0.8
    ymin, ymax = min(ys) - 0.8, max(ys) + 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"조인트 맵 이미지를 저장했습니다: {output_path}")


