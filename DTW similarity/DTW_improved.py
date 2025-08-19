import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings

# dtaidistance 라이브러리 임포트
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

warnings.filterwarnings('ignore')

class PoseSimilarityAnalyzer:
    """
    모션 캡처 데이터를 사용하여 두 포즈 시퀀스의 유사도를 분석하는 클래스.
    - 주요 기능: DTW, 프레임별 코사인 유사도, 관절별 차이 분석, 피드백 생성, 시각화.
    - 개선 사항: dtaidistance 라이브러리 사용으로 성능 향상, 시간적 특성을 고려한 정확한 유사도 계산.
    """
    def __init__(self):
        self.trainer_data: Optional[pd.DataFrame] = None
        self.member_data: Optional[pd.DataFrame] = None
        self.joint_names = [
            'Hip', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',
            'RShoulder', 'RUArm', 'RFArm', 'RHand', 'LThigh', 'LShin', 'LFoot',
            'RThigh', 'RShin', 'RFoot', 'LToe', 'RToe'
        ]
        # 분석 결과를 저장할 내부 변수
        self.analysis_results = {}

    def load_data(self, trainer_csv: str, member_csv: str) -> None:
        """트레이너와 회원 데이터 로드"""
        try:
            self.trainer_data = pd.read_csv(trainer_csv).rename(columns=str.strip)
            self.member_data = pd.read_csv(member_csv).rename(columns=str.strip)
            print(f"✅ 트레이너 데이터 로드 완료: {os.path.basename(trainer_csv)} ({len(self.trainer_data)} 프레임)")
            print(f"✅ 회원 데이터 로드 완료: {os.path.basename(member_csv)} ({len(self.member_data)} 프레임)")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"데이터 로드 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요.\n{e}")
        except Exception as e:
            raise IOError(f"데이터 로드 중 예상치 못한 오류 발생: {e}")

    def _preprocess_and_weight(self, data: pd.DataFrame, feature_cols: List[str], joint_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """데이터 전처리 및 관절 가중치 적용"""
        data_filled = data.fillna(method='ffill').fillna(method='bfill')
        features_df = data_filled[feature_cols]

        if joint_weights is None:
            joint_weights = self._get_default_weights()

        weighted_features_list = []
        for col in features_df.columns:
            weight = 1.0
            for joint, w in joint_weights.items():
                if col.startswith(joint + '.'):
                    weight = w
                    break
            weighted_features_list.append(features_df[col] * weight)

        return np.column_stack(weighted_features_list)

    def _get_default_weights(self) -> Dict[str, float]:
        """기본 관절 가중치 반환"""
        return {
            'Hip': 1.5, 'Ab': 1.3, 'Chest': 1.2, 'LThigh': 1.4, 'RThigh': 1.4,
            'LShin': 1.2, 'RShin': 1.2, 'LFoot': 1.0, 'RFoot': 1.0,
            'LShoulder': 1.1, 'RShoulder': 1.1, 'LUArm': 1.0, 'RUArm': 1.0,
        }

    def analyze(self) -> None:
        """모든 분석을 수행하고 결과를 내부에 저장"""
        if self.trainer_data is None or self.member_data is None:
            raise ValueError("데이터를 먼저 로드해주세요.")

        # --- BUG FIX: 두 데이터에 공통으로 존재하는 feature만 선택 ---
        trainer_cols = self.trainer_data.select_dtypes(include=np.number).columns
        member_cols = self.member_data.select_dtypes(include=np.number).columns
        common_cols = list(set(trainer_cols) & set(member_cols))
        feature_cols = [col for col in common_cols if col not in ['Frame', 'Time (Seconds)']]
        
        if not feature_cols:
            raise ValueError("두 데이터 파일에 공통된 특징(feature) 컬럼이 없습니다.")

        # --- 1. 전체 동작 유사도 분석 ---
        trainer_weighted = self._preprocess_and_weight(self.trainer_data, feature_cols)
        member_weighted = self._preprocess_and_weight(self.member_data, feature_cols)

        scaler = StandardScaler()
        combined_data = np.vstack([trainer_weighted, member_weighted])
        scaler.fit(combined_data)

        trainer_scaled = scaler.transform(trainer_weighted)
        member_scaled = scaler.transform(member_weighted)

        dtw_dist = dtw.distance(trainer_scaled, member_scaled, use_pruning=True)
        normalized_dtw_dist = dtw_dist / (len(trainer_scaled) + len(member_scaled))

        min_len = min(len(trainer_scaled), len(member_scaled))
        frame_cos_sim = [
            cosine_similarity(trainer_scaled[i].reshape(1, -1), member_scaled[i].reshape(1, -1))[0, 0]
            for i in range(min_len)
        ]
        avg_cos_sim = np.mean(frame_cos_sim) if frame_cos_sim else 0

        self.analysis_results['DTW_Similarity'] = max(0, 100 - normalized_dtw_dist * 25)
        self.analysis_results['Cosine_Similarity'] = avg_cos_sim * 100
        self.analysis_results['Overall_Score'] = (
            self.analysis_results['DTW_Similarity'] * 0.7 +
            self.analysis_results['Cosine_Similarity'] * 0.3
        )

        # --- 2. 관절별 움직임 차이 분석 ---
        joint_scores = {}
        for joint in self.joint_names:
            joint_cols = [col for col in feature_cols if col.startswith(joint + '.')]
            if not joint_cols:
                continue

            trainer_joint_seq = self.trainer_data[joint_cols].values
            member_joint_seq = self.member_data[joint_cols].values

            if trainer_joint_seq.shape[0] == 0 or member_joint_seq.shape[0] == 0:
                continue

            scaler_joint = StandardScaler()
            combined_joint_seq = np.vstack([trainer_joint_seq, member_joint_seq])
            scaler_joint.fit(combined_joint_seq)

            trainer_joint_scaled = scaler_joint.transform(trainer_joint_seq)
            member_joint_scaled = scaler_joint.transform(member_joint_seq)

            j_dist = dtw.distance(trainer_joint_scaled, member_joint_scaled, use_pruning=True)
            j_dist_norm = j_dist / (len(trainer_joint_scaled) + len(member_joint_scaled))
            joint_scores[joint] = max(0, 100 - j_dist_norm * 50)

        self.analysis_results['Joint_Scores'] = joint_scores
        self.analysis_results['scaled_data'] = (trainer_scaled, member_scaled)

    def get_feedback(self) -> str:
        """분석 결과에 기반한 텍스트 피드백 생성"""
        if not self.analysis_results:
            return "분석을 먼저 수행해주세요."
            
        overall = self.analysis_results.get('Overall_Score', 0)
        joint_scores = self.analysis_results.get('Joint_Scores', {})
        
        feedback = []
        if overall >= 90:
            feedback.append("🏆 훌륭합니다! 트레이너와 거의 일치하는 완벽한 자세입니다.")
        elif overall >= 80:
            feedback.append("👍 좋은 자세입니다. 조금만 더 개선하면 완벽해집니다.")
        elif overall >= 70:
            feedback.append("✅ 양호한 자세입니다. 개선이 필요한 부분을 확인해보세요.")
        else:
            feedback.append("🔴 자세 개선이 필요합니다. 트레이너의 동작과 차이가 큰 부분이 있습니다.")
        
        if joint_scores:
            worst_joints = sorted(joint_scores.items(), key=lambda x: x[1])[:3]
            if worst_joints and worst_joints[0][1] < 85:
                feedback.append("\n📍 특히 다음 부위의 '움직임' 개선이 필요합니다:")
                for joint, score in worst_joints:
                    if score < 85:
                        feedback.append(f"  - {joint} (유사도: {score:.1f}%)")
        
        return "\n".join(feedback)
        
    def display_results(self):
        """콘솔에 분석 결과 출력"""
        print("\n" + "="*25)
        print("📈  유사도 분석 종합 결과")
        print("="*25)
        for key, val in self.analysis_results.items():
            if key not in ['Joint_Scores', 'scaled_data']:
                print(f"{key:<20}: {val:.2f}")
        
        print("\n" + "="*25)
        print("🤸 관절별 움직임 유사도")
        print("="*25)
        joint_scores = self.analysis_results.get('Joint_Scores', {})
        if joint_scores:
            for joint, score in sorted(joint_scores.items(), key=lambda x: x[1]):
                status = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
                print(f"{status} {joint:<12}: {score:.1f}%")

    def visualize_analysis(self, save_plot: bool = False):
        """분석 결과 시각화"""
        # (시각화 코드는 변경사항이 없으므로 생략 - 이전 답변의 코드를 그대로 사용)
        """분석 결과 시각화"""
        if not self.analysis_results:
            print("분석을 먼저 수행해주세요.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 16))
        
        # 1. 종합 점수
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        metrics = ['DTW_Similarity', 'Cosine_Similarity', 'Overall_Score']
        values = [self.analysis_results.get(m, 0) for m in metrics]
        colors = ['skyblue', 'lightgreen', 'gold']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_title('Overall Similarity Scores', fontsize=16)
        ax1.set_ylabel('Score (0-100)', fontsize=12)
        ax1.set_ylim(0, 105)
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center')

        # 2. 관절별 유사도
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        joint_scores = self.analysis_results.get('Joint_Scores', {})
        if joint_scores:
            sorted_joints = sorted(joint_scores.items(), key=lambda x: x[1])
            joints = [j[0] for j in sorted_joints]
            scores = [j[1] for j in sorted_joints]
            colors = ['#d9534f' if s < 70 else '#f0ad4e' if s < 85 else '#5cb85c' for s in scores]
            ax2.barh(joints, scores, color=colors)
            ax2.set_title('Joint Movement Similarity', fontsize=16)
            ax2.set_xlabel('Similarity Score (%)', fontsize=12)
            ax2.axvline(x=85, color='gray', linestyle='--', alpha=0.7)

        # 3. 대표 관절 시계열 비교 (Hip.posY)
        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        if 'Time (Seconds)' in self.trainer_data.columns and 'Hip.posY' in self.trainer_data.columns:
            ax3.plot(self.trainer_data['Time (Seconds)'], self.trainer_data['Hip.posY'], label='Trainer', lw=2)
            ax3.plot(self.member_data['Time (Seconds)'], self.member_data['Hip.posY'], label='Member', lw=2, alpha=0.8)
            ax3.set_title('Hip Y-Position Trajectory', fontsize=16)
            ax3.set_xlabel('Time (s)', fontsize=12)
            ax3.set_ylabel('Position', fontsize=12)
            ax3.legend()

        # 4. DTW 시간 왜곡 경로 시각화 (개선된 시각화)
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        trainer_s, member_s = self.analysis_results.get('scaled_data', (None, None))
        if trainer_s is not None and member_s is not None:
            path = dtw.warping_path(trainer_s, member_s)
            dtwvis.plot_warping(trainer_s, member_s, path, ax=ax4)
            ax4.set_title('DTW Warping Path (Time Alignment)', fontsize=16)
            ax4.set_xlabel("Trainer's Frames", fontsize=12)
            ax4.set_ylabel("Member's Frames", fontsize=12)

        plt.tight_layout(pad=3.0)
        if save_plot:
            plt.savefig('pose_similarity_analysis.png', dpi=300)
            print("\n🖼️  'pose_similarity_analysis.png' 파일로 시각화 결과가 저장되었습니다.")
        plt.show()

def main():
    """메인 실행 함수"""
    # --- test 폴더의 실제 CSV 사용 ---
    base = os.path.join(os.path.dirname(__file__), 'test')
    trainer_file = os.path.join(base, 'jap_001.csv')
    member_file = os.path.join(base, 'p04_jap_main_013.csv')
    if not (os.path.isfile(trainer_file) and os.path.isfile(member_file)):
        print("❌ test 폴더에서 CSV를 찾을 수 없습니다. 경로를 확인하세요:")
        print(f"  trainer_file: {trainer_file}")
        print(f"  member_file : {member_file}")
        return
    
    # --- 분석기 실행 ---
    analyzer = PoseSimilarityAnalyzer()
    
    try:
        analyzer.load_data(trainer_file, member_file)
        
        # 모든 분석 수행
        analyzer.analyze()
        
        # 결과 출력
        analyzer.display_results()
        
        # 피드백 출력
        print("\n" + "="*25)
        print("📝 개선 피드백")
        print("="*25)
        print(analyzer.get_feedback())

        # 시각화
        analyzer.visualize_analysis(save_plot=True)

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()