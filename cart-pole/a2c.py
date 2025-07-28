import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class LoggingCallback(BaseCallback):
    """콜백에서 사용 가능한 값들을 로깅하는 콜백"""

    def __init__(self, log_interval=1, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0  # 에피소드 총 보상 추적
        self.start_exploration_rate = False  # 성공 기준점 플래그
        self.initial_exploration_rate = 1.0  # 초기 탐험률

    def _on_step(self) -> bool:
        self.step_count += 1

        # 벡터화된 환경에서는 dones 배열 사용
        dones = self.locals.get("dones", [False])

        # 에피소드가 끝났는지 확인 (벡터화된 환경)
        if dones[0]:  # 첫 번째 환경의 done 상태
            self.episode_count += 1

            # infos에서 에피소드 정보 추출
            episode_info = self.locals.get("infos", [{}])[0].get("episode", {})
            episode_reward = episode_info.get("r", 0.0)  # 총 보상
            episode_length = episode_info.get("l", 0)  # 에피소드 길이
            episode_time = episode_info.get("t", 0.0)  # 소요 시간

            if self.episode_count % self.log_interval == 0:
                print(
                    f"🔍 콜백 로깅 - 에피소드 {self.episode_count}, steps: {self.step_count}, reward: {episode_reward}"
                )

        return True


def create_env(video_folder=None):
    """CartPole-v1 환경을 생성합니다."""
    if video_folder:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = RecordVideo(env, video_folder)
        print(f"비디오 녹화가 활성화되었습니다: {video_folder}")
        return env

    env = gym.make("CartPole-v1")
    # env = Monitor(env)
    return env


def setup_logging():
    """로그 디렉토리를 설정합니다."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def setup_model_save_dir():
    """모델 저장 디렉토리를 설정합니다."""
    save_dir = "zips"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def create_a2c_model(env, log_dir="logs"):
    """A2C 모델을 생성합니다."""
    print("A2C 모델 생성...")

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.25,
        max_grad_norm=0.5,
        rms_prop_eps=1e-05,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=os.path.join(log_dir, "a2c"),
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
        },
        device="cuda",
    )

    return model


def evaluate_model(model, env, n_eval_episodes=10):
    """모델을 평가합니다."""
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def test_model(model, env, n_episodes=5):
    """훈련된 모델을 테스트하고 결과를 시각화합니다."""
    print(f"\n{model.__class__.__name__} 모델 테스트 시작...")

    episode_rewards = []
    episode_steps = []
    clear_count = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps >= 500:  # CartPole-v1의 최대 스텝
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        print(f"에피소드 {episode + 1}: 총 보상 = {total_reward}, 스텝 = {steps}")

        if total_reward >= 475:  # CartPole-v1 클리어 기준 (500 스텝 중 475 이상)
            print(f"🎉 에피소드 {episode + 1}에서 CartPole-v1을 클리어했습니다!")
            clear_count += 1
        else:
            print(f"❌ 에피소드 {episode + 1}에서 클리어하지 못했습니다.")

    # 통계 요약
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    clear_rate = clear_count / n_episodes * 100

    print(f"\n📊 테스트 결과 요약:")
    print(f"평균 보상: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"클리어율: {clear_rate:.1f}% ({clear_count}/{n_episodes})")
    print(f"최고 보상: {max(episode_rewards)}")
    print(f"최저 보상: {min(episode_rewards)}")

    return episode_rewards, episode_steps


def run_a2c(no_save=False):
    """A2C 알고리즘을 실행합니다."""
    env = create_env()
    log_dir = setup_logging()
    save_dir = setup_model_save_dir()

    print(f"\n{'='*50}")
    print("A2C 알고리즘 테스트")
    print(f"{'='*50}")

    # 모델 생성
    model = create_a2c_model(env, log_dir)

    # 학습 실행
    print("A2C 모델 학습 시작...")
    callback = LoggingCallback()
    model.learn(
        total_timesteps=20000,
        log_interval=100,
        callback=callback,
    )

    # 평가
    mean_reward, std_reward = evaluate_model(model, env, n_eval_episodes=20)

    # 테스트
    test_model(model, env, n_episodes=3)

    # 모델 저장
    if not no_save:
        model_path = os.path.join(save_dir, "cartpole_a2c")
        model.save(model_path)
        print(f"A2C 모델이 {model_path}.zip으로 저장되었습니다.")
    else:
        print("모델 저장을 건너뜁니다.")

    return mean_reward, std_reward


def play_a2c():
    """저장된 A2C 모델로 비디오를 찍습니다."""
    save_dir = setup_model_save_dir()
    model_path = os.path.join(save_dir, "cartpole_a2c")

    # 모델 로드
    try:
        model = A2C.load(model_path)
        print(f"모델을 로드했습니다: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("먼저 모델을 훈련시켜주세요.")
        return

    # 비디오 폴더 설정
    video_folder = "videos/a2c"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # 비디오 녹화 환경 생성
    env = create_env(video_folder=video_folder)

    print(f"\n{'='*50}")
    print("A2C 모델 비디오 녹화 시작")
    print(f"{'='*50}")

    # 비디오 녹화 실행
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if steps >= 500:  # CartPole-v1의 최대 스텝
            break

    env.close()

    print(f"비디오 녹화 완료!")
    print(f"총 보상: {total_reward}, 스텝: {steps}")
    print(f"비디오 파일 위치: {video_folder}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.video:
        play_a2c()
    else:
        mean_reward, std_reward = run_a2c(no_save=args.no_save)
        print(f"\n🏆 A2C 최종 결과: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
