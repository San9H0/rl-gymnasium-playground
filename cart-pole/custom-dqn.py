import copy
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo


class CustomReplayBuffer:
    """ """

    def __init__(self, buffer_size, batch_size):
        """ """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """ """
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(
        self,
    ):  # -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[A...:
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data])

        return state, action, reward, next_state, done


class CustomNStepReplayBuffer:
    """N-step Replay Buffer for DQN"""

    def __init__(self, buffer_size, batch_size, n_steps=3, gamma=0.99):
        """
        Initialize N-step Replay Buffer

        Args:
            buffer_size: Maximum size of the buffer
            batch_size: Size of batches to sample
            n_steps: Number of steps to look ahead for n-step returns
            gamma: Discount factor for n-step returns
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma

        # Temporary storage for n-step transitions
        self.n_step_buffer = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the n-step buffer"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we have enough steps or episode is done, add to main buffer
        if len(self.n_step_buffer) == self.n_steps or done:
            self._add_n_step_transition()

        # If episode is done, add remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                self._add_n_step_transition()

    def _add_n_step_transition(self):
        """Add n-step transition to main buffer"""
        if len(self.n_step_buffer) == 0:
            return

        # Get the first transition
        first_state, first_action, _, _, _ = self.n_step_buffer[0]

        # Calculate n-step return
        n_step_return = 0
        n_step_next_state = None
        n_step_done = False

        for i, (_, _, reward, next_state, done) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma**i) * reward
            if i == len(self.n_step_buffer) - 1:
                n_step_next_state = next_state
                n_step_done = done
            if done:
                n_step_done = True
                break

        # Add to main buffer
        data = (
            first_state,
            first_action,
            n_step_return,
            n_step_next_state,
            n_step_done,
        )
        self.buffer.append(data)

        # Remove the first transition from n-step buffer
        self.n_step_buffer.popleft()

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        """Sample a batch from the buffer"""
        if len(self.buffer) < self.batch_size:
            return None

        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data])

        return state, action, reward, next_state, done


class CustomQNet(nn.Module):
    """ """

    def __init__(self, action_size):
        """ """
        super().__init__()
        self.l1 = nn.Linear(4, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_size)

    def forward(self, x):
        """ """
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CustomDQNAgent:
    def __init__(self):
        """ """
        self.gamma = 0.98
        self.lr = 0.0005  # 5e-4
        self.buffer_size = 10000
        self.batch_size = 32  # 32
        self.action_size = 2  # CartPole은 2개 액션
        self.use_n_step = True
        self.n_steps = 20

        # Epsilon decay parameters (기본값 설정)
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.initial_epsilon
        # 300 에피소드의 절반(150) 동안 epsilon을 감소시킴
        self.epsilon_decay = (self.initial_epsilon - self.final_epsilon) / 150

        # Choose between regular and n-step replay buffer
        if self.use_n_step:
            self.replay_buffer = CustomNStepReplayBuffer(
                self.buffer_size,
                self.batch_size,
                n_steps=self.n_steps,
                gamma=self.gamma,
            )
            print(f"Using N-step Replay Buffer (n_steps={self.n_steps})")
        else:
            self.replay_buffer = CustomReplayBuffer(self.buffer_size, self.batch_size)
            print("Using Regular Replay Buffer")

        self.qnet = CustomQNet(self.action_size)
        self.qnet_target = CustomQNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def decay_epsilon(self):
        """Decay epsilon value"""
        if self.epsilon_decay is not None:
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def sync_qnet(self):
        """ """
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        """ """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # numpy -> torch tensor 변환
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        """DQN 방식의 파라미터 업데이트"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 배치 샘플링
        batch_data = self.replay_buffer.get_batch()
        if batch_data is None:
            return

        state, action, reward, next_state, done = batch_data
        # numpy -> torch tensor 변환
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # 현재 Q값
        qs = self.qnet(state)
        q = qs.gather(1, action.unsqueeze(1)).squeeze(1)

        # 다음 상태 Q값 (target network, gradient X)
        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(dim=1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        # 손실 계산 및 역전파
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class CustomDQN:
    """Custom DQN Model Class"""

    def __init__(
        self,
        env,
        episodes=300,
        sync_interval=20,
        video_folder=None,
        show_plot=True,
    ):
        """
        CustomDQN initialization

        Args:
            env: Gymnasium environment
            episodes: Number of episodes to train
            sync_interval: Target network update interval
            video_folder: Video save folder (None to disable video recording)
            show_plot: Whether to show training history plot
        """
        self.env = env
        self.episodes = episodes
        self.sync_interval = sync_interval
        self.video_folder = video_folder
        self.show_plot = show_plot

        # Environment information output
        print(f"observation_space: {self.env.observation_space}")
        print(f"action_space: {self.env.action_space}")

        # Video recording setup (only if video_folder is provided)
        if self.video_folder and self.video_folder.strip():
            self.env = RecordVideo(
                self.env,
                video_folder=self.video_folder,
                name_prefix="eval",
                episode_trigger=lambda x: x % 20 == 0,
            )
            print(f"Video recording enabled: {self.video_folder}")
        else:
            print("Video recording disabled")

        # DQN agent creation
        self.agent = CustomDQNAgent()
        self.reward_history = []

    def learn(self):
        """Model learning"""
        print(f"CustomDQN learning started - episodes: {self.episodes}")

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = float(0)

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += float(reward)

            if episode % self.sync_interval == 0:
                self.agent.sync_qnet()

            # Decay epsilon after each episode
            self.agent.decay_epsilon()

            self.reward_history.append(total_reward)
            if episode % 10 == 0:
                print(
                    "episode :{}, total reward : {}, epsilon : {}".format(
                        episode, total_reward, self.agent.epsilon
                    )
                )
        # Plot training history if enabled
        self.plot_training_history()
        print("Training completed!")

    def plot_training_history(self):
        """Plot training history"""
        if not self.show_plot:
            return

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.title("CustomDQN Training History")
        plt.show()

    def evaluate(self, n_episodes=1):
        """Model evaluation (greedy policy)"""
        print(f"Model evaluation started - {n_episodes} episodes")

        self.agent.epsilon = 0  # Greedy policy

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += float(reward)
                self.env.render()

            print(f"Episode {episode + 1} - Total Reward: {total_reward}")

        return total_reward

    def close(self):
        """Close environment"""
        self.env.close()


import argparse
import os


def setup_model_save_dir():
    """모델 저장 디렉토리를 설정합니다."""
    save_dir = "zips"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def run_custom_dqn(no_save=False):
    """Custom DQN 알고리즘을 실행합니다."""
    env = gym.make("CartPole-v1")

    print(f"\n{'='*50}")
    print("Custom DQN 알고리즘 테스트")
    print(f"{'='*50}")

    # CustomDQN model creation
    model = CustomDQN(
        env=env,
        episodes=300,
        sync_interval=20,
        video_folder=None,
        show_plot=True,
    )

    # Learning
    model.learn()

    # Evaluation
    final_reward = model.evaluate(n_episodes=1)

    # 모델 저장
    if not no_save:
        save_dir = setup_model_save_dir()
        model_path = os.path.join(save_dir, "cartpole_custom_dqn")
        # Custom DQN의 경우 모델 상태를 저장
        torch.save(
            {
                "qnet_state_dict": model.agent.qnet.state_dict(),
                "qnet_target_state_dict": model.agent.qnet_target.state_dict(),
                "optimizer_state_dict": model.agent.optimizer.state_dict(),
                "reward_history": model.reward_history,
            },
            model_path + ".pth",
        )
        print(f"Custom DQN 모델이 {model_path}.pth로 저장되었습니다.")
    else:
        print("모델 저장을 건너뜁니다.")

    # Close environment
    model.close()

    return final_reward


def play_custom_dqn():
    """저장된 Custom DQN 모델로 비디오를 찍습니다."""
    save_dir = setup_model_save_dir()
    model_path = os.path.join(save_dir, "cartpole_custom_dqn")

    # 모델 로드
    try:
        checkpoint = torch.load(model_path + ".pth")
        print(f"모델을 로드했습니다: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("먼저 모델을 훈련시켜주세요.")
        return

    # 비디오 폴더 설정
    video_folder = "videos/custom-dqn"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # 비디오 녹화 환경 생성
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder)
    print(f"비디오 녹화가 활성화되었습니다: {video_folder}")

    # Custom DQN 모델 생성 및 가중치 로드
    model = CustomDQN(
        env=env, episodes=1, sync_interval=20, video_folder=None, show_plot=False
    )

    # 저장된 가중치 로드
    model.agent.qnet.load_state_dict(checkpoint["qnet_state_dict"])
    model.agent.qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])
    model.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.reward_history = checkpoint["reward_history"]

    print(f"\n{'='*50}")
    print("Custom DQN 모델 비디오 녹화 시작")
    print(f"{'='*50}")

    # 비디오 녹화 실행
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        action = model.agent.get_action(obs)
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
        play_custom_dqn()
    else:
        final_reward = run_custom_dqn(no_save=args.no_save)
        print(f"\n🏆 Custom DQN 최종 결과: {final_reward:.2f}")


if __name__ == "__main__":
    main()
