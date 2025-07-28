# RL Playground - CartPole-v1 강화학습

이 프로젝트는 stable_baselines3를 사용하여 CartPole-v1 환경을 클리어하는 강화학습 실험입니다.

## 🎯 목표

CartPole-v1 환경에서 500 스텝 동안 막대를 균형잡는 것이 목표입니다. 475 스텝 이상을 달성하면 환경을 클리어한 것으로 간주합니다.

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 코드 실행
```bash
cd cart-pole
python dqn1.py
```

## 📊 구현된 알고리즘

- **DQN (Deep Q-Network)**: 딥러닝을 활용한 Q-러닝
- **PPO (Proximal Policy Optimization)**: 정책 기반 알고리즘
- **A2C (Advantage Actor-Critic)**: 액터-크리틱 알고리즘

## 🎮 CartPole-v1 환경

- **상태**: 카트의 위치, 속도, 막대의 각도, 각속도 (4차원)
- **행동**: 왼쪽(0) 또는 오른쪽(1)으로 카트 이동
- **보상**: 매 스텝마다 +1 (막대가 서있을 때)
- **종료 조건**: 막대가 너무 기울어지거나 카트가 경계를 벗어날 때
- **최대 스텝**: 500

## 📈 성능 평가

코드는 다음을 수행합니다:
1. 각 알고리즘을 15,000 스텝 동안 훈련
2. 20 에피소드로 성능 평가
3. 3 에피소드로 실제 테스트
4. 알고리즘 간 성능 비교
5. 훈련된 모델을 파일로 저장

## 💾 저장된 모델

훈련이 완료되면 다음 파일들이 생성됩니다:
- `cartpole_dqn.zip`: DQN 모델
- `cartpole_ppo.zip`: PPO 모델  
- `cartpole_a2c.zip`: A2C 모델

## 🔧 커스터마이징

`dqn1.py` 파일에서 다음 매개변수들을 조정할 수 있습니다:
- `total_timesteps`: 훈련 스텝 수
- `learning_rate`: 학습률
- `n_eval_episodes`: 평가 에피소드 수
- `n_episodes`: 테스트 에피소드 수

## 📝 예상 결과

일반적으로 PPO가 CartPole-v1에서 가장 안정적인 성능을 보이며, 대부분의 경우 500 스텝에 가까운 성능을 달성할 수 있습니다.
