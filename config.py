ENV_NAME = "CartPole-v1" # 강화학습 환경 이름

GAMMA = 0.99 # 할인율 - 미래 보상을 얼마나 주요하게 볼지
LR = 1e-3 # Q-network가 얼마나 세게 업데이트 될지. 너무 크면 => 폭주 / 너무 작으면 => 안배움 ===>>>> 이게 보통 default 값

EPS_START = 1.0 # 학습 시작 시 탐험 비율
EPS_END = 0.05 # 탐험의 최소 한계
EPS_DECAY = 0.9995 # 입실론이 줄어드는 속도 (병렬 환경이므로 천천히 줄임)

BATCH_SIZE = 64 # 한 번 학습할 때 꺼내는 기억 개수
MEMORY_SIZE = 10000 # replay buffer 최대 용량

EPISODES = 5000 # 총 학습 횟수 (병렬이므로 늘림)
NUM_ENVS = 9 # 병렬로 실행할 환경 개수
TARGET_UPDATE_FREQ = 100 # 타겟 네트워크 업데이트 주기 (스텝 단위)

# 학습에서 중요한건 EPS_DECAY, LR, GAMMA