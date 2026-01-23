import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV 추가

from dqn import DQN
from replay_buffer import ReplayBuffer
import config

# 병렬 환경 생성
# render_mode를 "rgb_array"로 설정하여 이미지 데이터를 받아옴
if hasattr(gym, "make_vec"):
    env = gym.make_vec(config.ENV_NAME, num_envs=config.NUM_ENVS, render_mode="rgb_array")
else:
    env = gym.vector.make(config.ENV_NAME, num_envs=config.NUM_ENVS, render_mode="rgb_array")

state_dim = env.single_observation_space.shape[0]
action_dim = env.single_action_space.n

q_net = DQN(state_dim, action_dim)
target_q_net = DQN(state_dim, action_dim) # 타겟 네트워크 추가
target_q_net.load_state_dict(q_net.state_dict())
target_q_net.eval() # 타겟 네트워크는 학습하지 않음

optimizer = optim.Adam(q_net.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()

memory = ReplayBuffer(config.MEMORY_SIZE)

rewards_history = []
epsilon = config.EPS_START

# 상태 초기화
states, _ = env.reset()

total_steps = 0
completed_episodes = 0

# 렌더링 윈도우 이름
window_name = "Parallel CartPole Training"

try:
    while completed_episodes < config.EPISODES:
        # --- 렌더링 처리 ---
        # 각 환경의 프레임을 가져옴 (num_envs, height, width, 3)
        frames = env.render()


        if frames is not None and len(frames) > 0:
            # 리스트로 반환될 수 있으므로 numpy 배열로 변환
            frame_list = [np.array(f) for f in frames]
            
            # 그리드 계산 (예: 8개면 2행 4열, 4개면 2행 2열)
            n = len(frame_list)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            
            # 빈 이미지로 채워서 직사각형 만들기
            h, w, c = frame_list[0].shape
            full_grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
            
            for idx, frame in enumerate(frame_list):
                r = idx // cols
                c_idx = idx % cols
                full_grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w, :] = frame

            # OpenCV는 BGR을 사용하므로 RGB -> BGR 변환
            bgr_grid = cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)

            # 화면이 너무 크면 리사이징 (가로 960px 기준)
            max_width = 960
            if bgr_grid.shape[1] > max_width:
                scale = max_width / bgr_grid.shape[1]
                dim = (max_width, int(bgr_grid.shape[0] * scale))
                bgr_grid = cv2.resize(bgr_grid, dim, interpolation=cv2.INTER_AREA)
            
            cv2.imshow(window_name, bgr_grid)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 'q'를 누르면 종료
                break
        # -------------------

        # Epsilon-Greedy Action Selection (병렬 처리)
        if np.random.random() < epsilon:
            actions = env.action_space.sample() 
        else:
            states_tensor = torch.FloatTensor(states)
            with torch.no_grad():
                q_values = q_net(states_tensor)
                actions = q_values.argmax(dim=1).numpy()

        # Step 진행
        next_states, rewards, terminals, truncations, infos = env.step(actions)
        dones = terminals | truncations

        for i in range(config.NUM_ENVS):
            real_next_state = next_states[i]
            if dones[i]:
                 if isinstance(infos, dict) and "final_observation" in infos:
                     real_next_state = infos["final_observation"][i]
                 elif isinstance(infos, list) and "final_observation" in infos[i]:
                      real_next_state = infos[i]["final_observation"]

            memory.push((states[i], actions[i], rewards[i], real_next_state, dones[i]))

        states = next_states
        total_steps += 1 # 스텝 수 증가 (배치 단위로 1스텝 취급)

        # 학습 (Train) - 데이터가 많이 들어오므로 여러 번 업데이트 수행
        if len(memory) >= config.BATCH_SIZE:
            for _ in range(4): # 데이터 수집 속도에 맞춰 4번 정도 업데이트
                batch = memory.sample(config.BATCH_SIZE)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                b_states = torch.FloatTensor(np.array(b_states))
                b_actions = torch.LongTensor(np.array(b_actions)).unsqueeze(1)
                b_rewards = torch.FloatTensor(np.array(b_rewards))
                b_next_states = torch.FloatTensor(np.array(b_next_states))
                b_dones = torch.FloatTensor(np.array(b_dones))

                q_values = q_net(b_states).gather(1, b_actions).squeeze()
                
                # 타겟 네트워크 사용
                with torch.no_grad():
                    next_q = target_q_net(b_next_states).max(1)[0]
                    target = b_rewards + config.GAMMA * next_q * (1 - b_dones)

                loss = loss_fn(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 타겟 네트워크 업데이트
        if total_steps % config.TARGET_UPDATE_FREQ == 0:
            target_q_net.load_state_dict(q_net.state_dict())

        if np.any(dones):
            for i, done in enumerate(dones):
                if done:
                    completed_episodes += 1
                    epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)
                    print(f"Episode {completed_episodes:3d} finished | Epsilon: {epsilon:.3f}")

            if completed_episodes >= config.EPISODES:
                break

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    env.close()
    cv2.destroyAllWindows()
    print("Training finished. Windows closed.")