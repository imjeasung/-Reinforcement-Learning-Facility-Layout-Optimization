import random
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical # 엔트로피 계산 및 샘플링에 사용

# --- 단계 0: GPU 장치 설정 (이전과 동일) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU 사용 불가능, CPU를 사용합니다.")

# --- 단계 1: 기본 데이터 정의 (이전과 동일) ---
class Machine:
    def __init__(self, id, name, footprint):
        self.id = id
        self.name = name
        self.footprint = footprint
        self.position = None

machines_definitions = [
    {"id": 0, "name": "선반_A", "footprint": (2, 3)},
    {"id": 1, "name": "밀링_B", "footprint": (3, 2)},
    {"id": 2, "name": "검사_C", "footprint": (1, 2)},
    {"id": 3, "name": "조립_D", "footprint": (2, 2)},
]
PROCESS_SEQUENCE = [0, 1, 3, 2] # 설비 ID 기준
FACTORY_WIDTH = 10
FACTORY_HEIGHT = 8

# --- 단계 2: 레이아웃 표현 및 배치 함수 (이전과 동일) ---
def initialize_layout_grid(width, height):
    return [[-1 for _ in range(height)] for _ in range(width)]

def can_place_machine(grid, machine_footprint, x, y):
    width, height = len(grid), len(grid[0])
    m_width, m_height = machine_footprint
    if not (0 <= x and x + m_width <= width and 0 <= y and y + m_height <= height):
        return False
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            if grid[i][j] != -1:
                return False
    return True

def place_machine_on_grid(grid, machine_id, machine_footprint, x, y):
    m_width, m_height = machine_footprint
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            grid[i][j] = machine_id
    return True

def print_layout(grid, machine_positions):
    print("--- 현재 레이아웃 ---")
    # ... (이전 코드와 동일, 생략)
    transposed_grid = [list(row) for row in zip(*grid)]
    for row in transposed_grid:
        print(" ".join(map(lambda x: f"{x:2d}" if x != -1 else "__", row)))
    if not machine_positions:
        print("배치된 설비 없음")
        return
    print("\n--- 설비 위치 (좌상단, 중심) ---")
    for machine_id, pos_data in machine_positions.items():
        machine_name = next(m_def["name"] for m_def in machines_definitions if m_def["id"] == machine_id)
        print(f"설비 ID {machine_id} ({machine_name}): 좌상단 ({pos_data['x']}, {pos_data['y']}), 중심 ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")


# --- 단계 3: 평가 함수 (이전과 동일) ---
def calculate_total_distance(machine_positions, process_sequence):
    # ... (이전 코드와 동일, 생략)
    total_distance = 0
    if len(machine_positions) < 2 or len(process_sequence) < 2:
        return 0 # 혹은 매우 큰 값으로 초기화하여 페널티
    
    # 모든 공정 설비가 배치되었는지 확인
    for m_id in process_sequence:
        if m_id not in machine_positions:
            return float('inf') # 공정 설비 중 하나라도 없으면 최악

    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        # 위에서 이미 확인했으므로 machine_positions에 m1_id, m2_id가 존재한다고 가정 가능
        pos1_center_x = machine_positions[m1_id]['center_x']
        pos1_center_y = machine_positions[m1_id]['center_y']
        pos2_center_x = machine_positions[m2_id]['center_x']
        pos2_center_y = machine_positions[m2_id]['center_y']
        distance = math.sqrt((pos1_center_x - pos2_center_x)**2 + (pos1_center_y - pos2_center_y)**2)
        total_distance += distance
    return total_distance

# --- 단계 4: Actor-Critic 신경망 정의 ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, grid_width, grid_height, num_machine_types, embedding_dim=10):
        super(ActorCriticNetwork, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # 공유 레이어
        self.machine_embedding = nn.Embedding(num_machine_types, embedding_dim)
        input_size = grid_width * grid_height + embedding_dim # Flattened grid + machine embedding
        
        self.fc_shared1 = nn.Linear(input_size, 128)
        self.fc_shared2 = nn.Linear(128, 64)
        
        # 액터 헤드 (정책 결정)
        self.actor_output_x = nn.Linear(64, grid_width)  # X 좌표에 대한 logits
        self.actor_output_y = nn.Linear(64, grid_height) # Y 좌표에 대한 logits
        
        # 크리틱 헤드 (상태 가치 평가)
        self.critic_output = nn.Linear(64, 1) # 현재 상태의 가치 (스칼라)

    def get_state_representation(self, current_grid_list, next_machine_id_to_place, device_to_use):
        grid_array = np.array(current_grid_list, dtype=np.float32).flatten() / (len(machines_definitions) - 1) # 정규화
        grid_tensor = torch.FloatTensor(grid_array).to(device_to_use)
        machine_id_tensor = torch.LongTensor([next_machine_id_to_place]).to(device_to_use)
        machine_embedded = self.machine_embedding(machine_id_tensor).squeeze(0)
        state_tensor = torch.cat((grid_tensor, machine_embedded))
        return state_tensor.unsqueeze(0)

    def forward(self, state_tensor):
        # 공유 레이어 통과
        shared_features = F.relu(self.fc_shared1(state_tensor))
        shared_features = F.relu(self.fc_shared2(shared_features))
        
        # 액터 출력 (행동 확률)
        x_logits = self.actor_output_x(shared_features)
        y_logits = self.actor_output_y(shared_features)
        prob_x = F.softmax(x_logits, dim=-1)
        prob_y = F.softmax(y_logits, dim=-1)
        
        # 크리틱 출력 (상태 가치)
        state_value = self.critic_output(shared_features)
        
        return prob_x, prob_y, state_value

# --- 단계 5: A2C 학습 로직 ---
def run_a2c_episode_and_learn(
    ac_net, optimizer, machines_to_place_defs, 
    factory_width, factory_height, process_sequence, current_device, 
    gamma=0.99, entropy_coeff=0.01, value_loss_coeff=0.5
):
    current_grid_list = initialize_layout_grid(factory_width, factory_height)
    current_machine_positions = {}
    machines_this_episode = [Machine(m['id'], m['name'], m['footprint']) for m in machines_to_place_defs]
    
    # 에피소드 동안의 기록 저장용 리스트
    log_probs_x_list = []
    log_probs_y_list = []
    values_list = []
    rewards_list = []
    entropies_list = [] # 각 스텝의 정책 엔트로피 저장

    num_machines_placed_successfully = 0

    # 각 스텝 (설비 하나 배치) 진행
    for machine_idx, machine_to_place in enumerate(machines_this_episode):
        state_tensor = ac_net.get_state_representation(current_grid_list, machine_to_place.id, current_device)
        prob_x, prob_y, state_value = ac_net(state_tensor) # prob_x, prob_y, V(s)
        
        # 행동 샘플링 및 로그 확률, 엔트로피 계산
        dist_x = Categorical(prob_x) # x좌표에 대한 분포
        dist_y = Categorical(prob_y) # y좌표에 대한 분포
        
        action_x_idx = dist_x.sample()
        action_y_idx = dist_y.sample()
        
        log_prob_x = dist_x.log_prob(action_x_idx)
        log_prob_y = dist_y.log_prob(action_y_idx)
        
        entropy_x = dist_x.entropy()
        entropy_y = dist_y.entropy()
        current_entropy = entropy_x + entropy_y

        # 선택된 행동으로 설비 배치 시도
        step_reward = 0
        placed_successfully_this_step = False
        if can_place_machine(current_grid_list, machine_to_place.footprint, action_x_idx.item(), action_y_idx.item()):
            place_machine_on_grid(current_grid_list, machine_to_place.id, machine_to_place.footprint, action_x_idx.item(), action_y_idx.item())
            machine_to_place.position = (action_x_idx.item(), action_y_idx.item())
            current_machine_positions[machine_to_place.id] = {
                "x": action_x_idx.item(), "y": action_y_idx.item(),
                "center_x": action_x_idx.item() + machine_to_place.footprint[0] / 2.0,
                "center_y": action_y_idx.item() + machine_to_place.footprint[1] / 2.0
            }
            step_reward = 0.1  # 배치 성공 시 작은 긍정적 보상
            num_machines_placed_successfully +=1
            placed_successfully_this_step = True
        else:
            step_reward = -1.0 # 배치 실패 시 페널티

        rewards_list.append(step_reward)
        log_probs_x_list.append(log_prob_x)
        log_probs_y_list.append(log_prob_y)
        values_list.append(state_value)
        entropies_list.append(current_entropy)

        if not placed_successfully_this_step and machine_idx < len(machines_this_episode) -1 : # 마지막 설비가 아닌데 배치 실패하면 다음 설비로 못감
             # print(f"  중간 설비 {machine_to_place.name} 배치 실패. 에피소드 조기 종료 가능성.")
             # 이 경우, 남은 스텝들은 진행하지 않거나, 매우 큰 페널티를 주고 종료할 수 있음
             # 여기서는 일단 계속 진행하되, 최종 보상에서 페널티
             pass


    # --- 에피소드 종료 후 학습 단계 ---
    # 최종 레이아웃에 대한 추가 보상 (마지막 스텝의 보상에 합산)
    final_layout_distance = calculate_total_distance(current_machine_positions, process_sequence)
    
    if num_machines_placed_successfully < len(machines_to_place_defs):
        rewards_list[-1] += -50.0 # 모든 설비 배치 못하면 큰 페널티 (마지막 스텝 보상에 추가)
    elif final_layout_distance == float('inf'):
        rewards_list[-1] += -20.0 # 일부 누락으로 거리 계산 불가
    else:
        # 거리가 짧을수록 보상. (최대 가능 거리 - 현재 거리) 또는 (1 / (1 + 거리)) 등
        # 여기서는 최대 이동 거리 (대각선) 대비 얼마나 줄였는지로 단순화
        max_possible_dist = math.sqrt(factory_width**2 + factory_height**2) * len(process_sequence)
        reward_from_distance = (max_possible_dist - final_layout_distance) / max_possible_dist * 10 # 스케일링
        rewards_list[-1] += reward_from_distance


    # 어드밴티지 및 손실 계산
    policy_loss_terms = []
    value_loss_terms = []
    
    # 다음 상태의 가치 (마지막 스텝 이후이므로 0으로 가정, 또는 마지막 상태의 가치를 예측)
    # 마지막 상태에 대한 예측은 없으므로, 마지막 보상 이후의 가치는 0으로.
    R = torch.tensor([0.0], device=current_device) # 누적 보상 ( discounted return )
    
    # 뒤에서부터 계산
    for i in reversed(range(len(rewards_list))):
        R = rewards_list[i] + gamma * R # 현재 스텝 i에서의 discounted return
        advantage = R - values_list[i] # A(s_i, a_i) = G_i - V(s_i)
        
        # 정책 손실 (액터)
        # 어드밴티지에 .detach()를 사용하여 크리틱으로 그래디언트가 흐르지 않도록 함
        policy_loss_terms.append(-(log_probs_x_list[i] + log_probs_y_list[i]) * advantage.detach())
        
        # 가치 손실 (크리틱) - MSE 또는 SmoothL1Loss
        # R은 텐서로 변환 필요. values_list[i]도 텐서.
        value_loss_terms.append(F.smooth_l1_loss(values_list[i].squeeze(0), R)) # R은 이미 텐서
                                           
    # 손실 합산 및 역전파
    optimizer.zero_grad()
    
    total_policy_loss = torch.stack(policy_loss_terms).sum()
    total_value_loss = torch.stack(value_loss_terms).sum()
    total_entropy = torch.stack(entropies_list).mean() # 평균 엔트로피

    # 전체 손실: 정책 손실 + 가치 손실 (가중치 적용) - 엔트로피 보너스 (탐험 장려)
    loss = total_policy_loss + value_loss_coeff * total_value_loss - entropy_coeff * total_entropy
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("경고: 손실 값에 NaN 또는 Inf 발생. 업데이트 건너뜀.")
    else:
        loss.backward()
        optimizer.step()

    return current_grid_list, current_machine_positions, final_layout_distance, sum(rewards_list), loss.item()


# --- 단계 6: 메인 실행 블록 (A2C 적용) ---
if __name__ == '__main__':
    print("설비 배치 최적화 프로그램 (A2C 적용) 시작")

    num_episodes = 1500  # 학습 에피소드 수 (A2C는 더 많은 에피소드 필요 가능성)
    learning_rate = 0.001 # 학습률 조정 가능
    gamma = 0.99         # 할인 계수
    entropy_coeff = 0.015 # 엔트로피 계수 (탐험 장려)
    value_loss_coeff = 0.5 # 가치 손실 가중치

    num_unique_machines = len(machines_definitions)
    # ActorCriticNetwork 사용
    ac_network = ActorCriticNetwork(FACTORY_WIDTH, FACTORY_HEIGHT, num_unique_machines).to(device)
    optimizer = optim.Adam(ac_network.parameters(), lr=learning_rate)

    best_overall_distance = float('inf')
    best_layout_info = None
    all_episode_rewards = []

    for episode in range(num_episodes):
        grid, positions, distance, total_ep_reward, ep_loss = run_a2c_episode_and_learn(
            ac_network, optimizer, machines_definitions,
            FACTORY_WIDTH, FACTORY_HEIGHT, PROCESS_SEQUENCE, device,
            gamma, entropy_coeff, value_loss_coeff
        )
        all_episode_rewards.append(total_ep_reward)

        print(f"에피소드 {episode + 1}/{num_episodes}: "
              f"총 이동 거리 = {distance:.2f}, 총 보상 = {total_ep_reward:.2f}, 손실 = {ep_loss:.3f}")

        if distance < best_overall_distance and distance != float('inf') and len(positions) == len(machines_definitions):
            best_overall_distance = distance
            best_layout_info = {"grid": copy.deepcopy(grid), "positions": copy.deepcopy(positions), "distance": distance}
            print(f"  *** 새로운 최적 레이아웃 발견 (에피소드 {episode+1})! 거리: {best_overall_distance:.2f} ***")

    print("\n--- 최종 결과 ---")
    if best_layout_info:
        print_layout(best_layout_info["grid"], best_layout_info["positions"])
        print(f"가장 짧은 총 이동 거리: {best_layout_info['distance']:.2f}")
    else:
        print("유효한 최적 레이아웃을 찾지 못했습니다.")
    
    # 간단한 보상 그래프 출력 (matplotlib 필요)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot([r.cpu().item() if torch.is_tensor(r) else r for r in all_episode_rewards]) # 텐서면 .cpu().item()
        plt.title('Episode Rewards over Time (A2C)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('a2c_rewards.png')
        print("에피소드별 보상 그래프 저장: a2c_rewards.png")
        # plt.show() # 주석 해제 시 그래프 창 표시
    except ImportError:
        print("matplotlib 라이브러리가 없어 보상 그래프를 그릴 수 없습니다. (pip install matplotlib)")


    # torch.save(ac_network.state_dict(), "a2c_placement_policy.pth")
    # print("학습된 A2C 모델 저장 완료: a2c_placement_policy.pth")

    print("\n프로그램 종료.")