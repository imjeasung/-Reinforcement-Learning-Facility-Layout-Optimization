import random
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- 단계 0: GPU 장치 설정 (이전과 동일) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU 사용 불가능, CPU를 사용합니다.")

# --- 단계 1: 기본 데이터 정의 (cycle_time 추가, 생산 목표 관련 상수 추가) ---
class Machine:
    def __init__(self, id, name, footprint, cycle_time): # cycle_time 추가
        self.id = id
        self.name = name
        self.footprint = footprint
        self.cycle_time = cycle_time  # 단위: 초 (제품 1개 처리 시간)
        self.position = None

machines_definitions = [ # cycle_time 값은 예시입니다.
    {"id": 0, "name": "선반_A", "footprint": (2, 3), "cycle_time": 30},
    {"id": 1, "name": "밀링_B", "footprint": (3, 2), "cycle_time": 45},
    {"id": 2, "name": "검사_C", "footprint": (1, 2), "cycle_time": 15},
    {"id": 3, "name": "조립_D", "footprint": (2, 2), "cycle_time": 60},
]
PROCESS_SEQUENCE = [0, 1, 3, 2]
FACTORY_WIDTH = 10
FACTORY_HEIGHT = 8

# 생산 목표 관련 상수
TARGET_PRODUCTION_PER_HOUR = 60  # 시간당 목표 생산량 (개)
SECONDS_PER_HOUR = 3600
MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND = 0.5 # 초당 이동 거리 단위 (그리드 기준)

# --- 단계 2: 레이아웃 표현 및 배치 함수 (이전과 동일) ---
# initialize_layout_grid, can_place_machine, place_machine_on_grid, print_layout
# 이전 코드와 동일하므로 생략합니다. (필요시 위 코드에서 복사)
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
    transposed_grid = [list(row) for row in zip(*grid)]
    for row in transposed_grid:
        print(" ".join(map(lambda x: f"{x:2d}" if x != -1 else "__", row)))
    if not machine_positions:
        print("배치된 설비 없음")
        return
    print("\n--- 설비 위치 (좌상단, 중심) ---")
    for machine_id, pos_data in machine_positions.items():
        # machines_definitions에서 해당 id의 설비 이름 찾기
        machine_def = next((m for m in machines_definitions if m["id"] == machine_id), None)
        machine_name = machine_def["name"] if machine_def else "알수없음"
        print(f"설비 ID {machine_id} ({machine_name}): 좌상단 ({pos_data['x']}, {pos_data['y']}), 중심 ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")


# --- 단계 3: 평가 함수 (calculate_total_distance는 이전과 동일, estimate_line_throughput 추가) ---
def calculate_total_distance(machine_positions, process_sequence):
    total_distance = 0
    if len(machine_positions) < 2 or len(process_sequence) < 2:
        return 0
    for m_id in process_sequence: # 공정 설비가 모두 배치되었는지 먼저 확인
        if m_id not in machine_positions:
            return float('inf') 
    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        pos1_center_x = machine_positions[m1_id]['center_x']
        pos1_center_y = machine_positions[m1_id]['center_y']
        pos2_center_x = machine_positions[m2_id]['center_x']
        pos2_center_y = machine_positions[m2_id]['center_y']
        distance = math.sqrt((pos1_center_x - pos2_center_x)**2 + (pos1_center_y - pos2_center_y)**2)
        total_distance += distance
    return total_distance

def get_machine_cycle_time(machine_id, all_machines_data):
    """주어진 ID의 설비 cycle_time 반환"""
    for m_def in all_machines_data:
        if m_def["id"] == machine_id:
            return m_def["cycle_time"]
    return float('inf') # 찾을 수 없으면 매우 큰 값

def estimate_line_throughput(machine_positions, process_sequence, all_machines_data, travel_speed):
    """
    배치된 설비 기준 라인의 시간당 예상 생산량을 추정합니다.
    """
    if not machine_positions or not process_sequence:
        return 0.0

    # 공정 순서의 모든 설비가 배치되었는지 확인
    for m_id in process_sequence:
        if m_id not in machine_positions:
            # print(f"경고: 공정 설비 {m_id}가 배치되지 않아 생산량 추정 불가.")
            return 0.0 # 하나라도 없으면 생산 불가

    max_stage_time = 0.0 # 가장 오래 걸리는 공정(병목) 시간

    for i in range(len(process_sequence)):
        current_machine_id = process_sequence[i]
        
        # 1. 현재 설비의 순수 작업 시간 (Cycle Time)
        machine_cycle_time = get_machine_cycle_time(current_machine_id, all_machines_data)
        if machine_cycle_time == float('inf'): return 0.0 # Cycle time 모르면 계산 불가

        # 2. 이전 설비에서 현재 설비까지의 자재 이동 시간
        travel_time = 0.0
        if i > 0: # 첫 번째 설비는 이동 시간 없음
            prev_machine_id = process_sequence[i-1]
            if prev_machine_id in machine_positions and current_machine_id in machine_positions:
                pos_prev = machine_positions[prev_machine_id]
                pos_curr = machine_positions[current_machine_id]
                
                distance = math.sqrt(
                    (pos_prev['center_x'] - pos_curr['center_x'])**2 +
                    (pos_prev['center_y'] - pos_curr['center_y'])**2
                )
                if travel_speed > 0:
                    travel_time = distance / travel_speed
                else: # 이동 속도가 0이면 이동 시간 무한대 (실제로는 발생하지 않도록)
                    travel_time = float('inf') 
        
        current_stage_total_time = machine_cycle_time + travel_time
        
        if current_stage_total_time > max_stage_time:
            max_stage_time = current_stage_total_time

    if max_stage_time == 0 or max_stage_time == float('inf'): # 유효한 병목 시간 계산 불가
        return 0.0
        
    # 시간당 생산량 = 시간(초) / 병목 공정 시간 (초/개)
    throughput_per_hour = SECONDS_PER_HOUR / max_stage_time
    return throughput_per_hour


# --- 단계 4: Actor-Critic 신경망 정의 (이전과 동일) ---
class ActorCriticNetwork(nn.Module):
    # ... (이전 코드와 동일, 생략)
    def __init__(self, grid_width, grid_height, num_machine_types, embedding_dim=10):
        super(ActorCriticNetwork, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        self.machine_embedding = nn.Embedding(num_machine_types, embedding_dim)
        input_size = grid_width * grid_height + embedding_dim
        
        self.fc_shared1 = nn.Linear(input_size, 128)
        self.fc_shared2 = nn.Linear(128, 64)
        
        self.actor_output_x = nn.Linear(64, grid_width)
        self.actor_output_y = nn.Linear(64, grid_height)
        self.critic_output = nn.Linear(64, 1)

    def get_state_representation(self, current_grid_list, next_machine_id_to_place, device_to_use):
        grid_array = np.array(current_grid_list, dtype=np.float32).flatten() / (len(machines_definitions) -1)
        grid_tensor = torch.FloatTensor(grid_array).to(device_to_use)
        machine_id_tensor = torch.LongTensor([next_machine_id_to_place]).to(device_to_use)
        machine_embedded = self.machine_embedding(machine_id_tensor).squeeze(0)
        state_tensor = torch.cat((grid_tensor, machine_embedded))
        return state_tensor.unsqueeze(0)

    def forward(self, state_tensor):
        shared_features = F.relu(self.fc_shared1(state_tensor))
        shared_features = F.relu(self.fc_shared2(shared_features))
        x_logits = self.actor_output_x(shared_features)
        y_logits = self.actor_output_y(shared_features)
        prob_x = F.softmax(x_logits, dim=-1)
        prob_y = F.softmax(y_logits, dim=-1)
        state_value = self.critic_output(shared_features)
        return prob_x, prob_y, state_value


# --- 단계 5: A2C 학습 로직 (보상 함수에 생산량 목표 반영) ---
def run_a2c_episode_and_learn(
    ac_net, optimizer, all_machines_data, # machines_definitions 대신 all_machines_data 사용
    factory_width, factory_height, process_sequence, current_device, 
    target_throughput, travel_speed, # 생산량 목표 및 이동 속도 추가
    gamma=0.99, entropy_coeff=0.01, value_loss_coeff=0.5
):
    current_grid_list = initialize_layout_grid(factory_width, factory_height)
    current_machine_positions = {}
    # all_machines_data에서 현재 에피소드에 사용할 설비 객체 생성
    machines_this_episode = [
        Machine(m['id'], m['name'], m['footprint'], m['cycle_time']) for m in all_machines_data
    ]
    
    log_probs_x_list, log_probs_y_list, values_list, rewards_list, entropies_list = [], [], [], [], []
    num_machines_placed_successfully = 0

    for machine_idx, machine_to_place in enumerate(machines_this_episode):
        state_tensor = ac_net.get_state_representation(current_grid_list, machine_to_place.id, current_device)
        prob_x, prob_y, state_value = ac_net(state_tensor)
        
        dist_x, dist_y = Categorical(prob_x), Categorical(prob_y)
        action_x_idx, action_y_idx = dist_x.sample(), dist_y.sample()
        log_prob_x, log_prob_y = dist_x.log_prob(action_x_idx), dist_y.log_prob(action_y_idx)
        current_entropy = dist_x.entropy() + dist_y.entropy()

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
            step_reward = 0.1 
            num_machines_placed_successfully +=1
            placed_successfully_this_step = True
        else:
            step_reward = -0.5 # 이전보다 페널티 살짝 줄임 (최종 보상이 더 중요)

        rewards_list.append(step_reward)
        log_probs_x_list.append(log_prob_x)
        log_probs_y_list.append(log_prob_y)
        values_list.append(state_value)
        entropies_list.append(current_entropy)

    # --- 에피소드 종료 후 최종 보상 계산 (생산량 반영) ---
    final_reward_component = 0
    estimated_throughput = 0 # 초기화
    final_layout_distance = float('inf') # 초기화

    if num_machines_placed_successfully < len(machines_this_episode):
        final_reward_component = -20.0 # 모든 설비 배치 못하면 큰 페널티
    else:
        # 모든 설비가 성공적으로 배치된 경우에만 생산량 및 거리 계산
        estimated_throughput = estimate_line_throughput(current_machine_positions, process_sequence, all_machines_data, travel_speed)
        final_layout_distance = calculate_total_distance(current_machine_positions, process_sequence) # 참고용 또는 작은 페널티용

        # 1. 생산량 목표 달성도에 따른 보상
        throughput_achievement_ratio = estimated_throughput / target_throughput if target_throughput > 0 else 0
        
        if estimated_throughput == 0 : # 생산량 계산 불가 (예: 일부 설비 누락 등으로 병목 시간 무한대)
            throughput_reward = -15.0
        elif throughput_achievement_ratio >= 1.0: # 목표 달성 또는 초과
            throughput_reward = 10.0 + (throughput_achievement_ratio - 1.0) * 2 # 초과분에 대해 추가 보상
        else: # 목표 미달
            throughput_reward = (throughput_achievement_ratio * 10.0) - 7.0 # 0~1 사이 값 * 10점 만점, 기본 페널티 7점

        # 2. 이동 거리에 대한 작은 페널티 (선택 사항, 이미 이동시간이 생산량에 반영됨)
        # distance_penalty = - (final_layout_distance / (factory_width * factory_height)) * 0.5 # 정규화된 거리에 대한 페널티
        # 여기서는 생산량에 집중하므로 distance_penalty는 일단 생략하거나 매우 작게.
        
        final_reward_component = throughput_reward # + distance_penalty

    if rewards_list: # rewards_list가 비어있지 않다면 마지막 스텝의 보상에 합산
        rewards_list[-1] += final_reward_component
    else: # 모든 스텝이 실패하여 rewards_list가 비었다면 (이론상 발생 어려움)
        # 이 경우에 대한 처리 필요 (예: 에피소드 자체에 매우 낮은 보상 부여)
        pass

    # 어드밴티지 및 손실 계산 (이전 A2C와 동일 로직)
    # ... (이전 코드와 동일, 생략)
    policy_loss_terms = []
    value_loss_terms = []
    R = torch.tensor([0.0], device=current_device)
    
    if not rewards_list: # rewards_list가 비어있으면 학습 불가
        # print("  경고: 보상 리스트가 비어있어 학습을 건너뜁니다.")
        return current_grid_list, current_machine_positions, final_layout_distance, estimated_throughput, 0, torch.tensor(0.0)


    for i in reversed(range(len(rewards_list))):
        R = rewards_list[i] + gamma * R 
        advantage = R - values_list[i]
        policy_loss_terms.append(-(log_probs_x_list[i] + log_probs_y_list[i]) * advantage.detach())
        value_loss_terms.append(F.smooth_l1_loss(values_list[i].squeeze(0), R))
                                           
    optimizer.zero_grad()
    total_policy_loss = torch.stack(policy_loss_terms).sum()
    total_value_loss = torch.stack(value_loss_terms).sum()
    total_entropy = torch.stack(entropies_list).mean()
    loss = total_policy_loss + value_loss_coeff * total_value_loss - entropy_coeff * total_entropy
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("경고: 손실 값에 NaN 또는 Inf 발생. 업데이트 건너뜀.")
        actual_loss_val = float('nan')
    else:
        loss.backward()
        optimizer.step()
        actual_loss_val = loss.item()

    # 반환값에 estimated_throughput 추가
    return current_grid_list, current_machine_positions, final_layout_distance, estimated_throughput, sum(rewards_list), actual_loss_val


# --- 단계 6: 메인 실행 블록 (생산량 목표 반영) ---
if __name__ == '__main__':
    print("설비 배치 최적화 프로그램 (A2C + 생산량 목표) 시작")

    num_episodes = 4000  # 학습 에피소드 수 (더 필요할 수 있음)
    learning_rate = 0.0005 # 학습률 미세 조정
    gamma = 0.99
    entropy_coeff = 0.015 # 엔트로피 계수 미세 조정
    value_loss_coeff = 0.5

    num_unique_machines = len(machines_definitions)
    ac_network = ActorCriticNetwork(FACTORY_WIDTH, FACTORY_HEIGHT, num_unique_machines).to(device)
    optimizer = optim.Adam(ac_network.parameters(), lr=learning_rate)

    best_overall_throughput = -1.0 # 최고 생산량 기록 (음수로 시작)
    best_layout_info_for_throughput = None
    all_episode_total_rewards = []
    all_episode_throughputs = []

    for episode in range(num_episodes):
        grid, positions, distance, throughput, total_ep_reward, ep_loss = run_a2c_episode_and_learn(
            ac_network, optimizer, machines_definitions, # all_machines_data로 전달
            FACTORY_WIDTH, FACTORY_HEIGHT, PROCESS_SEQUENCE, device,
            TARGET_PRODUCTION_PER_HOUR, MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND, # 생산량 목표 및 이동 속도 전달
            gamma, entropy_coeff, value_loss_coeff
        )
        
        # total_ep_reward가 텐서일 수 있으므로 .item() 처리
        current_total_reward_val = total_ep_reward.item() if torch.is_tensor(total_ep_reward) else total_ep_reward
        all_episode_total_rewards.append(current_total_reward_val)
        all_episode_throughputs.append(throughput)


        print(f"에피소드 {episode + 1}/{num_episodes}: "
              f"생산량 = {throughput:.2f} (목표:{TARGET_PRODUCTION_PER_HOUR}), "
              f"이동거리 = {distance:.2f}, 총보상 = {current_total_reward_val:.2f}, 손실 = {ep_loss:.3f}")

        # 최고 생산량 기준으로 최적 레이아웃 업데이트
        if throughput > best_overall_throughput and len(positions) == len(machines_definitions) and distance != float('inf'):
            best_overall_throughput = throughput
            best_layout_info_for_throughput = {
                "grid": copy.deepcopy(grid), 
                "positions": copy.deepcopy(positions), 
                "distance": distance,
                "throughput": throughput
            }
            print(f"  🚀 새로운 최고 생산량 레이아웃! 생산량: {best_overall_throughput:.2f} (목표:{TARGET_PRODUCTION_PER_HOUR}) 🚀")

    print("\n--- 최종 결과 (최고 생산량 기준) ---")
    if best_layout_info_for_throughput:
        print_layout(best_layout_info_for_throughput["grid"], best_layout_info_for_throughput["positions"])
        print(f"최고 시간당 생산량: {best_layout_info_for_throughput['throughput']:.2f} (목표: {TARGET_PRODUCTION_PER_HOUR})")
        print(f"해당 레이아웃의 총 이동 거리: {best_layout_info_for_throughput['distance']:.2f}")
    else:
        print("유효한 최적 레이아웃을 찾지 못했습니다.")
    
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(12, 5))

        color = 'tab:red'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward', color=color)
        ax1.plot([r.cpu().item() if torch.is_tensor(r) else r for r in all_episode_total_rewards], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Throughput', color=color)  # we already handled the x-label with ax1
        ax2.plot(all_episode_throughputs, color=color, linestyle=':')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=TARGET_PRODUCTION_PER_HOUR, color='gray', linestyle='--', label=f'Target Throughput ({TARGET_PRODUCTION_PER_HOUR})')
        ax2.legend(loc='lower right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Episode Rewards & Throughput over Time (A2C + Throughput)')
        plt.savefig('a2c_throughput_rewards.png')
        print("에피소드별 보상 및 생산량 그래프 저장: a2c_throughput_rewards.png")
    except ImportError:
        print("matplotlib 라이브러리가 없어 그래프를 그릴 수 없습니다.")
    except Exception as e:
        print(f"그래프 저장 중 오류 발생: {e}")


    # torch.save(ac_network.state_dict(), "a2c_throughput_placement_policy.pth")
    # print("학습된 A2C 모델(생산량 목표) 저장 완료: a2c_throughput_placement_policy.pth")

    print("\n프로그램 종료.")