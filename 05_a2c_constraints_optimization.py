import random
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# --- 단계 0: GPU 장치 설정 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU 사용 불가능, CPU를 사용합니다.")

# --- 단계 1: 기본 데이터 정의 (제약 조건 속성 추가) ---
class Machine:
    def __init__(self, id, name, footprint, cycle_time, clearance=0, wall_affinity=False):
        self.id = id
        self.name = name
        self.footprint = footprint
        self.cycle_time = cycle_time
        self.clearance = clearance
        self.wall_affinity = wall_affinity
        self.position = None

machines_definitions = [
    {"id": 0, "name": "선반_A",   "footprint": (2, 3), "cycle_time": 30, "clearance": 1, "wall_affinity": True},
    {"id": 1, "name": "밀링_B",   "footprint": (3, 2), "cycle_time": 45, "clearance": 1, "wall_affinity": False},
    {"id": 2, "name": "검사_C",   "footprint": (1, 2), "cycle_time": 15, "clearance": 0, "wall_affinity": False},
    {"id": 3, "name": "조립_D",   "footprint": (2, 2), "cycle_time": 60, "clearance": 1, "wall_affinity": True},
]
PROCESS_SEQUENCE = [0, 1, 3, 2]
FACTORY_WIDTH = 10
FACTORY_HEIGHT = 8

TARGET_PRODUCTION_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND = 0.5

# --- 단계 2: 레이아웃 표현 및 배치 함수 ---
def initialize_layout_grid(width, height):
    """공장 바닥을 나타내는 빈 그리드를 생성합니다 (-1은 빈 공간)."""
    return [[-1 for _ in range(height)] for _ in range(width)]

def can_place_machine(grid, machine_footprint, machine_clearance, x, y, factory_w, factory_h):
    """주어진 위치 (x,y)에 설비(clearance 포함)를 배치할 수 있는지 확인합니다."""
    m_width, m_height = machine_footprint

    # 1. 설비 자체가 공장 경계 내에 있는지 확인
    if not (0 <= x and x + m_width <= factory_w and 0 <= y and y + m_height <= factory_h):
        return False

    # 2. Clearance를 포함한 영역이 다른 설비와 겹치는지 확인
    # 확인 범위: 실제 설비 위치 (x,y)를 기준으로 clearance 만큼 확장된 영역
    min_x_check = x - machine_clearance
    max_x_check_exclusive = x + m_width + machine_clearance # range 끝값은 포함 안되므로 +1 안함
    min_y_check = y - machine_clearance
    max_y_check_exclusive = y + m_height + machine_clearance

    for i in range(max(0, min_x_check), min(factory_w, max_x_check_exclusive)):
        for j in range(max(0, min_y_check), min(factory_h, max_y_check_exclusive)):
            # 현재 검사하는 (i, j)가 실제 설비가 "차지할" 영역인지 구분
            is_machine_body_area = (x <= i < x + m_width) and (y <= j < y + m_height)

            if is_machine_body_area:
                # 설비 몸체가 놓일 부분: 다른 설비가 있으면 안됨 (-1 이어야 함)
                if grid[i][j] != -1:
                    return False
            else:
                # 순수 clearance 영역 (설비 몸체 제외): 다른 설비가 있으면 안됨 (-1 이어야 함)
                if grid[i][j] != -1:
                    return False
    return True

def place_machine_on_grid(grid, machine_id, machine_footprint, x, y):
    """그리드의 (x,y) 위치에 설비를 배치하고, 점유된 칸에 설비 ID를 표시합니다."""
    m_width, m_height = machine_footprint
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            grid[i][j] = machine_id
    return True

def print_layout(grid, machine_positions):
    """현재 레이아웃과 설비 위치를 보기 좋게 출력합니다."""
    print("--- 현재 레이아웃 ---")
    transposed_grid = [list(row) for row in zip(*grid)]
    for row_idx, row_data in enumerate(transposed_grid):
        # 그리드 Y축 인덱스와 함께 출력
        print(f"Y{row_idx: <2}| " + " ".join(map(lambda val: f"{val:2d}" if val != -1 else "__", row_data)))
    # 그리드 X축 인덱스 출력
    x_indices = "    " + " ".join(f"X{i:<2}" for i in range(len(grid)))
    print(x_indices)
    
    if not machine_positions:
        print("배치된 설비 없음")
        return
        
    print("\n--- 설비 위치 (좌상단, 중심) ---")
    for machine_id, pos_data in machine_positions.items():
        machine_def = next((m for m in machines_definitions if m["id"] == machine_id), None)
        machine_name = machine_def["name"] if machine_def else "알수없음"
        print(f"설비 ID {machine_id} ({machine_name}): 좌상단 ({pos_data['x']}, {pos_data['y']}), 중심 ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")

def is_near_wall(x, y, machine_footprint, factory_w, factory_h):
    """설비가 벽에 인접해 있는지 확인합니다."""
    m_width, m_height = machine_footprint
    if x == 0 or (x + m_width == factory_w):
        return True
    if y == 0 or (y + m_height == factory_h):
        return True
    return False

# --- 단계 3: 평가 함수 ---
def calculate_total_distance(machine_positions, process_sequence):
    total_distance = 0
    if len(machine_positions) < 2 or len(process_sequence) < 2:
        return 0.0
    
    for m_id in process_sequence:
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
    for m_def in all_machines_data:
        if m_def["id"] == machine_id:
            return m_def["cycle_time"]
    return float('inf')

def estimate_line_throughput(machine_positions, process_sequence, all_machines_data, travel_speed):
    if not machine_positions or not process_sequence:
        return 0.0

    for m_id in process_sequence:
        if m_id not in machine_positions:
            return 0.0

    max_stage_time = 0.0
    for i in range(len(process_sequence)):
        current_machine_id = process_sequence[i]
        machine_cycle_time = get_machine_cycle_time(current_machine_id, all_machines_data)
        if machine_cycle_time == float('inf'): return 0.0

        travel_time = 0.0
        if i > 0:
            prev_machine_id = process_sequence[i-1]
            # 이전 설비와 현재 설비가 모두 배치된 경우에만 이동 시간 계산
            if prev_machine_id in machine_positions and current_machine_id in machine_positions:
                pos_prev = machine_positions[prev_machine_id]
                pos_curr = machine_positions[current_machine_id]
                distance = math.sqrt(
                    (pos_prev['center_x'] - pos_curr['center_x'])**2 +
                    (pos_prev['center_y'] - pos_curr['center_y'])**2
                )
                if travel_speed > 0:
                    travel_time = distance / travel_speed
                else:
                    travel_time = float('inf') 
        
        current_stage_total_time = machine_cycle_time + travel_time
        if current_stage_total_time > max_stage_time:
            max_stage_time = current_stage_total_time

    if max_stage_time <= 0 or max_stage_time == float('inf'): # 병목 시간이 0이거나 무한대면 생산 불가
        return 0.0
        
    throughput_per_hour = SECONDS_PER_HOUR / max_stage_time
    return throughput_per_hour

# --- 단계 4: Actor-Critic 신경망 정의 ---
class ActorCriticNetwork(nn.Module):
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
        # 그리드 값의 범위를 고려하여 정규화 (예: -1 ~ num_machines-1)
        # 여기서는 간단히 0~1 사이로 만들기 위해 (설비 ID + 1) / (설비 종류 수)
        # 빈칸(-1)은 0으로, 설비 ID는 (ID+1)로 매핑
        normalized_grid_array = np.array(current_grid_list, dtype=np.float32)
        normalized_grid_array = (normalized_grid_array + 1) / (len(machines_definitions)) # 0 ~ 1 범위로 정규화
        grid_tensor = torch.FloatTensor(normalized_grid_array.flatten()).to(device_to_use)
        
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

# --- 단계 5: A2C 학습 로직 ---
def run_a2c_episode_and_learn(
    ac_net, optimizer, all_machines_data,
    factory_width, factory_height, process_sequence, current_device,
    target_throughput, travel_speed,
    gamma=0.99, entropy_coeff=0.01, value_loss_coeff=0.5, wall_affinity_penalty_coeff=0.05
):
    current_grid_list = initialize_layout_grid(factory_width, factory_height)
    current_machine_positions = {}
    machines_this_episode = [
        Machine(m['id'], m['name'], m['footprint'], m['cycle_time'], 
                m.get('clearance', 0), m.get('wall_affinity', False))
        for m in all_machines_data
    ]
    
    log_probs_x_list, log_probs_y_list, values_list, rewards_list, entropies_list = [], [], [], [], []
    num_machines_placed_successfully = 0

    for machine_idx, machine_to_place in enumerate(machines_this_episode):
        state_tensor = ac_net.get_state_representation(current_grid_list, machine_to_place.id, current_device)
        prob_x, prob_y, state_value = ac_net(state_tensor)
        
        dist_x, dist_y = Categorical(prob_x), Categorical(prob_y)
        action_x_idx, action_y_idx = dist_x.sample(), dist_y.sample() # 텐서 형태로 반환됨
        log_prob_x, log_prob_y = dist_x.log_prob(action_x_idx), dist_y.log_prob(action_y_idx)
        current_entropy = dist_x.entropy() + dist_y.entropy()

        step_reward = 0
        # can_place_machine 호출 시 clearance와 factory_width/height 전달
        if can_place_machine(current_grid_list, machine_to_place.footprint, machine_to_place.clearance,
                             action_x_idx.item(), action_y_idx.item(), factory_width, factory_height):
            place_machine_on_grid(current_grid_list, machine_to_place.id, machine_to_place.footprint, action_x_idx.item(), action_y_idx.item())
            current_machine_positions[machine_to_place.id] = {
                "x": action_x_idx.item(), "y": action_y_idx.item(),
                "center_x": action_x_idx.item() + machine_to_place.footprint[0] / 2.0,
                "center_y": action_y_idx.item() + machine_to_place.footprint[1] / 2.0
            }
            step_reward = 0.1 
            num_machines_placed_successfully +=1
            
            if machine_to_place.wall_affinity:
                if not is_near_wall(action_x_idx.item(), action_y_idx.item(), machine_to_place.footprint, factory_width, factory_height):
                    step_reward -= wall_affinity_penalty_coeff
        else:
            step_reward = -0.5 

        rewards_list.append(step_reward)
        log_probs_x_list.append(log_prob_x)
        log_probs_y_list.append(log_prob_y)
        values_list.append(state_value) # state_value는 (1,1) 형태의 텐서일 수 있음
        entropies_list.append(current_entropy) # current_entropy는 (1,) 형태의 텐서일 수 있음

    final_reward_component = 0
    estimated_throughput = 0.0
    final_layout_distance = float('inf')

    if num_machines_placed_successfully < len(machines_this_episode):
        final_reward_component = -20.0
    else:
        estimated_throughput = estimate_line_throughput(current_machine_positions, process_sequence, all_machines_data, travel_speed)
        final_layout_distance = calculate_total_distance(current_machine_positions, process_sequence)
        
        throughput_achievement_ratio = estimated_throughput / target_throughput if target_throughput > 0 else 0.0
        if estimated_throughput <= 0 : # 생산량 0 또는 계산 불가
            throughput_reward = -15.0
        elif throughput_achievement_ratio >= 1.0:
            throughput_reward = 10.0 + (throughput_achievement_ratio - 1.0) * 2.0
        else: 
            throughput_reward = (throughput_achievement_ratio * 10.0) - 7.0
        final_reward_component = throughput_reward

    if rewards_list:
        rewards_list[-1] += final_reward_component
    
    policy_loss_terms = []
    value_loss_terms = []
    R = torch.tensor([0.0], device=current_device)
    
    if not rewards_list: 
        return current_grid_list, current_machine_positions, final_layout_distance, estimated_throughput, 0.0, torch.tensor(0.0)

    for i in reversed(range(len(rewards_list))):
        R = torch.tensor(rewards_list[i], device=current_device) + gamma * R # R을 스칼라 텐서로 만듦
        advantage = R - values_list[i].squeeze() # values_list[i]는 (1,1) or (1)
        
        policy_loss_terms.append(-(log_probs_x_list[i].squeeze() + log_probs_y_list[i].squeeze()) * advantage.detach())
        value_loss_terms.append(F.smooth_l1_loss(values_list[i].squeeze(), R.squeeze())) # squeeze()로 스칼라 텐서로 만듦
                                           
    optimizer.zero_grad()
    # 각 리스트의 텐서들을 .squeeze()로 스칼라 텐서로 만든 후 stack
    total_policy_loss = torch.stack([p.squeeze() for p in policy_loss_terms]).sum()
    total_value_loss = torch.stack([v.squeeze() for v in value_loss_terms]).sum()
    total_entropy = torch.stack([e.squeeze() for e in entropies_list]).mean()

    loss = total_policy_loss + value_loss_coeff * total_value_loss - entropy_coeff * total_entropy
    
    actual_loss_val = float('nan')
    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        actual_loss_val = loss.item()
    else:
        print("경고: 손실 값에 NaN 또는 Inf 발생. 업데이트 건너뜀.")


    total_reward_sum = sum(rewards_list) # rewards_list는 파이썬 float 리스트
    return current_grid_list, current_machine_positions, final_layout_distance, estimated_throughput, total_reward_sum, actual_loss_val

# --- 단계 6: 메인 실행 블록 ---
if __name__ == '__main__':
    print("설비 배치 최적화 프로그램 (A2C + 생산량 목표 + 제약조건) 시작")

    num_episodes = 3000
    learning_rate = 0.0005 
    gamma = 0.99
    entropy_coeff = 0.015 
    value_loss_coeff = 0.5
    wall_affinity_penalty_coeff = 0.1

    num_unique_machines = len(machines_definitions) 
    ac_network = ActorCriticNetwork(FACTORY_WIDTH, FACTORY_HEIGHT, num_unique_machines).to(device)
    optimizer = optim.Adam(ac_network.parameters(), lr=learning_rate)

    best_overall_throughput = -1.0 
    best_layout_info_for_throughput = None
    all_episode_total_rewards = []
    all_episode_throughputs = []

    for episode in range(num_episodes):
        grid, positions, distance, throughput, total_ep_reward, ep_loss = run_a2c_episode_and_learn(
            ac_network, optimizer, machines_definitions,
            FACTORY_WIDTH, FACTORY_HEIGHT, PROCESS_SEQUENCE, device,
            TARGET_PRODUCTION_PER_HOUR, MATERIAL_TRAVEL_SPEED_UNITS_PER_SECOND,
            gamma, entropy_coeff, value_loss_coeff, wall_affinity_penalty_coeff
        )
        
        all_episode_total_rewards.append(total_ep_reward) # total_ep_reward는 파이썬 float
        all_episode_throughputs.append(throughput)

        print(f"에피소드 {episode + 1}/{num_episodes}: "
              f"생산량 = {throughput:.2f} (목표:{TARGET_PRODUCTION_PER_HOUR}), "
              f"이동거리 = {distance:.2f}, 총보상 = {total_ep_reward:.2f}, 손실 = {ep_loss:.3f}")

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
        print("\n--- 최종 레이아웃 제약 조건 만족도 (예시) ---")
        for m_id_key in best_layout_info_for_throughput["positions"]:
            pos_data = best_layout_info_for_throughput["positions"][m_id_key]
            m_def = next((m for m in machines_definitions if m["id"] == m_id_key), None)
            if m_def and m_def.get('wall_affinity', False):
                is_wall = is_near_wall(pos_data['x'], pos_data['y'], m_def['footprint'], FACTORY_WIDTH, FACTORY_HEIGHT)
                print(f"설비 {m_def['name']}(ID:{m_id_key}): 벽 근접 선호 - {'만족' if is_wall else '불만족'}")
    else:
        print("유효한 최적 레이아웃을 찾지 못했습니다.")
    
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(12, 5))

        color = 'tab:red'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward', color=color)
        ax1.plot(all_episode_total_rewards, color=color) # all_episode_total_rewards는 파이썬 float 리스트
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Throughput', color=color)
        ax2.plot(all_episode_throughputs, color=color, linestyle=':')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=TARGET_PRODUCTION_PER_HOUR, color='gray', linestyle='--', label=f'Target ({TARGET_PRODUCTION_PER_HOUR})')
        ax2.legend(loc='lower right')

        fig.tight_layout()
        plt.title('Episode Rewards & Throughput (A2C + Constraints)')
        plt.savefig('a2c_constraints_rewards_throughput.png')
        print("에피소드별 보상 및 생산량 그래프 저장: a2c_constraints_rewards_throughput.png")
    except ImportError:
        print("matplotlib 라이브러리가 없어 그래프를 그릴 수 없습니다. (pip install matplotlib)")
    except Exception as e:
        print(f"그래프 저장 중 오류 발생: {e}")

    # torch.save(ac_network.state_dict(), "a2c_constraints_placement_policy.pth")
    # print("학습된 A2C 모델(제약조건 포함) 저장 완료: a2c_constraints_placement_policy.pth")

    print("\n프로그램 종료.")