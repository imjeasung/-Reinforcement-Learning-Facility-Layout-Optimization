import random
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 단계 0: GPU 장치 설정 ---
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
PROCESS_SEQUENCE = [0, 1, 3, 2]
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
    total_distance = 0
    if len(machine_positions) < 2 or len(process_sequence) < 2:
        return 0
    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        if m1_id in machine_positions and m2_id in machine_positions:
            pos1_center_x = machine_positions[m1_id]['center_x']
            pos1_center_y = machine_positions[m1_id]['center_y']
            pos2_center_x = machine_positions[m2_id]['center_x']
            pos2_center_y = machine_positions[m2_id]['center_y']
            distance = math.sqrt((pos1_center_x - pos2_center_x)**2 + (pos1_center_y - pos2_center_y)**2)
            total_distance += distance
        else:
            return float('inf')
    return total_distance

# --- 단계 4: PyTorch 정책 신경망 정의 (device 인자 추가 및 내부 텐서 device 설정) ---
class PolicyNetwork(nn.Module):
    def __init__(self, grid_width, grid_height, num_machine_types, embedding_dim=10): # device 인자 제거 (모델 자체를 .to(device)로 옮김)
        super(PolicyNetwork, self).__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        self.machine_embedding = nn.Embedding(num_machine_types, embedding_dim)
        input_size = grid_width * grid_height + embedding_dim
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_x = nn.Linear(64, grid_width)
        self.output_y = nn.Linear(64, grid_height)

    def get_state_representation(self, current_grid_list, next_machine_id_to_place, device_to_use): # device 인자 추가
        """ 현재 그리드와 다음에 배치할 설비 ID를 지정된 device의 텐서로 변환 """
        grid_array = np.array(current_grid_list, dtype=np.float32).flatten() / (len(machines_definitions) -1)
        # grid_tensor를 생성할 때 바로 device로 보냄
        grid_tensor = torch.FloatTensor(grid_array).to(device_to_use)
        
        # machine_id_tensor를 생성할 때 바로 device로 보냄
        machine_id_tensor = torch.LongTensor([next_machine_id_to_place]).to(device_to_use)
        # self.machine_embedding 레이어가 이미 model.to(device)를 통해 GPU에 있다면,
        # 입력 텐서(machine_id_tensor)도 동일한 장치에 있어야 함.
        machine_embedded = self.machine_embedding(machine_id_tensor).squeeze(0)
        
        state_tensor = torch.cat((grid_tensor, machine_embedded))
        return state_tensor.unsqueeze(0) # 배치 차원 추가

    def forward(self, state_tensor): # state_tensor는 이미 올바른 device에 있다고 가정
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        x_logits = self.output_x(x)
        y_logits = self.output_y(x)
        prob_x = F.softmax(x_logits, dim=-1)
        prob_y = F.softmax(y_logits, dim=-1)
        return prob_x, prob_y

# --- 단계 5: 정책 기반 배치 및 간단한 학습 로직 (device 사용) ---
def place_machines_with_policy_and_learn(policy_net, optimizer, machines_to_place_defs, 
                                         factory_width, factory_height, process_sequence, current_device, gamma=0.99): # current_device 인자 추가
    current_grid_list = initialize_layout_grid(factory_width, factory_height)
    current_machine_positions = {}
    machines_this_episode = [Machine(m['id'], m['name'], m['footprint']) for m in machines_to_place_defs]
    log_probs_x, log_probs_y = [], []
    placement_successful_for_all = True

    for machine_to_place in machines_this_episode:
        # get_state_representation 호출 시 device 전달
        state_tensor = policy_net.get_state_representation(current_grid_list, machine_to_place.id, current_device)
        # state_tensor는 이미 올바른 device에 있음. policy_net도 마찬가지.
        
        prob_x, prob_y = policy_net(state_tensor) # 출력도 policy_net과 동일한 device에 생성됨
        
        try:
            if torch.isnan(prob_x).any() or torch.isinf(prob_x).any() or \
               torch.isnan(prob_y).any() or torch.isinf(prob_y).any():
                action_x_idx = torch.randint(0, factory_width, (1,), device=current_device).item() # device 명시
                action_y_idx = torch.randint(0, factory_height, (1,), device=current_device).item() # device 명시
                log_prob_x = torch.log(torch.tensor(1e-6, device=current_device)) # device 명시
                log_prob_y = torch.log(torch.tensor(1e-6, device=current_device)) # device 명시
            else:
                action_x_idx = torch.multinomial(prob_x.squeeze(0), 1).item()
                action_y_idx = torch.multinomial(prob_y.squeeze(0), 1).item()
                log_prob_x = torch.log(prob_x.squeeze(0)[action_x_idx]) # prob_x가 이미 device에 있으므로 log_prob_x도 동일 device
                log_prob_y = torch.log(prob_y.squeeze(0)[action_y_idx])
        except RuntimeError as e:
             action_x_idx = torch.randint(0, factory_width, (1,), device=current_device).item()
             action_y_idx = torch.randint(0, factory_height, (1,), device=current_device).item()
             log_prob_x = torch.log(torch.tensor(1e-6, device=current_device))
             log_prob_y = torch.log(torch.tensor(1e-6, device=current_device))

        log_probs_x.append(log_prob_x)
        log_probs_y.append(log_prob_y)

        if can_place_machine(current_grid_list, machine_to_place.footprint, action_x_idx, action_y_idx):
            place_machine_on_grid(current_grid_list, machine_to_place.id, machine_to_place.footprint, action_x_idx, action_y_idx)
            machine_to_place.position = (action_x_idx, action_y_idx)
            current_machine_positions[machine_to_place.id] = {
                "x": action_x_idx, "y": action_y_idx,
                "center_x": action_x_idx + machine_to_place.footprint[0] / 2.0,
                "center_y": action_y_idx + machine_to_place.footprint[1] / 2.0
            }
        else:
            placement_successful_for_all = False
    
    final_distance = calculate_total_distance(current_machine_positions, process_sequence)
    reward_val = 0.0 # Python float으로 reward 계산
    if not placement_successful_for_all or len(current_machine_positions) < len(machines_to_place_defs):
        reward_val = -100.0
    elif final_distance == float('inf'):
        reward_val = -50.0
    elif final_distance == 0 and len(machines_to_place_defs) > 1:
        reward_val = -50.0
    else:
        reward_val = 100.0 / (1.0 + final_distance)
    
    # 보상값도 텐서로 만들어서 device로 보낼 수 있지만, 스칼라 곱셈은 CPU 스칼라와 GPU 텐서 간에도 잘 동작하는 경우가 많음.
    # 명시적으로 하려면: reward_tensor = torch.tensor(reward_val, device=current_device)
    # loss += -(log_p_x + log_p_y) * reward_tensor

    loss = torch.tensor(0.0, device=current_device) # loss를 device에 초기화
    if log_probs_x: # 비어있지 않은 경우에만
        for log_p_x, log_p_y in zip(log_probs_x, log_probs_y):
            loss += -(log_p_x + log_p_y) * reward_val # reward_val은 스칼라
    
    if not torch.is_tensor(loss) or loss.item() == 0.0 : #  loss가 0이거나 텐서가 아닌 경우 (log_probs가 비었을때)
        pass # 업데이트 안함
    else:
        optimizer.zero_grad()
        loss.backward() # GPU에서 연산된 loss에 대해 역전파
        optimizer.step()

    return current_grid_list, current_machine_positions, final_distance, reward_val, loss


# --- 단계 6: 메인 실행 블록 (device 설정 반영) ---
if __name__ == '__main__':
    print("설비 배치 최적화 프로그램 (PyTorch + GPU 지원) 시작")

    num_episodes = 600
    learning_rate = 0.005

    num_unique_machines = len(machines_definitions)
    policy_net = PolicyNetwork(FACTORY_WIDTH, FACTORY_HEIGHT, num_unique_machines)
    policy_net.to(device) # <<-- 모델을 설정된 device로 이동 -->>
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    best_overall_distance = float('inf')
    best_layout_info = None

    for episode in range(num_episodes):
        # place_machines_with_policy_and_learn 호출 시 device 전달
        grid, positions, distance, ep_reward, ep_loss = place_machines_with_policy_and_learn(
            policy_net, optimizer, machines_definitions,
            FACTORY_WIDTH, FACTORY_HEIGHT, PROCESS_SEQUENCE,
            device # <<-- 현재 device 전달 -->>
        )
        
        current_loss_val = ep_loss.item() if torch.is_tensor(ep_loss) else ep_loss # ep_loss가 스칼라 0.0일 수 있음

        print(f"에피소드 {episode + 1}/{num_episodes}: "
              f"총 이동 거리 = {distance:.2f}, 보상 = {ep_reward:.2f}, 손실 = {current_loss_val:.3f}")

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
    
    # torch.save(policy_net.state_dict(), "simple_placement_policy_gpu.pth")
    # print("학습된 정책 모델 저장 완료: simple_placement_policy_gpu.pth")

    print("\n프로그램 종료.")