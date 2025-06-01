import random
import math
import copy # 객체 복사를 위해 사용

# --- 단계 1: 기본 데이터 정의 ---

class Machine:
    def __init__(self, id, name, footprint):
        self.id = id
        self.name = name
        self.footprint = footprint  # (가로, 세로) 크기 (단위: 칸 수 또는 m)
        self.position = None        # (x, y) 좌표, 배치 후 할당됨

# 예시 설비 리스트 (이전보다 단순화)
# footprint를 그리드 칸 수로 가정 (예: (2,3)은 가로 2칸, 세로 3칸)
machines_definitions = [
    {"id": 0, "name": "선반_A", "footprint": (2, 3)},
    {"id": 1, "name": "밀링_B", "footprint": (3, 2)},
    {"id": 2, "name": "검사_C", "footprint": (1, 2)},
    {"id": 3, "name": "조립_D", "footprint": (2, 2)},
]

# 공정 순서 (제품 A: 0 -> 1 -> 3 -> 2) - machine_id 기준
PROCESS_SEQUENCE = [0, 1, 3, 2]

# 공장 레이아웃 크기 (그리드 칸 수)
FACTORY_WIDTH = 10  # 가로 칸 수
FACTORY_HEIGHT = 8  # 세로 칸 수

# --- 단계 2: 레이아웃 표현 및 배치 함수 ---

def initialize_layout_grid(width, height):
    """공장 바닥을 나타내는 빈 그리드를 생성합니다 (-1은 빈 공간)."""
    return [[-1 for _ in range(height)] for _ in range(width)]

def can_place_machine(grid, machine_footprint, x, y):
    """주어진 위치 (x,y)에 설비를 배치할 수 있는지 확인합니다."""
    width, height = len(grid), len(grid[0])
    m_width, m_height = machine_footprint

    # 1. 공장 경계 확인
    if not (0 <= x and x + m_width <= width and 0 <= y and y + m_height <= height):
        return False

    # 2. 다른 설비와 겹치는지 확인 (그리드 값이 -1이 아니면 이미 점유)
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            if grid[i][j] != -1:
                return False
    return True

def place_machine_on_grid(grid, machine_id, machine_footprint, x, y):
    """그리드의 (x,y) 위치에 설비를 배치하고, 점유된 칸에 설비 ID를 표시합니다."""
    m_width, m_height = machine_footprint
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            grid[i][j] = machine_id # 설비 ID로 점유 표시
    return True # 실제로는 이미 can_place_machine으로 확인했으므로 항상 성공

def print_layout(grid, machine_positions):
    """현재 레이아웃과 설비 위치를 보기 좋게 출력합니다."""
    print("--- 현재 레이아웃 ---")
    # 그리드를 보기 좋게 출력 (0,0이 좌상단이라고 가정)
    # 실제 화면 출력 시에는 y축을 뒤집거나, 인덱스 순서대로 출력
    transposed_grid = [list(row) for row in zip(*grid)] # 보기 편하게 행/열 전환
    for row in transposed_grid: # y축부터 출력
        print(" ".join(map(lambda x: f"{x:2d}" if x != -1 else "__", row)))
    
    print("\n--- 설비 위치 (중심점 또는 좌상단점 기준) ---")
    if not machine_positions:
        print("배치된 설비 없음")
        return
        
    for machine_id, pos_data in machine_positions.items():
        machine_name = next(m_def["name"] for m_def in machines_definitions if m_def["id"] == machine_id)
        print(f"설비 ID {machine_id} ({machine_name}): 좌상단 ({pos_data['x']}, {pos_data['y']}), 중심 ({pos_data['center_x']:.1f}, {pos_data['center_y']:.1f})")


# --- 단계 3: 평가 함수 ---

def calculate_total_distance(machine_positions, process_sequence):
    """
    주어진 공정 순서에 따라 설비 간 총 이동 거리를 계산합니다.
    설비 위치는 각 설비의 중심점을 기준으로 합니다.
    """
    total_distance = 0
    if len(machine_positions) < 2 or len(process_sequence) < 2 : # 설비가 2개 미만이거나 공정 순서가 너무 짧으면 거리 0
        return 0

    for i in range(len(process_sequence) - 1):
        m1_id = process_sequence[i]
        m2_id = process_sequence[i+1]

        if m1_id in machine_positions and m2_id in machine_positions:
            pos1_center_x = machine_positions[m1_id]['center_x']
            pos1_center_y = machine_positions[m1_id]['center_y']
            pos2_center_x = machine_positions[m2_id]['center_x']
            pos2_center_y = machine_positions[m2_id]['center_y']
            
            # 유클리드 거리
            distance = math.sqrt((pos1_center_x - pos2_center_x)**2 + (pos1_center_y - pos2_center_y)**2)
            total_distance += distance
        else:
            # 하나라도 설비가 배치되지 않았다면 매우 큰 페널티 (또는 예외 처리)
            return float('inf') 
            
    return total_distance

# --- 단계 4: 간단한 배치 전략 및 실행 ---

def generate_random_valid_layout(machines_to_place_defs, factory_width, factory_height):
    """무작위로 유효한 위치에 설비들을 배치하는 하나의 레이아웃을 생성합니다."""
    current_grid = initialize_layout_grid(factory_width, factory_height)
    current_machine_positions = {} # {machine_id: {"x": x, "y": y, "center_x": cx, "center_y": cy}}

    # 정의된 설비 객체 생성 (복사해서 사용)
    machines_to_place = [Machine(m['id'], m['name'], m['footprint']) for m in machines_to_place_defs]

    for machine in machines_to_place: # 모든 설비를 배치 시도
        placed = False
        # 최대 100번 랜덤 위치 시도
        for _ in range(100): 
            # 배치할 랜덤 위치 (좌상단 기준)
            rand_x = random.randint(0, factory_width - machine.footprint[0])
            rand_y = random.randint(0, factory_height - machine.footprint[1])

            if can_place_machine(current_grid, machine.footprint, rand_x, rand_y):
                place_machine_on_grid(current_grid, machine.id, machine.footprint, rand_x, rand_y)
                machine.position = (rand_x, rand_y)
                current_machine_positions[machine.id] = {
                    "x": rand_x, 
                    "y": rand_y,
                    "center_x": rand_x + machine.footprint[0] / 2.0,
                    "center_y": rand_y + machine.footprint[1] / 2.0
                }
                placed = True
                break # 현재 설비 배치 성공, 다음 설비로
        
        if not placed:
            # print(f"경고: {machine.name}(ID:{machine.id}) 설비를 배치할 공간을 찾지 못했습니다.")
            return None, None # 하나의 설비라도 배치 못하면 실패한 레이아웃

    return current_grid, current_machine_positions


def find_best_layout_simple(num_attempts=100):
    """여러 번의 무작위 배치를 시도하여 가장 좋은 레이아웃을 찾습니다."""
    
    best_layout_grid = None
    best_machine_positions = None
    min_distance = float('inf')

    print(f"{num_attempts}번의 무작위 배치를 시도합니다...")

    for i in range(num_attempts):
        print(f"\n시도 {i+1}/{num_attempts}")
        grid, positions = generate_random_valid_layout(machines_definitions, FACTORY_WIDTH, FACTORY_HEIGHT)

        if grid and positions: # 유효한 레이아웃이 생성된 경우
            # print_layout(grid, positions) # 모든 시도 출력 원하면 주석 해제
            current_distance = calculate_total_distance(positions, PROCESS_SEQUENCE)
            print(f"  생성된 레이아웃의 총 이동 거리: {current_distance:.2f}")

            if current_distance < min_distance:
                min_distance = current_distance
                best_layout_grid = copy.deepcopy(grid) # 가장 좋은 레이아웃 저장 (깊은 복사)
                best_machine_positions = copy.deepcopy(positions)
                print(f"  *** 새로운 최적 레이아웃 발견! (거리: {min_distance:.2f}) ***")
        else:
            print("  유효한 레이아웃 생성 실패.")


    if best_layout_grid:
        print("\n--- 최종 최적 레이아웃 ---")
        print_layout(best_layout_grid, best_machine_positions)
        print(f"최소 총 이동 거리: {min_distance:.2f}")
    else:
        print("유효한 레이아웃을 찾지 못했습니다. 공장 크기나 설비 크기를 확인해보세요.")
    
    return best_layout_grid, best_machine_positions, min_distance

# --- 단계 5: 메인 실행 블록 ---
if __name__ == '__main__':
    print("설비 배치 최적화 프로그램 (간단한 버전) 시작")

    # 설정된 값으로 최적 배치 탐색 실행
    # 시도 횟수를 늘리면 더 좋은 결과를 찾을 확률이 높아지지만 시간이 오래 걸립니다.
    final_grid, final_positions, final_distance = find_best_layout_simple(num_attempts=200) # 시도 횟수 조절 가능

    # 여기에 PyTorch를 사용한 고도화 아이디어를 추가할 수 있습니다.
    # 예를 들어, generate_random_valid_layout 대신, PyTorch로 학습된 정책 신경망이
    # 설비 배치 위치를 순차적으로 결정하도록 만들 수 있습니다. (이전 답변의 PolicyNetwork 참고)
    # 그 학습 과정에는 여기서 사용된 calculate_total_distance 와 같은 평가 함수가 보상(reward) 계산에 활용됩니다.

    print("\n프로그램 종료.")