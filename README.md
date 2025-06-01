# -Reinforcement-Learning-Facility-Layout-Optimization
# 강화학습 기반 생산 설비 배치 최적화 (Reinforcement Learning Facility Layout Optimization)

## 🇰🇷 소개 (Introduction - Korean)

이 프로젝트는 강화학습(Deep Reinforcement Learning)을 활용하여 공장 내 설비 배치를 최적화하는 방법을 탐구하는 과정을 담고 있습니다. 목표는 단순히 이동 거리를 최소화하는 것을 넘어, 실제 생산 목표(예: 시간당 생산량)와 설비별 제약 조건까지 고려하여 최적의 설비 배치를 찾는 것입니다. 이 레포지토리는 아이디어 구상부터 점진적으로 기능을 개선해 나가는 학습 여정을 기록합니다.

이 프로젝트는 학습 및 포트폴리오 목적으로 진행되었으며, 실제 산업 현장에 바로 적용하기에는 단순화된 모델과 가정이 포함되어 있습니다.

## 🇬🇧 Introduction (소개 - English)

This project exploring how to optimize facility layouts within a factory using Deep Reinforcement Learning (DRL). The goal extends beyond simply minimizing travel distance to finding optimal equipment placements by considering actual production targets (e.g., throughput per hour) and machine-specific constraints. This repository records the learning process, from initial ideas to incremental feature improvements.

This project was developed for learning and portfolio purposes and includes simplified models and assumptions that may not be directly applicable to real-world industrial scenarios without further refinement.

---

## 🇰🇷 프로젝트 진행 단계 (Project Evolution - Korean)

이 프로젝트는 다음과 같은 단계별로 개발되었습니다. 각 Python 파일은 해당 단계의 구현 내용을 담고 있습니다.

1.  **`01_random_search_layout.py`**:
    * 기본적인 설비 배치 문제 정의.
    * 무작위 탐색(Random Search)을 통해 설비 간 총 이동 거리를 최소화하는 레이아웃 탐색.
    * PyTorch나 딥러닝 없이 순수 Python으로 구현.

2.  **`02_pytorch_policy_gradient.py`**:
    * PyTorch를 처음 도입하여 간단한 정책 경사(Policy Gradient, REINFORCE 유사) 알고리즘 구현.
    * 신경망이 설비 배치 위치를 순차적으로 결정하도록 학습.
    * GPU 사용 지원 추가.

3.  **`03_a2c_basic_layout.py`**:
    * 강화학습 알고리즘을 A2C(Advantage Actor-Critic)로 개선.
    * 액터(정책망)와 크리틱(가치망)을 사용하여 학습 안정성 및 효율 증대 시도.
    * 주요 최적화 목표는 여전히 설비 간 총 이동 거리 최소화.

4.  **`04_a2c_throughput_optimization.py`**:
    * A2C 알고리즘의 보상 함수에 실제 생산성 지표인 '시간당 목표 생산량(Throughput)' 반영.
    * 각 설비의 사이클 타임과 설비 간 자재 이동 시간을 고려하여 라인 생산량 추정.
    * 이동 거리 최소화와 생산량 목표 달성 간의 균형을 맞추도록 학습.

5.  **`05_a2c_constraints_optimization.py`**:
    * 더 현실적인 설비 배치를 위해 설비별 제약 조건 추가.
    * **최소 이격 거리 (Clearance)**: 설비 주변에 필요한 최소 빈 공간을 하드 제약으로 적용.
    * **벽 근접 선호 (Wall Affinity)**: 특정 설비가 벽에 가깝게 배치되도록 유도 (소프트 제약 - 보상 페널티).

6.  **`06_ppo_constraints_optimization.py`**:
    * 강화학습 알고리즘을 PPO(Proximal Policy Optimization)로 한 단계 더 발전.
    * PPO의 클리핑 메커니즘을 통해 학습 안정성 및 샘플 효율성 향상 기대.
    * 생산량 목표 및 설비 제약 조건은 계속 유지하며 최적화 수행.

## 🇬🇧 Project Evolution (프로젝트 진행 단계 - English)

This project was developed iteratively through the following stages. Each Python file contains the implementation for that stage.

1.  **`01_random_search_layout.py`**:
    * Defines the basic facility layout problem.
    * Explores layouts to minimize total material travel distance using Random Search.
    * Implemented in pure Python without PyTorch or deep learning.

2.  **`02_pytorch_policy_gradient.py`**:
    * Introduces PyTorch by implementing a simple Policy Gradient (REINFORCE-like) algorithm.
    * Trains a neural network to sequentially decide machine placement positions.
    * Adds support for GPU usage.

3.  **`03_a2c_basic_layout.py`**:
    * Upgrades the reinforcement learning algorithm to A2C (Advantage Actor-Critic).
    * Attempts to improve learning stability and efficiency using an actor (policy network) and a critic (value network).
    * The primary optimization objective remains minimizing total travel distance between machines.

4.  **`04_a2c_throughput_optimization.py`**:
    * Incorporates a key productivity metric, 'target throughput per hour,' into the A2C algorithm's reward function.
    * Estimates line throughput by considering machine cycle times and material travel times between machines.
    * Trains the agent to balance minimizing travel distance with achieving production targets.

5.  **`05_a2c_constraints_optimization.py`**:
    * Adds machine-specific constraints for more realistic layouts.
    * **Clearance**: Enforces minimum empty space around machines as a hard constraint.
    * **Wall Affinity**: Encourages certain machines to be placed near walls as a soft constraint (via reward penalties).

6.  **`06_ppo_constraints_optimization.py`**:
    * Further advances the reinforcement learning algorithm to PPO (Proximal Policy Optimization).
    * Aims to improve learning stability and sample efficiency using PPO's clipping mechanism.
    * Continues to optimize for throughput targets and machine constraints.

---

## 🇰🇷 실행 방법 (How to Run - Korean)
https://github.com/imjeasung/-Reinforcement-Learning-Facility-Layout-Optimization.git
1.  이 레포지토리를 로컬 컴퓨터에 클론합니다:
    ```bash
    git clone https://github.com/imjeasung/-Reinforcement-Learning-Facility-Layout-Optimization.git
    cd YOUR_REPOSITORY_NAME
    ```
2.  필요한 라이브러리를 설치합니다. (Python 3.8+ 권장)
    ```bash
    pip install torch numpy matplotlib
    ```
    (PyTorch 설치 시, 사용자의 CUDA 버전에 맞는 명령어를 [PyTorch 공식 웹사이트](https://pytorch.org/)에서 확인하는 것이 좋습니다.)
3.  각 단계별 Python 파일을 실행하여 결과를 확인할 수 있습니다. 예를 들어, 가장 최신 버전인 PPO 알고리즘을 실행하려면:
    ```bash
    python 06_ppo_constraints_optimization.py
    ```
4.  스크립트 내의 하이퍼파라미터(예: `num_episodes`, `learning_rate` 등)를 조정하여 학습 과정을 실험해볼 수 있습니다.

## 🇬🇧 How to Run (실행 방법 - English)

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
2.  Install the required libraries. (Python 3.7+ recommended)
    ```bash
    pip install torch numpy matplotlib
    ```
    (When installing PyTorch, it's advisable to check the [official PyTorch website](https://pytorch.org/) for commands specific to your CUDA version if you plan to use a GPU.)
3.  You can run each Python file to observe the results of that stage. For example, to run the latest PPO version:
    ```bash
    python 06_ppo_constraints_optimization.py
    ```
4.  You can adjust hyperparameters within the scripts (e.g., `num_episodes`, `learning_rate`) to experiment with the learning process.

---

## 🇰🇷 향후 개선 방향 (Potential Future Work - Korean)

* 더 복잡하고 다양한 설비 제약 조건 추가 (예: 특정 설비 간 인접/이격 조건).
* 생산 라인의 변동성(설비 고장, 작업 시간 변동 등)을 고려한 시뮬레이션 및 최적화.
* 사용자 인터페이스(UI) 개발을 통한 사용 편의성 증대.
* 다른 최신 강화학습 알고리즘 적용 및 비교.
* 하이퍼파라미터 최적화 자동화.

## 🇬🇧 Potential Future Work (향후 개선 방향 - English)

* Adding more complex and diverse machine constraints (e.g., proximity/exclusion zones between specific machines).
* Simulation and optimization considering production line variability (e.g., machine breakdowns, fluctuating task times).
* Developing a user interface (UI) for easier interaction.
* Applying and comparing other state-of-the-art reinforcement learning algorithms.
* Automating hyperparameter optimization.

---
