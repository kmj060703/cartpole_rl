## 주요 기능
**Parallel Environments**: 9개의 환경을 동시에 실행하여 데이터 수집 속도 극대화
**DQN (Deep Q-Network)**: Experience Replay와 Target Network가 적용된 DQN 알고리즘
**Real-time Visualization**: OpenCV를 활용하여 9개의 학습 화면을 하나의 윈도우에서 실시간 모니터링

## 설치 및 실행 방법 (Setup & Usage)

### 1. 프로젝트 클론 및 이동
```bash
git clone https://github.com/kmj060703/cartpole_rl.git
cd cartpole_rl
```

### 2. 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 4. 학습 시작
```bash
python train.py
```

## 추후 수정 방안
**Checkpoint 저장/로드**: 학습 중간 가중치(`model.pth`)를 저장하고 불러오는 기능 추가
**Hyperparameter Tuning**: `config.py`의 파라미터(LR, Gamma 등) 최적화
