# A2RL_a3c_colab.py 실행 매뉴얼

`A2RL_a3c_colab.py`는 A3C(Asynchronous Advantage Actor-Critic) 알고리즘을 사용하여 이미지 미적 점수(Aesthetic Score)를 최적화하는 에이전트를 학습시키기 위한 스크립트입니다. Google Colab 및 Kaggle 환경에 최적화되어 있습니다.

## 1. 커맨드라인 인자 (Command-Line Arguments)

스크립트 실행 시 다음과 같은 인자를 사용할 수 있습니다.

| 인자 | 타입 | 설명 |
| :--- | :--- | :--- |
| `--resume` | `str` | 기존 학습 상태(메타데이터, Fold, Epoch 포함)를 복구하여 학습을 재개할 모델의 경로입니다. |
| `--load_weights` | `str` | 초기 가중치로 사용할 모델의 경로입니다. 메타데이터는 무시됩니다. (기본값: `config.LOAD_WEIGHTS`) |
| `--download_weights` | `store_true` | Google Drive의 `save_model` 폴더에서 최신 가중치를 자동으로 다운로드합니다. |
| `--evaluate` | `str` | 학습 대신 평가 모드로 실행합니다. 평가할 모델 파일의 경로(확장자 제외)를 입력합니다. |
| `--preprocess` | `store_true` | 학습 전 데이터셋 전처리를 수행합니다. 미적 점수가 낮은 이미지를 필터링합니다. |
| `--workers` | `int` | 전처리 시 사용할 워커 프로세스의 수입니다. |

## 2. 주요 환경 변수 (Key Environment Variables)

`config_colab.py`를 통해 제어되는 주요 하이퍼파라미터 및 설정입니다.

### 경로 관련
*   `A2RL_DATA_ROOT`: 학습 데이터셋의 루트 경로 (기본값: Colab(`/content/data`), Kaggle(`/kaggle/input/a2rl-ava`))
*   `A2RL_EVAL_DATA_ROOT`: 평가용 데이터셋의 루트 경로

### 학습 하이퍼파라미터
*   `A2RL_THREADS`: 학습 시 사용할 스레드 수 (기본값: `4`)
*   `A2RL_ACTOR_LR`: Actor 학습률 (기본값: `1.0e-5`)
*   `A2RL_CRITIC_LR`: Critic 학습률 (기본값: `1.0e-5`)
*   `A2RL_BETA`: Entropy 정규화 계수 (기본값: `0.01`)
*   `A2RL_EPOCH_SIZE`: 한 Epoch 당 에피소드 수 (기본값: `200`)
*   `A2RL_T_MAX`: 한 에피소드의 최대 스텝 수 (기본값: `50`)

### 네트워크 및 로직 설정
*   `A2RL_USE_LSTM`: LSTM 사용 여부 (0: MLP, 1: LSTM, 기본값: `0`)
*   `A2RL_USE_LAYER_NORM`: Layer Normalization 사용 여부 (기본값: `1`)
*   `A2RL_ENABLE_MINI_BATCH`: 미니배치 학습 사용 여부 (기본값: `1`)
*   `A2RL_USE_K_FOLD`: K-Fold 교차 검증 사용 여부 (기본값: `0`)

## 3. 실행 예제 (Execution Examples)

### A. 일반 학습 시작
기본 설정을 사용하여 학습을 시작합니다.
```bash
python colab/A2RL_a3c_colab.py
```

### B. 특정 가중치 로드 후 학습 시작
사전 학습된 가중치를 불러와 초기값으로 사용합니다.
```bash
python colab/A2RL_a3c_colab.py --load_weights ./models/pretrained_weights
```

### C. 학습 재개 (Resumption)
비정상 종료 시 시스템 메타데이터를 포함하여 이전 지점부터 다시 시작합니다.
```bash
python colab/A2RL_a3c_colab.py --resume ./models/backup_checkpoint
```

### D. 데이터 전처리만 수행
학습 전 이미지 필터링만 수행하고 종료합니다.
```bash
python colab/A2RL_a3c_colab.py --preprocess --workers 8
```

### E. 모델 평가
학습된 모델을 사용하여 성능을 평가합니다.
```bash
python colab/A2RL_a3c_colab.py --evaluate ./models/final_model
```

### F. 하이퍼파라미터 조정 실행 (환경 변수 활용)
학습률과 스레드 수를 명시적으로 지정하여 실행합니다.
```bash
export A2RL_ACTOR_LR=5e-6
export A2RL_THREADS=8
python colab/A2RL_a3c_colab.py
```

## 4. 참고 사항
*   **Kaggle/Colab 감지**: `config_colab.py`는 환경 변수를 기반으로 자동으로 경로를 최적화합니다.
*   **Google Drive 백업**: `A2RL_GDRIVE_BACKUP_ENABLED=1` 설정 시 주기적으로 학습 결과가 Google Drive로 업로드됩니다.
