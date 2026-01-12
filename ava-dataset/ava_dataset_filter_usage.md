# AVA Dataset Filter Usage Manual

`ava_dataset_filter.py`는 AVA(Aesthetic Visual Analysis) 데이터셋을 특정 카테고리와 점수 기준에 따라 필터링하고, 학습(Train) 및 검증(Validation) 세트로 층화 추출(Stratified Sampling)하는 기능을 제공하는 스크립트입니다.

## 1. 개요

이 도구는 다음과 같은 작업을 수행합니다:
- **AVA 데이터 로드**: `AVA.txt` 및 태그 매핑 정보를 읽어들입니다.
- **점수 계산**: 10개의 점수 빈도수를 기반으로 가중 평균 점수를 산출합니다.
- **카테고리 필터링**: 사용자가 지정한 특정 카테고리(Landscape, Nature, Architecture 등)가 포함된 이미지를 선별합니다.
- **층화 추출 및 분할**: 점수대별(Low, Mid, High)로 이미지를 샘플링하고, 지정된 비율로 Train/Val 세트를 구성합니다.

## 2. 필수 파일 및 준비 사항

스크립트 실행을 위해 다음 파일들이 필요합니다:
- **AVA.txt**: AVA 데이터셋의 이미지 ID와 점수 정보가 포함된 텍스트 파일.
- **tags.txt**: Tag ID와 카테고리 이름 매핑 정보가 포함된 파일 (탭 구분자 사용).
- **이미지 디렉토리**: AVA 데이터셋의 원본 이미지들이 저장된 폴더.

## 3. 주요 기능 및 설정

### 대상 카테고리 (Target Categories)
기본적으로 다음 카테고리들이 필터링 대상으로 설정되어 있습니다:
`Landscape`, `Nature`, `Sky`, `Travel`, `Architecture`, `Rural`, `Transportation`, `Performance`

### 점수 범위 (Score Ranges)
층화 추출 시 다음 3가지 범위로 구분됩니다:
1. **low_score**: 평균 점수 ≤ 4.0
2. **mid_score**: 4.0 < 평균 점수 < 7.0
3. **high_score**: 평균 점수 ≥ 7.0

## 4. 사용 방법

### 초기화
`AVADatasetFilter` 클래스를 인스턴스화할 때 주요 경로를 설정합니다.

```python
filter_obj = AVADatasetFilter(
    ava_txt_path="path/to/AVA.txt",
    semantics_path="path/to/tags.txt",
    original_images_dir="path/to/images",
    output_dir="path/to/output",
    docs_dir="path/to/docs"
)
```

### 실행 (8:2 분할 샘플링)
`run_stratified_sampling` 메서드를 통해 샘플링을 수행합니다.

```python
# 각 점수 범위별로 최대 3,000장씩 확보하며 8:2로 분할
filter_obj.run_stratified_sampling(sample_size=3000, train_ratio=0.8)
```

## 5. 결과물 구조

실행이 완료되면 `output_dir`에 다음과 같은 디렉토리 구조가 생성됩니다:

```text
output_dir/
├── stratified_samples.csv        # 샘플링된 전체 이미지 메타데이터
├── train/                        # 학습용 데이터 (80%)
│   ├── low_score/
│   ├── mid_score/
│   └── high_score/
└── val/                          # 검증용 데이터 (20%)
    ├── low_score/
    ├── mid_score/
    └── high_score/
```

- **stratified_samples.csv**: `image_id`, `file_name`, `mean_score`, `category`, `score_range`, `split` 정보를 포함합니다.
- **docs_dir**: `stratified_samples_info.csv`가 추가로 저장됩니다.

## 6. 주의 사항
- 스크립트는 원본 이미지의 확장자(`jpg`, `jpeg`, `png` 등)를 자동으로 감지합니다.
- 파일이 실제로 존재하지 않는 경우 샘플링에서 제외됩니다.
- `shutil.copy2`를 사용하여 이미지의 메타데이터를 유지하며 복사합니다.
