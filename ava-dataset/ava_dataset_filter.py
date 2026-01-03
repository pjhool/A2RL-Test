import pandas as pd
import numpy as np
from pathlib import Path

class AVADatasetFilter:
    def __init__(self, ava_txt_path, semantics_path, original_images_dir, output_dir='filtered_images', docs_dir='docs'):
        """
        AVA Dataset 필터링 초기화
        
        Args:
            ava_txt_path: AVA.txt 파일 경로
            semantics_path: 태그 파일 경로
            original_images_dir: 원본 이미지 디렉토리
            output_dir: 결과 이미지 저장 경로
            docs_dir: 필터링 결과 리스트(CSV) 저장 경로
        """
        self.ava_txt_path = ava_txt_path
        self.semantics_path = semantics_path
        self.original_images_dir = Path(original_images_dir)
        self.output_dir = Path(output_dir)
        self.docs_dir = Path(docs_dir)
        self.target_categories = ['Landscape', 'Nature', 'Sky' , 'Travel', 'Architecture',  'Rural' , 'Transportation' , 'Performance']
        
        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_output_path = self.output_dir / 'filtered_images.csv'
        
    def load_ava_scores(self):
        """
        AVA.txt 파일에서 정보 로드
        표준 형식: Index ImageID Score1...Score10 Tag1 Tag2 OriginalID
        """
        print("AVA 정보 로드 중...")
        # 필요한 컬럼만 정의 (Index, ImageID, Scores 1-10, Tag1, Tag2)
        all_names = ['index', 'image_id'] + [f'score_{i}' for i in range(1, 11)] + ['tag_id_1', 'tag_id_2', 'orig_id']
        
        self.ava_scores = pd.read_csv(
            self.ava_txt_path,
            header=None,
            sep=' ',
            names=all_names
        )
        print(f"로드된 이미지: {len(self.ava_scores)}")
        return self.ava_scores
    
    def load_semantics(self):
        """
        태그 매핑 정보 로드 (TagID TagName 형식)
        """
        print(f"태그 매핑 정보 로드 중: {self.semantics_path}")
        
        df_raw = pd.read_csv(
            self.semantics_path,
            header=None,
            names=['raw'],
            sep='\t',
            engine='python'
        )
        
        self.tag_mapping = df_raw['raw'].str.split(n=1, expand=True)
        self.tag_mapping.columns = ['tag_id', 'category_name']
        self.tag_mapping['tag_id'] = pd.to_numeric(self.tag_mapping['tag_id'], errors='coerce')
        
        print(f"로드된 태그 매핑: {len(self.tag_mapping)}")
        return self.tag_mapping
    
    def calculate_mean_score(self, scores_df):
        """
        AVA 데이터셋의 가중 평균 점수 계산 
        (각 점수(1-10)의 빈도수 * 점수값)의 합 / 전체 빈도수의 합
        """
        score_counts = scores_df[[f'score_{i}' for i in range(1, 11)]].values
        ratings = np.arange(1, 11)
        
        # 가중 평균 계산
        mean_score = (score_counts * ratings).sum(axis=1) / score_counts.sum(axis=1)
        return mean_score
    
    def filter_by_category_and_score(self, max_score=10.0):
        """
        필터링 로직:
        1. tag_id_1 또는 tag_id_2가 타겟 카테고리에 해당하는지 확인
        2. 평균 점수 기준 필터링
        """
        print(f"\n필터링 시작 (최대 점수: {max_score})...")
        
        # 점수 평균 계산
        self.ava_scores['mean_score'] = self.calculate_mean_score(self.ava_scores)
        
        # 1. Tag ID 1에 대해 매핑
        merged_1 = self.ava_scores.merge(
            self.tag_mapping,
            left_on='tag_id_1',
            right_on='tag_id',
            how='left'
        ).drop(columns=['tag_id']) # 중복되는 mapping용 ID 컬럼 제거
        
        # 2. Tag ID 2에 대해 매핑 (필요시)
        merged_2 = merged_1.merge(
            self.tag_mapping,
            left_on='tag_id_2',
            right_on='tag_id',
            how='left',
            suffixes=('', '_2')
        ).drop(columns=['tag_id']) # 중복되는 mapping용 ID 컬럼 제거
        
        # 타겟 카테고리 중 하나라도 포함되는지 확인
        target_lower = [c.lower() for c in self.target_categories]
        
        def is_target(row):
            cat1 = str(row['category_name']).lower()
            cat2 = str(row['category_name_2']).lower()
            return cat1 in target_lower or cat2 in target_lower
            
        merged_2['has_target'] = merged_2.apply(is_target, axis=1)
        
        # 필터링 적용
        filtered_df = merged_2[
            (merged_2['has_target']) & 
            (merged_2['mean_score'] <= max_score)
        ].copy()
        
        # 카테고리 정보 정리 (편의를 위해)
        filtered_df['category'] = filtered_df.apply(
            lambda x: x['category_name'] if str(x['category_name']).lower() in target_lower else x['category_name_2'],
            axis=1
        )
        
        print(f"Target 카테고리({', '.join(self.target_categories)}) 및 점수 조건 만족: {len(filtered_df)}")
        return filtered_df
    
    def analyze_filtered_data(self, filtered_df):
        """
        필터링된 데이터 분석
        """
        print("\n=== 필터링된 데이터 분석 ===")
        print(f"총 이미지 수: {len(filtered_df)}")
        print(f"\n점수 분포:")
        print(filtered_df['mean_score'].describe())
        
        # 카테고리별 분포
        category_count = filtered_df['category'].value_counts()
        
        print(f"\n카테고리별 이미지 수:")
        for cat, count in category_count.items():
            print(f"  - {cat}: {count}")
    
    def copy_images_to_output(self, filtered_df, sub_dir=None):
        """
        필터링된 이미지를 출력 디렉토리로 복사
        
        Args:
            filtered_df: 복사할 데이터프레임
            sub_dir: 출력 디렉토리 하위의 특정 폴더명 (예: 'low', 'mid', 'high')
        """
        import shutil
        
        target_dir = self.output_dir
        if sub_dir:
            target_dir = self.output_dir / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"\n이미지 복사 중 (대상: {target_dir.name})...")
        
        copied_count = 0
        failed_count = 0
        
        for idx, row in filtered_df.iterrows():
            image_id = row['image_id']
            
            # 원본 이미지 파일 찾기 (확장자 자동 감지)
            source_path = None
            for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                potential_path = self.original_images_dir / f"{int(image_id)}.{ext}"
                if potential_path.exists():
                    source_path = potential_path
                    break
            
            if source_path is None:
                failed_count += 1
                continue
            
            # 목적지 경로
            dest_path = target_dir / source_path.name
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                
                if copied_count % 100 == 0:
                    print(f"  {copied_count}개 이미지 복사 완료...")
            except Exception as e:
                print(f"  오류 - {image_id}: {str(e)}")
                failed_count += 1
        
        print(f"이미지 복사 완료 ({target_dir.name}):")
        print(f"  - 성공: {copied_count}개")
        print(f"  - 실패: {failed_count}개")
        
        return copied_count, failed_count
    
    def save_filtered_results(self, filtered_df):
        """
        필터링 결과를 CSV로 저장하고 이미지 복사
        """
        # CSV 저장
        output_df = filtered_df[['image_id', 'mean_score', 'category']].copy()
        
        output_df.to_csv(self.csv_output_path, index=False)
        print(f"\nMetadata CSV가 저장되었습니다: {self.csv_output_path}")
        
        # 이미지 복사
        copied_count, failed_count = self.copy_images_to_output(filtered_df)
        
        print(f"\n=== 저장 완료 ===")
        print(f"출력 디렉토리: {self.output_dir.absolute()}")
        print(f"  - 이미지 파일: {copied_count}개")
        print(f"  - Metadata CSV: {self.csv_output_path.name}")
        
        return output_df
    
    def run_stratified_sampling(self, sample_size=3000, train_ratio=0.8):
        """
        3개의 점수 범위별로 이미지 샘플링 및 Train/Val 분할 복사:
        1. 4점 이하
        2. 4점 초과 ~ 7점 미만
        3. 7점 이상
        
        구조:
        output_dir/train/range_name/
        output_dir/val/range_name/
        """
        # 데이터 로드
        self.load_ava_scores()
        self.load_semantics()
        
        # 카테고리 필터링된 전체 데이터 확보
        all_categorized = self.filter_by_category_and_score(max_score=10.0)
        
        ranges = [
            ('low_score', all_categorized[all_categorized['mean_score'] <= 4.0]),
            ('mid_score', all_categorized[(all_categorized['mean_score'] > 4.0) & (all_categorized['mean_score'] < 7.0)]),
            ('high_score', all_categorized[all_categorized['mean_score'] >= 7.0])
        ]
        
        print(f"\n=== 범위별 8:2 분할 샘플링 시작 (Train {train_ratio*100:.0f}%, Val {(1-train_ratio)*100:.0f}%) ===")
        
        all_sampled_results = []
        extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
        
        for range_name, range_df in ranges:
            current_count = len(range_df)
            print(f"\n[{range_name}]")
            print(f"  후보 이미지: {current_count}개")
            
            if current_count == 0:
                print(f"  경고: 해당 범위의 데이터가 없습니다.")
                continue

            # 1. 파일 존재 여부 확인하며 최대 sample_size만큼 확보
            shuffled_df = range_df.sample(frac=1, random_state=42).copy()
            confirmed_samples = []
            
            print(f"  파일 존재 여부 확인 중 (최대 {sample_size}개)...")
            for _, row in shuffled_df.iterrows():
                image_id = int(row['image_id'])
                found_ext = None
                for ext in extensions:
                    if (self.original_images_dir / f"{image_id}.{ext}").exists():
                        found_ext = ext
                        break
                
                if found_ext:
                    row['file_name'] = f"{image_id}.{found_ext}"
                    confirmed_samples.append(row)
                
                if len(confirmed_samples) >= sample_size:
                    break
            
            if not confirmed_samples:
                print(f"  경고: 실존하는 파일이 없습니다.")
                continue

            # 2. Train/Val 분할
            count = len(confirmed_samples)
            train_count = int(count * train_ratio)
            
            train_samples = confirmed_samples[:train_count]
            val_samples = confirmed_samples[train_count:]
            
            print(f"  확정된 샘플: {count}개 (Train: {len(train_samples)}, Val: {len(val_samples)})")
            
            # 3. 이미지 복사
            if train_samples:
                train_df = pd.DataFrame(train_samples)
                self.copy_images_to_output(train_df, sub_dir=f"train/{range_name}")
                train_df['split'] = 'train'
                train_df['score_range'] = range_name
                all_sampled_results.append(train_df)
                
            if val_samples:
                val_df = pd.DataFrame(val_samples)
                self.copy_images_to_output(val_df, sub_dir=f"val/{range_name}")
                val_df['split'] = 'val'
                val_df['score_range'] = range_name
                all_sampled_results.append(val_df)
        
        # 모든 결과 통합 및 CSV 저장
        if all_sampled_results:
            final_df = pd.concat(all_sampled_results)
            
            # 컬럼 순서 정리
            cols = ['image_id', 'file_name', 'mean_score', 'category', 'score_range', 'split']
            
            # 1. output_dir에 저장
            final_csv_path = self.output_dir / 'stratified_samples.csv'
            final_df[cols].to_csv(final_csv_path, index=False)
            
            # 2. docs_dir에도 저장
            docs_csv_path = self.docs_dir / 'stratified_samples_info.csv'
            final_df[cols].to_csv(docs_csv_path, index=False)
            
            print(f"\nMetadata CSV 저장됨:")
            print(f"  - {final_csv_path}")
            print(f"  - {docs_csv_path}")
            
        print("\n=== 모든 작업 완료 ===")
        return all_sampled_results


# 사용 예제
if __name__ == "__main__":
    # 파일 경로 설정
    ava_txt_path = r"Y:\Project_A2RL\AVA_dataset\AVA.txt"
    semantics_path = r"Y:\Project_A2RL\AVA_dataset\tags.txt"
    original_images_dir = r"Y:\Project_A2RL\AVA_dataset\images\images"
    
    # ava-dataset 폴더에 구성
    output_dir = r"y:\Project_A2RL\A2RL-Test\ava-dataset"
    docs_dir = r"y:\Project_A2RL\A2RL-Test\docs"
    
    # 필터러 초기화
    filter_obj = AVADatasetFilter(
        ava_txt_path=ava_txt_path,
        semantics_path=semantics_path,
        original_images_dir=original_images_dir,
        output_dir=output_dir,
        docs_dir=docs_dir
    )
    
    # 필터링 및 8:2 분할 샘플링 실행 (각 범위별 최종 3,000장 확보 시도)
    filter_obj.run_stratified_sampling(sample_size=3000, train_ratio=0.8)
    
    print(f"\n생성된 디렉토리 구조 ({output_dir}):")
    print(f"  ├── stratified_samples.csv")
    print(f"  ├── train/")
    print(f"  │   ├── low_score/")
    print(f"  │   ├── mid_score/")
    print(f"  │   └── high_score/")
    print(f"  └── val/")
    print(f"      ├── low_score/")
    print(f"      ├── mid_score/")
    print(f"      └── high_score/")
