import pandas as pd
from pathlib import Path

class AVALandscapeImageSearcher:
    def __init__(self, ava_txt_path, landscape_list_path):
        """
        AVA Dataset Landscape 이미지 검색 초기화
        
        Args:
            ava_txt_path: AVA.txt 파일 경로 (이미지 점수 정보)
            landscape_list_path: landscape_test.jpgl 파일 경로 (landscape 이미지 ID 목록)
        """
        self.ava_txt_path = ava_txt_path
        self.landscape_list_path = landscape_list_path
        self.ava_data = None
        self.landscape_image_ids = None
    
    def load_landscape_image_ids(self):
        """
        landscape_test.jpgl 파일에서 이미지 ID 로드
        형식: 각 줄에 하나의 image_id
        """
        print("Landscape 이미지 ID 로드 중...")
        image_ids = []
        
        with open(self.landscape_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        image_id = int(line)
                        image_ids.append(image_id)
                    except ValueError:
                        continue
        
        self.landscape_image_ids = image_ids
        print(f"로드된 Landscape 이미지: {len(image_ids)}개")
        return image_ids
    
    def load_ava_scores(self):
        """
        AVA.txt 파일에서 점수 정보 로드
        형식: image_id score_1 score_2 ... score_10
        """
        print("\nAVA 점수 정보 로드 중...")
        self.ava_data = pd.read_csv(
            self.ava_txt_path,
            header=None,
            sep=' ',
            names=['image_id'] + [f'score_{i}' for i in range(1, 11)]
        )
        print(f"로드된 전체 이미지: {len(self.ava_data)}개")
        return self.ava_data
    
    def calculate_mean_score(self, df):
        """
        점수의 평균값 계산
        """
        score_cols = [col for col in df.columns if col.startswith('score_')]
        return df[score_cols].mean(axis=1)
    
    def search_landscape_images(self):
        """
        Landscape 이미지 검색 및 점수 계산
        
        Returns:
            landscape 이미지 데이터프레임 (image_id, 점수들, mean_score 포함)
        """
        print(f"\n{'='*60}")
        print("Landscape 이미지 검색 시작")
        print(f"{'='*60}")
        
        # 데이터 로드
        self.load_landscape_image_ids()
        self.load_ava_scores()
        
        # Landscape 이미지만 필터링
        print(f"\n[검색 중] Landscape 이미지 필터링...")
        landscape_ids_set = set(self.landscape_image_ids)
        result_df = self.ava_data[self.ava_data['image_id'].isin(landscape_ids_set)].copy()
        
        # 평균 점수 계산
        result_df['mean_score'] = self.calculate_mean_score(result_df)
        
        # 평균 점수 기준으로 정렬
        result_df = result_df.sort_values('mean_score', ascending=False)
        
        print(f"검색 완료: {len(result_df)}개 이미지 발견")
        
        return result_df
    
    def display_results(self, result_df, top_n=30):
        """
        검색 결과 분석 및 표시
        """
        if result_df is None or len(result_df) == 0:
            print("검색 결과가 없습니다.")
            return
        
        print(f"\n{'='*60}")
        print("Landscape 이미지 검색 결과")
        print(f"{'='*60}")
        
        print(f"\n[통계 정보]")
        print(f"총 이미지 수: {len(result_df)}개")
        print(f"\n점수 분포:")
        print(result_df['mean_score'].describe())
        
        # 점수 범위별 분석
        print(f"\n[점수 범위별 이미지 분포]")
        score_ranges = [
            (9.0, 10.0, "9.0 ~ 10.0 (매우 높음)"),
            (8.0, 9.0, "8.0 ~ 9.0 (높음)"),
            (7.0, 8.0, "7.0 ~ 8.0 (중상)"),
            (6.0, 7.0, "6.0 ~ 7.0 (중간)"),
            (5.0, 6.0, "5.0 ~ 6.0 (중하)"),
            (0.0, 5.0, "0.0 ~ 5.0 (낮음)")
        ]
        
        for low, high, label in score_ranges:
            count = len(result_df[(result_df['mean_score'] >= low) & 
                                 (result_df['mean_score'] < high)])
            percentage = (count / len(result_df)) * 100
            print(f"  {label}: {count}개 ({percentage:.1f}%)")
        
        # 상위 이미지 표시
        print(f"\n[상위 {top_n}개 이미지 (점수 높은 순)]")
        print(f"{'Image ID':<15} {'Mean Score':<15} {'Score Details':<50}")
        print("-" * 80)
        
        top_images = result_df.head(top_n)
        for idx, row in top_images.iterrows():
            image_id = row['image_id']
            mean_score = row['mean_score']
            scores = [str(row[f'score_{i}']) for i in range(1, 11)]
            score_str = ", ".join(scores)
            print(f"{image_id:<15} {mean_score:<15.2f} {score_str:<50}")
        
        # 하위 이미지 표시
        print(f"\n[하위 {top_n}개 이미지 (점수 낮은 순)]")
        print(f"{'Image ID':<15} {'Mean Score':<15} {'Score Details':<50}")
        print("-" * 80)
        
        bottom_images = result_df.tail(top_n)
        for idx, row in bottom_images.iterrows():
            image_id = row['image_id']
            mean_score = row['mean_score']
            scores = [str(row[f'score_{i}']) for i in range(1, 11)]
            score_str = ", ".join(scores)
            print(f"{image_id:<15} {mean_score:<15.2f} {score_str:<50}")
    
    def filter_by_score(self, result_df, min_score=5.0, max_score=10.0):
        """
        점수 범위로 필터링
        
        Args:
            result_df: 검색 결과 데이터프레임
            min_score: 최소 점수
            max_score: 최대 점수
            
        Returns:
            필터링된 데이터프레임
        """
        filtered_df = result_df[
            (result_df['mean_score'] >= min_score) & 
            (result_df['mean_score'] <= max_score)
        ].copy()
        
        print(f"\n[필터링 결과]")
        print(f"점수 범위: {min_score} ~ {max_score}")
        print(f"필터링된 이미지: {len(filtered_df)}개")
        
        return filtered_df
    
    def save_results(self, result_df, output_file='landscape_images.csv'):
        """
        검색 결과를 CSV 파일로 저장
        """
        if result_df is not None and len(result_df) > 0:
            output_df = result_df[['image_id', 'mean_score'] + 
                                   [f'score_{i}' for i in range(1, 11)]].copy()
            output_df.to_csv(output_file, index=False)
            print(f"\n결과가 저장되었습니다: {output_file}")
            return output_df
        return None
    
    def save_image_ids(self, result_df, output_file='landscape_image_ids.txt'):
        """
        이미지 ID만 텍스트 파일로 저장
        """
        if result_df is not None and len(result_df) > 0:
            with open(output_file, 'w') as f:
                for image_id in result_df['image_id']:
                    f.write(f"{image_id}\n")
            print(f"이미지 ID가 저장되었습니다: {output_file}")


# 사용 예제
if __name__ == "__main__":
    # 파일 경로 설정
    ava_txt_path = "AVA.txt"  # AVA.txt 파일 경로
    landscape_list_path = "landscape_test.jpgl"  # landscape_test.jpgl 파일 경로
    
    # 검색기 초기화
    searcher = AVALandscapeImageSearcher(
        ava_txt_path=ava_txt_path,
        landscape_list_path=landscape_list_path
    )
    
    # Landscape 이미지 검색
    result_df = searcher.search_landscape_images()
    
    # 결과 표시 (상위/하위 30개)
    searcher.display_results(result_df, top_n=30)
    
    # 점수별로 필터링 (선택사항)
    print("\n" + "="*60)
    print("추가 필터링 예제")
    print("="*60)
    
    # 높은 점수 이미지만 (7점 이상)
    high_score_df = searcher.filter_by_score(result_df, min_score=7.0, max_score=10.0)
    print(f"높은 점수 이미지: {len(high_score_df)}개")
    
    # 낮은 점수 이미지만 (5점 이하)
    low_score_df = searcher.filter_by_score(result_df, min_score=0.0, max_score=5.0)
    print(f"낮은 점수 이미지: {len(low_score_df)}개")
    
    # 결과 저장
    searcher.save_results(result_df, 'landscape_results.csv')
    searcher.save_image_ids(high_score_df, 'landscape_high_score_ids.txt')
    searcher.save_image_ids(low_score_df, 'landscape_low_score_ids.txt')
    
    print("\n검색 및 저장 완료!")
