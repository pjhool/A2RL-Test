import pandas as pd
import os

def load_tags_to_dataframe(file_path):
    """
    텍스트 파일에서 ID와 카테고리를 읽어 2개의 컬럼을 가진 DataFrame으로 변환합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} 파일이 존재하지 않습니다.")
        return None

    # 1. 파일을 한 줄씩 읽어서 하나의 컬럼으로 로드
    # sep='\t'을 사용하여 줄 전체를 읽어옵니다. (pandas에서 \n은 구분자로 쓸 수 없습니다)
    df_raw = pd.read_csv(file_path, header=None, names=['raw_line'], sep='\t')

    # 2. 첫 번째 공백(n=1)을 기준으로 ID와 Category 분리
    # 'Black and White'와 같이 중간에 공백이 있는 카테고리명을 안전하게 처리합니다.
    df = df_raw['raw_line'].str.split(n=1, expand=True)
    df.columns = ['ID', 'Category']
    
    return df

if __name__ == "__main__":
    # 테스트를 위한 샘플 파일 경로
    sample_file = 'tags_sample.txt'
    
    # 샘플 파일 생성 (실제 파일이 있다면 이 부분은 생략 가능)
    data = """43 Astrophotography
57 Birds
21 Black and White
34 Digital Art
37 Diptych / Triptych
49 DPChallenge GTGs"""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(data)
    
    # 데이터 처리 실행
    result_df = load_tags_to_dataframe(sample_file)
    
    if result_df is not None:
        print("=== 2 Column DataFrame Result ===")
        print(result_df)
        print("\n=== Sample Check ('Black and White') ===")
        # ID가 21인 행 확인
        print(result_df[result_df['ID'] == '21'])
