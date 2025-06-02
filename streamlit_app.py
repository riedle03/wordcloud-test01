import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import numpy as np
import os
import re
from collections import Counter
import math

# 페이지 설정
st.set_page_config(
    page_title="한국어 키워드 추출 & 워드클라우드 생성기",
    page_icon="🔍",
    layout="wide"
)

# 한국어 불용어 리스트 (교육 분야에 맞게 조정)
KOREAN_STOPWORDS = {
    '이', '그', '저', '것', '들', '는', '은', '을', '를', '에', '의', '가', '와', '과', '도', '로', '으로',
    '이다', '있다', '없다', '하다', '되다', '같다', '다른', '많은', '작은', '큰', '좋은', '나쁜',
    '또한', '그리고', '하지만', '그러나', '따라서', '즉', '예를', '들어', '바로', '단지', '다만',
    '때문', '위해', '통해', '대해', '관해', '에서', '에게', '부터', '까지', '라고', '라는', '이라는',
    '이런', '그런', '저런', '이것', '그것', '저것', '여기', '거기', '저기', '지금', '오늘',
    '내일', '어제', '언제', '어디', '누구', '무엇', '왜', '어떻게', '어떤', '모든', '각각',
    '수', '때', '곳', '사람', '것들', '점', '면', '등', '중', '간', '후', '전', '내', '외',
    '상', '하', '좌', '우', '앞', '뒤', '위', '아래', '사이', '속', '밖', '안', '여러', '각종',
    '하나', '둘', '셋', '있는', '없는', '되는', '하는', '큰', '작은', '새로운', '오래된',
    '그런데', '그래서', '또', '또한', '역시', '물론', '당연히', '확실히', '아마', '정말',
    '너무', '매우', '상당히', '꽤', '조금', '약간', '살짝', '좀', '잠깐', '한번', '두번',
    '처음', '마지막', '다음', '이전', '계속', '항상', '가끔', '자주', '때때로', '보통',
    '일반적', '특별한', '중요한', '필요한', '가능한', '어려운', '쉬운', '복잡한', '간단한'
}

def clean_text(text):
    """텍스트 전처리"""
    # 특수문자 제거 (한글, 숫자, 공백만 남김)
    text = re.sub(r'[^가-힣0-9\s]', ' ', text)
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords_simple(text, min_length=2, max_keywords=20):
    """간단한 키워드 추출 (단어 빈도 기반)"""
    # 텍스트 전처리
    cleaned_text = clean_text(text)
    
    # 단어 분리 (공백 기준)
    words = cleaned_text.split()
    
    # 필터링: 길이, 불용어, 숫자만으로 구성된 단어 제외
    filtered_words = []
    for word in words:
        if (len(word) >= min_length and 
            word not in KOREAN_STOPWORDS and 
            not word.isdigit() and
            re.search(r'[가-힣]', word)):  # 한글이 포함된 단어만
            filtered_words.append(word)
    
    # 빈도 계산
    word_counts = Counter(filtered_words)
    
    # 상위 키워드 추출
    top_keywords = word_counts.most_common(max_keywords)
    
    # 가중치 계산 (최대값을 10으로 정규화)
    if not top_keywords:
        return {}
    
    max_count = top_keywords[0][1]
    weighted_keywords = {}
    
    for word, count in top_keywords:
        # 1~10 사이의 가중치로 계산
        weight = max(1, round((count / max_count) * 10))
        weighted_keywords[word] = weight
    
    return weighted_keywords

def extract_keywords_ngram(text, min_length=2, max_keywords=20):
    """N-gram 기반 키워드 추출"""
    # 텍스트 전처리
    cleaned_text = clean_text(text)
    
    # 단어와 2-gram, 3-gram 추출
    words = cleaned_text.split()
    
    # 1-gram (단어)
    unigrams = []
    for word in words:
        if (len(word) >= min_length and 
            word not in KOREAN_STOPWORDS and 
            not word.isdigit() and
            re.search(r'[가-힣]', word)):
            unigrams.append(word)
    
    # 2-gram
    bigrams = []
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        if (len(bigram) >= 4 and 
            words[i] not in KOREAN_STOPWORDS and 
            words[i + 1] not in KOREAN_STOPWORDS and
            re.search(r'[가-힣]', bigram)):
            bigrams.append(bigram)
    
    # 빈도 계산
    all_terms = unigrams + bigrams
    term_counts = Counter(all_terms)
    
    # TF-IDF 스타일의 가중치 적용 (간단 버전)
    weighted_terms = {}
    total_terms = len(all_terms)
    
    for term, count in term_counts.items():
        # TF (용어 빈도)
        tf = count / total_terms
        # 간단한 가중치 (빈도와 길이 고려)
        length_bonus = min(2.0, len(term.split()) * 0.5 + 1)
        score = tf * length_bonus * 1000  # 스케일링
        weighted_terms[term] = score
    
    # 상위 키워드 선택
    top_terms = sorted(weighted_terms.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    
    # 1~10 사이의 가중치로 정규화
    if not top_terms:
        return {}
    
    max_score = top_terms[0][1]
    final_keywords = {}
    
    for term, score in top_terms:
        weight = max(1, round((score / max_score) * 10))
        final_keywords[term] = weight
    
    return final_keywords

def format_keywords_output(keywords_dict):
    """키워드를 요청한 형식으로 출력"""
    if not keywords_dict:
        return "키워드를 찾을 수 없습니다."
    
    # 가중치 순으로 정렬
    sorted_keywords = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)
    
    # "키워드A 5, 키워드B 4, 키워드C 3" 형식으로 변환
    formatted_parts = []
    for keyword, weight in sorted_keywords:
        formatted_parts.append(f"{keyword} {weight}")
    
    return ", ".join(formatted_parts)

def create_wordcloud_from_keywords(keywords_dict, width=800, height=600):
    """키워드 딕셔너리로부터 워드클라우드 생성"""
    if not keywords_dict:
        return None
    
    # 폰트 경로 확인
    font_path = "./fonts/NanumGothic-Regular.ttf"
    if not os.path.exists(font_path):
        st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        st.info("폰트 파일이 ./fonts/NanumGothic-Regular.ttf 경로에 있는지 확인해주세요.")
        return None
    
    try:
        # 워드클라우드 생성
        wc = WordCloud(
            font_path=font_path,
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            colormap='viridis'
        ).generate_from_frequencies(keywords_dict)
        
        return wc
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def wordcloud_to_image(wc, width=800, height=600):
    """워드클라우드를 PIL Image로 변환"""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    image = Image.open(buf)
    return image

def get_image_download_link(img, filename):
    """이미지 다운로드 링크 생성"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 워드클라우드 다운로드</a>'
    return href

# 앱 제목 및 설명
st.title("🔍 한국어 키워드 추출 & 워드클라우드 생성기")
st.markdown("한국어 문단을 입력하면 키워드를 추출하고 가중치와 함께 워드클라우드를 생성해드립니다!")

# 사이드바 설정
st.sidebar.header("⚙️ 추출 설정")

extraction_method = st.sidebar.selectbox(
    "추출 방법",
    ["단순 빈도 기반", "N-gram 기반 (추천)"],
    index=1
)

min_word_length = st.sidebar.slider("최소 단어 길이", 1, 5, 2)
max_keywords = st.sidebar.slider("최대 키워드 수", 10, 50, 20)

# 워드클라우드 설정
st.sidebar.header("🎨 워드클라우드 설정")
wc_width = st.sidebar.slider("너비", 400, 1200, 800)
wc_height = st.sidebar.slider("높이", 300, 800, 600)

# 메인 인터페이스
st.header("📝 문단 입력")

# 사용법 안내
with st.expander("💡 사용법 및 예시"):
    st.markdown("""
    **사용법:**
    1. 아래 텍스트 상자에 분석하고 싶은 한국어 문단을 입력하세요
    2. '키워드 추출' 버튼을 클릭하세요
    3. 추출된 키워드와 가중치를 확인하세요
    4. 워드클라우드도 자동으로 생성됩니다
    
    **예시 텍스트:**
    ```
    교육은 미래를 준비하는 가장 중요한 과정입니다. 학생들은 창의적 사고와 
    문제 해결 능력을 기르기 위해 다양한 학습 경험이 필요합니다. 
    협력 학습을 통해 소통 능력을 향상시키고, 디지털 기술을 활용한 
    혁신적인 교육 방법으로 학습 효과를 극대화할 수 있습니다.
    ```
    
    **출력 형식:**
    키워드A 5, 키워드B 4, 키워드C 3
    """)

# 텍스트 입력 영역
text_input = st.text_area(
    "분석할 한국어 문단을 입력하세요",
    height=200,
    placeholder="예시: 교육은 미래를 준비하는 가장 중요한 과정입니다. 학생들은 창의적 사고와 문제 해결 능력을 기르기 위해...",
    help="한국어 문단을 입력하면 키워드를 자동으로 추출합니다."
)

# 키워드 추출 버튼
if st.button("🔍 키워드 추출", type="primary"):
    if text_input.strip():
        with st.spinner("키워드를 추출하는 중..."):
            # 키워드 추출
            if extraction_method == "단순 빈도 기반":
                keywords_dict = extract_keywords_simple(text_input, min_word_length, max_keywords)
            else:
                keywords_dict = extract_keywords_ngram(text_input, min_word_length, max_keywords)
            
            if keywords_dict:
                st.success(f"총 {len(keywords_dict)}개의 키워드를 추출했습니다!")
                
                # 결과 표시
                st.header("📊 추출 결과")
                
                # 요청된 형식으로 출력
                formatted_output = format_keywords_output(keywords_dict)
                st.subheader("🎯 키워드 및 가중치")
                st.code(formatted_output, language=None)
                
                # 복사 가능한 텍스트박스
                st.text_area(
                    "복사용 결과", 
                    value=formatted_output, 
                    height=100,
                    help="이 텍스트를 복사해서 워드클라우드 생성기에 사용할 수 있습니다."
                )
                
                # 상세 키워드 정보
                with st.expander("🔍 상세 키워드 정보"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**키워드**")
                        for keyword in keywords_dict.keys():
                            st.write(f"• {keyword}")
                    with col2:
                        st.write("**가중치**")
                        for weight in keywords_dict.values():
                            st.write(f"• {weight}")
                
                # 워드클라우드 생성
                st.header("☁️ 워드클라우드")
                
                with st.spinner("워드클라우드를 생성하는 중..."):
                    wordcloud = create_wordcloud_from_keywords(keywords_dict, wc_width, wc_height)
                    
                    if wordcloud:
                        # 워드클라우드를 이미지로 변환
                        img = wordcloud_to_image(wordcloud, wc_width, wc_height)
                        
                        # 이미지 표시
                        st.image(img, caption="생성된 워드클라우드", use_column_width=True)
                        
                        # 다운로드 링크
                        download_link = get_image_download_link(img, "keyword_wordcloud.png")
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # 생성 정보
                        st.info(f"크기: {wc_width}x{wc_height} | 추출 방법: {extraction_method} | 키워드 수: {len(keywords_dict)}개")
                
            else:
                st.error("키워드를 추출할 수 없습니다. 더 긴 텍스트를 입력해주세요.")
    else:
        st.warning("분석할 텍스트를 입력해주세요.")

# 푸터
st.markdown("---")
st.markdown("💡 **팁**: 더 정확한 키워드 추출을 위해 충분히 긴 문단을 입력해주세요!")
st.markdown("🎓 교육용 키워드 추출기 | Made with Streamlit")