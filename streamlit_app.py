import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import numpy as np
import os
import re

# 페이지 설정
st.set_page_config(
    page_title="키워드 워드클라우드 생성기",
    page_icon="☁️",
    layout="wide"
)

# 앱 제목 및 설명
st.title("☁️ 키워드 워드클라우드 생성기")
st.markdown("키워드를 입력하면 아름다운 워드클라우드를 생성해드립니다!")

# 사이드바에 설정 옵션
st.sidebar.header("⚙️ 설정 옵션")

# 배경색 선택
background_color = st.sidebar.selectbox(
    "배경색 선택",
    ["흰색", "검정", "투명"],
    index=0
)

# 색상 테마 선택
color_theme = st.sidebar.selectbox(
    "색상 테마",
    ["기본 (랜덤)", "파스텔", "비비드", "블루 계열", "그린 계열"],
    index=0
)

# 워드클라우드 크기 설정
width = st.sidebar.slider("너비", 400, 1200, 800)
height = st.sidebar.slider("높이", 300, 800, 600)

# 최대 단어 수
max_words = st.sidebar.slider("최대 단어 수", 50, 200, 100)

def get_color_func(theme):
    """색상 테마에 따른 색상 함수 반환"""
    if theme == "파스텔":
        def pastel_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#F0E68C', '#DDA0DD', '#F4A460']
            return np.random.choice(colors)
        return pastel_color_func
    elif theme == "비비드":
        def vivid_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ['#FF1493', '#00BFFF', '#32CD32', '#FFD700', '#FF4500', '#8A2BE2']
            return np.random.choice(colors)
        return vivid_color_func
    elif theme == "블루 계열":
        def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ['#000080', '#0000CD', '#4169E1', '#1E90FF', '#00BFFF', '#87CEEB']
            return np.random.choice(colors)
        return blue_color_func
    elif theme == "그린 계열":
        def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ['#006400', '#228B22', '#32CD32', '#7CFC00', '#90EE90', '#98FB98']
            return np.random.choice(colors)
        return green_color_func
    else:
        return None

def parse_keywords(text):
    """키워드 텍스트를 파싱하여 딕셔너리로 변환"""
    keywords = {}
    
    # 쉼표나 줄바꿈으로 분리
    items = re.split(r'[,\n]', text)
    
    for item in items:
        item = item.strip()
        if not item:
            continue
            
        # 가중치가 있는지 확인 (키워드:숫자 형태)
        if ':' in item and len(item.split(':')) == 2:
            keyword, weight = item.split(':')
            keyword = keyword.strip()
            try:
                weight = float(weight.strip())
                keywords[keyword] = weight
            except ValueError:
                # 가중치가 숫자가 아니면 키워드로만 처리
                keywords[item] = 1
        else:
            keywords[item] = 1
    
    return keywords

def create_wordcloud(keywords_dict, bg_color, color_theme, w, h, max_w):
    """워드클라우드 생성"""
    
    # 배경색 설정
    bg_map = {"흰색": "white", "검정": "black", "투명": None}
    background = bg_map[bg_color]
    
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
            width=w,
            height=h,
            background_color=background,
            max_words=max_w,
            relative_scaling=0.5,
            min_font_size=10,
            colormap='viridis' if color_theme == "기본 (랜덤)" else None,
            color_func=get_color_func(color_theme)
        ).generate_from_frequencies(keywords_dict)
        
        return wc
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def wordcloud_to_image(wc):
    """워드클라우드를 PIL Image로 변환"""
    # matplotlib figure 생성
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    # 이미지를 메모리 버퍼에 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    # PIL Image로 변환
    image = Image.open(buf)
    return image

def get_image_download_link(img, filename):
    """이미지 다운로드 링크 생성"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 워드클라우드 다운로드</a>'
    return href

# 메인 인터페이스
st.header("📝 키워드 입력")

# 사용법 안내
with st.expander("💡 사용법 보기"):
    st.markdown("""
    **기본 사용법:**
    - 키워드를 쉼표(,) 또는 줄바꿈으로 구분하여 입력하세요
    - 예시: `교육, 학습, 성장, 발전`
    
    **가중치 설정:**
    - 키워드 뒤에 콜론(:)과 숫자를 입력하면 해당 키워드의 크기를 조절할 수 있습니다
    - 예시: `교육:5, 학습:3, 성장:2, 발전:1`
    
    **팁:**
    - 가중치가 높을수록 워드클라우드에서 더 크게 표시됩니다
    - 한글, 영문, 숫자 모두 지원합니다
    """)

# 키워드 입력 영역
keywords_input = st.text_area(
    "키워드를 입력하세요",
    height=150,
    placeholder="예시:\n교육:5\n학습:4\n성장:3\n창의성:2\n협력:2\n소통:1\n\n또는\n\n교육, 학습, 성장, 창의성, 협력, 소통",
    help="쉼표 또는 줄바꿈으로 키워드를 구분하고, 콜론(:) 뒤에 숫자를 입력하면 가중치를 설정할 수 있습니다."
)

# 워드클라우드 생성 버튼
if st.button("🎨 워드클라우드 생성", type="primary"):
    if keywords_input.strip():
        with st.spinner("워드클라우드를 생성하는 중..."):
            # 키워드 파싱
            keywords_dict = parse_keywords(keywords_input)
            
            if keywords_dict:
                st.success(f"총 {len(keywords_dict)}개의 키워드를 인식했습니다.")
                
                # 인식된 키워드 표시
                with st.expander("🔍 인식된 키워드 확인"):
                    for keyword, weight in keywords_dict.items():
                        st.write(f"• **{keyword}**: {weight}")
                
                # 워드클라우드 생성
                wordcloud = create_wordcloud(
                    keywords_dict, 
                    background_color, 
                    color_theme, 
                    width, 
                    height, 
                    max_words
                )
                
                if wordcloud:
                    # 결과 표시
                    st.header("🎉 생성 결과")
                    
                    # 워드클라우드를 이미지로 변환
                    img = wordcloud_to_image(wordcloud)
                    
                    # 이미지 표시
                    st.image(img, caption="생성된 워드클라우드", use_column_width=True)
                    
                    # 다운로드 링크
                    download_link = get_image_download_link(img, "wordcloud.png")
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    # 생성 정보
                    st.info(f"크기: {width}x{height} | 최대 단어 수: {max_words} | 배경: {background_color} | 테마: {color_theme}")
            else:
                st.error("키워드를 인식할 수 없습니다. 입력 형식을 확인해주세요.")
    else:
        st.warning("키워드를 입력해주세요.")

# 푸터
st.markdown("---")
st.markdown("💡 **팁**: 왼쪽 사이드바에서 다양한 설정을 변경해보세요!")
st.markdown("🎓 교육용 워드클라우드 생성기 | Made with Streamlit")