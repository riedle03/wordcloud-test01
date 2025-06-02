# GPT API 키워드 추출 & 워드클라우드 + 네트워크 분석 Streamlit 앱 (한글 글꼴 깨짐 수정 버전)
# -----------------------------------------------------------------------------
# 주요 변경점: draw_network_graph() 함수에서 font_family 매개변수를 제거하여 
#            전역 rcParams 설정(한글 폰트)이 그대로 적용되도록 수정.
# -----------------------------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import numpy as np
import os
import re
import requests
import networkx as nx
from collections import Counter
import itertools

# =========================================
# 1. 한글 폰트 설정
# =========================================
# Pretendard TTF 경로
FONT_PATH = "./fonts/Pretendard-Bold.ttf"

# FontProperties 생성
pretendard_prop = fm.FontProperties(fname=FONT_PATH)
font_name = pretendard_prop.get_name()  # 실제 내부 폰트 이름 ("Pretendard")

# rcParams에도 등록
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False



def setup_korean_font() -> bool:
    """Pretendard 폰트를 matplotlib 기본 폰트로 등록한다."""
    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 – 기호 깨짐 방지
        return True
    st.warning(f"한글 폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
    st.info("`/fonts/Pretendard-Bold.ttf` 경로에 폰트 파일을 추가해 주세요.")
    return False


setup_korean_font()

# =========================================
# 2. Streamlit 페이지 설정
# =========================================
st.set_page_config(
    page_title="GPT 키워드 추출 & 워드클라우드",
    page_icon="🤖",
    layout="wide",
)

# =========================================
# 3. OpenAI API 호출
# =========================================

def call_openai_api(text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """텍스트에서 키워드를 추출하도록 OpenAI ChatCompletion 호출"""
    prompt = f"""
다음 텍스트에서 가장 중요한 키워드들을 추출하고 중요도에 따라 1~10 사이의 가중치를 부여해주세요.

텍스트: "{text}"

요구사항:
1. 10~20개의 핵심 키워드만 추출
2. 각 키워드에 1~10 사이의 가중치 부여 (10이 가장 중요)
3. 불용어(조사, 어미, 일반적인 단어) 제외
4. 복합어나 중요한 구문도 포함 가능
5. 아래 형식으로만 답변해주세요:

키워드A 5, 키워드B 4, 키워드C 3

위 형식 외의 다른 설명은 하지 마세요.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"API 오류: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"네트워크 오류: {e}"
    except Exception as e:
        return f"오류 발생: {e}"

# =========================================
# 4. GPT 응답 파싱
# =========================================

def parse_gpt_response(response_text: str) -> dict[str, int]:
    """'키워드 5, 키워드2 4, ...' 형식을 dict 로 변환"""
    keywords: dict[str, int] = {}
    for item in response_text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            keywords[parts[0].strip()] = int(parts[1])
        else:
            keywords[item] = 5  # 기본 가중치
    return keywords

# =========================================
# 5. 워드클라우드 생성
# =========================================

def create_wordcloud_from_keywords(keywords: dict[str, int], width=800, height=600, bg_color="white"):
    if not keywords:
        return None
    if not os.path.exists(FONT_PATH):
        st.error("폰트 파일이 없습니다. 워드클라우드를 생성할 수 없습니다.")
        return None
    try:
        wc = WordCloud(
            font_path=FONT_PATH,
            width=width,
            height=height,
            background_color=bg_color,
            max_words=100,
            relative_scaling=0.5,
            prefer_horizontal=0.7,
        ).generate_from_frequencies(keywords)
        return wc
    except Exception as e:
        st.error(f"워드클라우드 생성 오류: {e}")
        return None


def wordcloud_to_image(wc, width=800, height=600):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# =========================================
# 6. 네트워크 분석
# =========================================

def create_network_analysis(text: str, keywords: dict[str, int], min_cooccurrence: int = 1):
    if len(keywords) < 2:
        return None, None

    sentences = re.split(r"[.!?]\s*", text)
    co_occurrence = Counter()
    keys = list(keywords.keys())

    for sentence in sentences:
        found = [kw for kw in keys if kw in sentence]
        for a, b in itertools.combinations(found, 2):
            co_occurrence[tuple(sorted((a, b)))] += 1

    filtered = {k: v for k, v in co_occurrence.items() if v >= min_cooccurrence}
    if not filtered:
        return None, None

    G = nx.Graph()
    nodes = set(itertools.chain.from_iterable(filtered.keys()))
    for n in nodes:
        G.add_node(n, weight=keywords.get(n, 1))
    for (a, b), w in filtered.items():
        G.add_edge(a, b, weight=w)

    return G, filtered


# -------------------
# draw_network_graph (FIX)
# -------------------

def draw_network_graph(G: nx.Graph, keywords: dict[str, int]):
    if G is None or len(G.nodes()) < 2:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.8 / np.sqrt(G.number_of_nodes()), iterations=50)

    node_sizes = [keywords.get(n, 1) * 300 for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 1.5 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7, edgecolors="darkblue", linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color="gray", ax=ax)

    # ★ font_family 매개변수를 지정하지 않는다 (rcParams 값이 그대로 적용!)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in G.nodes()},
        font_size=9,
        font_color="black",
        font_weight="bold",
        ax=ax,
        font_family=font_name  # 여기서 정확한 내부 폰트 이름을 줘야 함
    )

    ax.set_title("키워드 네트워크 분석", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# =========================================
# 7. 네트워크 지표 분석 (optional)
# =========================================

def analyze_network_metrics(G: nx.Graph):
    if G is None or len(G.nodes()) < 2:
        return {}
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G) if nx.is_connected(G) else "N/A",
        "components": nx.number_connected_components(G),
    }
    deg = nx.degree_centrality(G)
    bt = nx.betweenness_centrality(G)
    cl = nx.closeness_centrality(G)
    metrics["degree_top5"] = dict(sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5])
    metrics["betweenness_top5"] = dict(sorted(bt.items(), key=lambda x: x[1], reverse=True)[:5])
    metrics["closeness_top5"] = dict(sorted(cl.items(), key=lambda x: x[1], reverse=True)[:5])
    return metrics

# =========================================
# 8. Streamlit UI
# =========================================

st.title("🤖 GPT API 키워드 추출 & 워드클라우드 생성기")
st.markdown("GPT를 활용해 텍스트에서 핵심 키워드를 추출하고 시각화합니다!")

st.sidebar.header("🔑 API 설정")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_choice = st.sidebar.selectbox("GPT 모델", ["gpt-4o-mini", "gpt-3.5-turbo"], index=0)

st.sidebar.header("🎨 시각화 설정")
wc_width = st.sidebar.slider("워드클라우드 너비", 400, 1200, 800)
wc_height = st.sidebar.slider("워드클라우드 높이", 300, 800, 600)
bg_color = st.sidebar.selectbox("배경색", ["white", "black"], index=0)

show_network = st.sidebar.checkbox("🌐 네트워크 분석 포함", value=True)
min_co = st.sidebar.slider("최소 동시출현 횟수", 1, 5, 1)

st.header("📝 텍스트 입력")
text_input = st.text_area("분석할 텍스트를 입력하세요", height=200)

if st.button("🤖 GPT로 키워드 추출", disabled=not api_key):
    if not text_input.strip():
        st.warning("텍스트를 입력해 주세요.")
    else:
        with st.spinner("GPT 호출 중..."):
            response = call_openai_api(text_input, api_key, model_choice)
        if response.startswith("API 오류") or response.startswith("네트워크 오류"):
            st.error(response)
        else:
            st.success("키워드 추출 완료!")
            st.subheader("📜 GPT 응답")
            st.code(response)

            keywords = parse_gpt_response(response)
            if not keywords:
                st.error("응답 파싱 실패")
            else:
                # 키워드 목록 표시
                st.subheader("📊 추출된 키워드")
                for k, v in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- **{k}**: {v}")

                # 워드클라우드
                st.subheader("☁️ 워드클라우드")
                wc = create_wordcloud_from_keywords(keywords, wc_width, wc_height, bg_color)
                if wc:
                    img_wc = wordcloud_to_image(wc, wc_width, wc_height)
                    st.image(img_wc, caption="WordCloud", use_column_width=True)

                # 네트워크 분석
                if show_network:
                    st.subheader("🌐 키워드 네트워크")
                    G, co = create_network_analysis(text_input, keywords, min_co)
                    if G:
                        img_net = draw_network_graph(G, keywords)
                        st.image(img_net, caption="Keyword Network", use_column_width=True)
                        # 지표
                        metrics = analyze_network_metrics(G)
                        st.markdown("### 네트워크 지표")
                        st.json(metrics)
                    else:
                        st.info("네트워크를 구성할 충분한 동시출현 키워드가 없습니다.")
