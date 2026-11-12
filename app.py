import pickle
import numpy as np
import streamlit as st
from Orange.data import Table, Domain

# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 모델 로드
# ---------------------------------------------------------
st.set_page_config(
    page_title="어린이 보호구역 교통사고 위험도 예측",
    page_icon="🚦",
    layout="centered"
)

# ⚠️ [중요] 오렌지3 학습 데이터셋의 실제 컬럼명과 동일하게 맞춰주세요.
VAR_SIDO = "시도"           # 범주형 변수명 (예: 시도, 시군구 등)
VAR_ACCIDENT = "사고건수"    # 연속형 변수명 1
VAR_MINOR = "경상자수"       # 연속형 변수명 2

@st.cache_resource
def load_orange_model(model_path: str):
    """오렌지3 .pkcls 모델 파일을 캐싱하여 로드합니다."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("🚨 'traffic_casualty_model.pkcls' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 배치해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. UI 구성: 제목 및 입력 위젯 (사이드바 또는 메인)
# ---------------------------------------------------------
st.title("🚦 어린이 보호구역 사고 위험도 예측기")
st.markdown("지역과 사고 통계를 조절하면 **예상 사망자 수**를 실시간으로 분석하고 위험 신호등을 안내합니다.")
st.divider()

# 모델의 Domain(변수 구조)에서 시도 목록 자동 추출 (실패 시 기본 목록 사용)
default_sido_list = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]

try:
    sido_var = model.domain[VAR_SIDO]
    sido_options = list(sido_var.values) if hasattr(sido_var, "values") else default_sido_list
except KeyError:
    sido_options = default_sido_list

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📍 지역 및 발생 현황")
    selected_sido = st.selectbox("시도 선택", options=sido_options)
    accident_count = st.slider("사고건수 (건)", min_value=0, max_value=500, value=30, step=1)
    minor_injury = st.slider("경상자수 (명)", min_value=0, max_value=500, value=25, step=1)

# ---------------------------------------------------------
# 3. 오렌지3 모델 예측 함수 실행
# ---------------------------------------------------------
def predict_casualties(orange_model, sido: str, accidents: int, minors: int) -> float:
    """입력값을 Orange3 Table 객체로 변환하여 예측값을 반환합니다."""
    domain = orange_model.domain
    
    # 모델이 요구하는 입력 특성(attributes) 순서대로 데이터 행 구성
    row = []
    for attr in domain.attributes:
        if attr.name == VAR_SIDO:
            # 범주형 변수는 문자열 그대로 입력하거나 인덱스 매핑
            row.append(sido)
        elif attr.name == VAR_ACCIDENT:
            row.append(float(accidents))
        elif attr.name == VAR_MINOR:
            row.append(float(minors))
        else:
            # 혹시 학습에 포함된 다른 변수가 있다면 결측치(np.nan) 처리
            row.append(np.nan)
            
    # 타겟(예측할 값) 자리는 결측치로 비워둠
    target_row = [np.nan] * len(domain.class_vars)
    
    # Orange3 Table 생성
    input_table = Table.from_list(domain, [row + target_row])
    
    # 예측 실행 (회귀 모델 기준 스칼라 값 추출)
    prediction = orange_model(input_table)
    predicted_value = float(prediction[0])
    
    # 음수 예측 방지
    return max(0.0, predicted_value)

try:
    pred_deaths = predict_casualties(model, selected_sido, accident_count, minor_injury)
except Exception as e:
    st.warning(f"⚠️ 예측 실행 중 데이터 매핑 오류가 발생했습니다. 코드 상단의 변수명을 확인하세요.\n\n상세 오류: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. 결과 시각화: 대형 신호등 & 안전 수칙 카드
# ---------------------------------------------------------
# 위험도 임계값 설정 (데이터 특성에 맞게 숫자 조정 가능)
if pred_deaths < 0.5:
    status_emoji = "🟢"
    status_title = "안전 (양호)"
    bg_color = "#E8F5E9"
    border_color = "#2E7D32"
    text_color = "#1B5E20"
    tips = [
        "**시속 30km 이하 서행 유지:** 어린이 보호구역 내 제한속도를 철저히 준수하세요.",
        "**횡단보도 앞 일시정지:** 보행자가 없더라도 횡단보도 앞에서는 반드시 일시정지 후 출발합니다.",
        "**방어운전 습관화:** 불법 주정차 차량 사이에서 어린이가 갑자기 튀어나올 수 있음을 항상 염두에 두세요."
    ]
elif pred_deaths < 2.0:
    status_emoji = "🟡"
    status_title = "주의 (경계)"
    bg_color = "#FFFDE7"
    border_color = "#F9A825"
    text_color = "#F57F17"
    tips = [
        "**등·하교 시간대 집중 주의:** 오전 8~9시, 오후 1~4시 사이에는 보행량이 급증하므로 각별히 주의하세요.",
        "**전방 주시 및 스마트폰 사용 금지:** 순간의 방심이 중대 사고로 이어질 수 있습니다.",
        "**우회전 시 2회 일시정지:** 교차로 우회전 시 전방 차량 신호와 횡단보도 보행자 신호를 이중으로 확인하세요."
    ]
else:
    status_emoji = "🔴"
    status_title = "위험 (심각)"
    bg_color = "#FFEBEE"
    border_color = "#C62828"
    text_color = "#B71C1C"
    tips = [
        "**가급적 우회 도로 이용:** 사고 위험도가 매우 높은 구간이므로 불필요한 스쿨존 진입을 자제하세요.",
        "**서행 및 비상등 활용:** 시야 확보가 어려운 구간에서는 서행하며 주변 차량과 보행자에게 주의를 알리세요.",
        "**어린이 보행 안전 지도 강화:** 보호자 동반 보행 및 안전 펜스·과속단속카메라 등 시설물 점검이 시급합니다."
    ]

with col2:
    st.subheader("📊 실시간 예측 및 위험도 판정")
    
    # 대형 신호등 카드 렌더링
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 3px solid {border_color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="font-size: 72px; line-height: 1.1; margin-bottom: 10px;">
                {status_emoji}
            </div>
            <h2 style="color: {text_color}; margin: 0 0 8px 0;">
                위험 단계: {status_title}
            </h2>
            <p style="font-size: 18px; color: #333; margin: 0;">
                예상 사망자 수: <strong style="font-size: 24px; color: {text_color};">{pred_deaths:.2f} 명</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 어린이 보호구역 안전 수칙 카드
st.subheader("🛡️ 어린이 보호구역 핵심 안전 수칙")
with st.container(border=True):
    st.markdown(f"**[{selected_sido}] 지역 맞춤형 행동 요령**")
    for tip in tips:
        st.markdown(f"- {tip}")