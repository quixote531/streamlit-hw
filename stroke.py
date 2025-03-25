import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 데이터 불러오기
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# 전처리
df.drop(columns=["id"], inplace=True)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

df["stroke"] = df["stroke"].astype(int)
df["hypertension"] = df["hypertension"].astype(int)
df["heart_disease"] = df["heart_disease"].astype(int)

categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop(columns=["stroke"])
y = df["stroke"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("메뉴 선택", ["Home", "데이터 분석", "데이터 시각화", "머신러닝 보고서"])

def home():
    st.title("Stroke Prediction Dashboard")
    st.markdown("""
    ### 데이터 개요
    
    - **출처**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
    - **목표**: 건강 정보를 바탕으로 뇌졸중 발생 여부 예측
    
    세계보건기구(WHO)에 따르면 뇌졸중은 전 세계 사망 원인 중 두 번째로 높은 비율을 차지하며, 전체 사망의 약 11%를 차지합니다. 
    이 데이터는 환자의 성별, 나이, 고혈압, 심장질환, 혈당 수치, BMI, 흡연 상태 등을 포함하고 있어 뇌졸중 예측에 유용합니다.
    """)

def data_analysis():
    st.title("데이터 분석")
    st.markdown("""
    ### 주요 변수 설명
    - **gender**: 성별
    - **age**: 나이
    - **hypertension**: 고혈압 여부
    - **heart_disease**: 심장 질환 여부
    - **ever_married**: 결혼 여부
    - **work_type**: 직업
    - **Residence_type**: 거주지 유형
    - **avg_glucose_level**: 평균 혈당
    - **bmi**: 체질량지수
    - **smoking_status**: 흡연 상태
    - **stroke**: 뇌졸중 발생 여부
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["상위 데이터", "기술 통계", "컬럼 보기", "조건 검색"])

    with tab1:
        st.dataframe(df.head(10))

    with tab2:
        st.dataframe(df.describe())

    with tab3:
        selected_cols = st.multiselect("컬럼 선택", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_cols])

    with tab4:
        age_range = st.slider("나이 범위", 0, 100, (40, 70))
        gender = st.selectbox("성별", df["gender"].unique())
        filtered = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) & (df["gender"] == gender)]
        st.dataframe(filtered)

def eda():
    st.title("데이터 시각화")
    st.markdown("#### 주요 변수들의 분포를 확인하고, 뇌졸중과 관련된 패턴을 시각화합니다.")

    tab1, tab2, tab3, tab4 = st.tabs(["변수별 히스토그램", "고혈압/나이/뇌졸중 관계", "카테고리별 뇌졸중 비율", "상관관계"])

    with tab1:
        st.subheader("수치형 변수 히스토그램")
        numeric_cols = ["age", "avg_glucose_level", "bmi"]
        col1, col2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax, bins=30, color="skyblue")
            ax.set_title(f"Distribution of {col}")
            (col1 if i % 2 == 0 else col2).pyplot(fig)

        st.markdown("---")
        
    with tab2:
        st.subheader("고혈압 여부와 나이의 관계 (Stroke 기준)")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="hypertension", y="age", hue="stroke", palette="Set1", ax=ax)
        ax.set_xticklabels(["No Hypertension", "Hypertension"])
        ax.set_title("Hypertension vs Age by Stroke")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["No Stroke", "Stroke"])
        st.pyplot(fig)

    with tab3:
        st.subheader("Stroke Rate by Categorical Features")

        fig2, ax2 = plt.subplots()
        sns.barplot(x="work_type", y="stroke", data=df, estimator =np.mean, ci=None, palette="Set3", ax=ax2)
        ax2.set_title("Stroke Rate by Work Type")
        ax2.set_xticklabels(label_encoders["work_type"].inverse_transform(sorted(df["work_type"].unique())), rotation=30)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.barplot(x="smoking_status", y="stroke", data=df, estimator=np.mean, ci=None, palette="Pastel1", ax=ax3)
        ax3.set_title("Stroke Rate by Smoking Status")
        ax3.set_xticklabels(label_encoders["smoking_status"].inverse_transform(sorted(df["smoking_status"].unique())), rotation=30)
        st.pyplot(fig3)

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)



def report():
    st.title("머신러닝 보고서")
    st.write(f"모델 정확도: **{accuracy * 100:.2f}%**")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("분류 리포트")
    renamed_report_df = report_df.rename(index={"0": "No Stroke", "1": "Stroke"})
    st.dataframe(renamed_report_df.style.format("{:.2f}"))

    st.subheader("특성 중요도")
    importances = model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    features.plot(kind='barh', ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

    st.subheader("새로운 데이터 예측")
    age = st.number_input("나이", 0, 100, 45)
    hypertension = st.selectbox("고혈압 여부", [0, 1])
    heart_disease = st.selectbox("심장 질환 여부", [0, 1])
    ever_married = st.selectbox("결혼 여부", label_encoders["ever_married"].classes_)
    work_type = st.selectbox("직업", label_encoders["work_type"].classes_)
    residence = st.selectbox("거주지", label_encoders["Residence_type"].classes_)
    avg_glucose = st.number_input("평균 혈당", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 28.0)
    smoking_status = st.selectbox("흡연 상태", label_encoders["smoking_status"].classes_)
    gender = st.selectbox("성별", label_encoders["gender"].classes_)

    if st.button("예측 실행"):
        user_input = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence],
            "avg_glucose_level": [avg_glucose],
            "bmi": [bmi],
            "smoking_status": [smoking_status]
        })

        for col in categorical_cols:
            le = label_encoders[col]
            user_input[col] = le.transform(user_input[col].astype(str))

        user_scaled = scaler.transform(user_input)
        prediction = model.predict(user_scaled)
        result = "⚠️ 뇌졸중 발생 가능성 있음" if prediction[0] == 1 else "✅ 뇌졸중 위험 낮음"
        st.success(f"예측 결과: {result}")

if menu == "Home":
    home()
elif menu == "데이터 분석":
    data_analysis()
elif menu == "데이터 시각화":
    eda()
elif menu == "머신러닝 보고서":
    report()
