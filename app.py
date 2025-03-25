import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 데이터 불러오기
df = pd.read_csv("data/Obesity Classification.csv")  # 파일 경로 주의

# 전처리
categorical_cols = ["Gender"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
df["Label"] = target_encoder.fit_transform(df["Label"])

X = df.drop(columns=["Label", "ID"])
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)).transpose()

st.set_page_config(page_title="Obesity Prediction Dashboard", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("메뉴 선택", ["Home", "데이터 분석", "데이터 시각화", "머신러닝 보고서"])

def home():
    st.title("Obesity Classification Dashboard")
    st.markdown("""
    ### 데이터 개요
    - **목표**: 나이, 성별, 키, 몸무게 등을 기반으로 비만 등급을 예측하는 분류 모델 구축
    - **출처**: [Kaggle - Obesity Classification.csv] (https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset)
    """)

    st.subheader("비만 등급 예측")
    age = st.number_input("나이", 10, 100, 25)
    height = st.number_input("키 (cm)", 100, 250, 170)
    weight = st.number_input("몸무게 (kg)", 30, 200, 70)
    gender = st.selectbox("성별", label_encoders["Gender"].classes_)

    if st.button("예측 실행"):
        bmi = weight / ((height / 100) ** 2)

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": label_encoders["Gender"].transform([gender]),
            "Height": [height],
            "Weight": [weight],
            "BMI": [bmi]
        })

        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_scaled = scaler.transform(input_data[X.columns])
        prediction = model.predict(input_scaled)
        result_label = target_encoder.inverse_transform(prediction)[0]
        st.success(f"예측된 비만 등급: **{result_label}**")


def data_analysis():
    st.title("데이터 분석")

    st.markdown("""
    - **Age**: 나이 (10세 이상)
    - **Gender**: 성별 (남성/여성)
    - **Height**: 키 (cm 단위)
    - **Weight**: 몸무게 (kg 단위)
    - **BMI**: 체질량지수 (Body Mass Index, 체중 ÷ 키²)
    - **Label**: 비만 등급
        - Underweight: 저체중
        - Normal Weight: 정상체중
        - Overweight: 과체중
        - Obese: 비만
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
        age_range = st.slider("나이 범위", int(df["Age"].min()), int(df["Age"].max()), (20, 40))
        gender_label = st.selectbox("성별", label_encoders["Gender"].classes_)
        gender_encoded = label_encoders["Gender"].transform([gender_label])[0]
        filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Gender"] == gender_encoded)]
        st.dataframe(filtered)

def eda():
    st.title("데이터 시각화")
    tab1, tab2, tab3 = st.tabs(["히스토그램", "상자 그림", "상관관계"])

    with tab1:
        cols = st.columns(2)
        hist_cols = ["Age", "Weight", "Height", "BMI"]
        for i, col in enumerate(hist_cols):
            fig = px.histogram(df, x=col, nbins=30, opacity=0.7, color_discrete_sequence=["indianred"])
            fig.update_layout(title=f"{col} 분포", bargap=0.1, template="simple_white")
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)
            fig.update_layout(title=f"{col} 분포", bargap=0.1, template="simple_white")

    with tab2:
        fig = px.box(df, x="Gender", y="Weight", color=df["Label"].map(lambda x: target_encoder.inverse_transform([x])[0]))
        fig.update_layout(title="Weight by Gender and Obesity Level")
        st.plotly_chart(fig)

    with tab3:
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix.round(2), text_auto=True, title="Correlation Heatmap", width=1000, height=800)
        st.plotly_chart(fig)

def report():
    st.title("머신러닝 보고서")
    st.write(f"모델 정확도: **{accuracy * 100:.2f}%**")

    st.subheader("Classification Report")
    st.dataframe(report_df.style.format("{:.2f}"))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", 
                    x=target_encoder.classes_, y=target_encoder.classes_)
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    fig = px.bar(x=features.values, y=features.index, orientation='h', labels={'x': 'Importance', 'y': 'Features'}, title="Feature Importance")
    st.plotly_chart(fig)

    st.subheader("비만 등급 예측")
    age = st.number_input("나이", 10, 100, 25)
    height = st.number_input("키 (cm)", 100, 250, 170)
    weight = st.number_input("몸무게 (kg)", 30, 200, 70)
    gender = st.selectbox("성별", label_encoders["Gender"].classes_)

    if st.button("예측 실행"):
        bmi = weight / ((height / 100) ** 2)

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": label_encoders["Gender"].transform([gender]),
            "Height": [height],
            "Weight": [weight],
            "BMI": [bmi]
        })

        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_scaled = scaler.transform(input_data[X.columns])
        prediction = model.predict(input_scaled)
        result_label = target_encoder.inverse_transform(prediction)[0]
        st.success(f"예측된 비만 등급: **{result_label}**")


if menu == "Home":
    home()
elif menu == "데이터 분석":
    data_analysis()
elif menu == "데이터 시각화":
    eda()
elif menu == "머신러닝 보고서":
    report()