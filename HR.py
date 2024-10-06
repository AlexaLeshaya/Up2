import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Загрузка модели
pickle_in = open("HR.pkl", "rb")
rfc = pickle.load(pickle_in)

# Функция предсказания
def predict_employee_turnover(satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales):
    prediction = rfc.predict([[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales]])
    return prediction

# Основная функция приложения
def main():
    st.title("Прогнозирование текучести кадров в компании")

    st.markdown(
        """
        <div style="background-color:maroon; padding:10px">
        <h2 style="color:white;text-align:center;">Forecasting employee turnover in the company. Based on data from kaggle.com, algorithm RandomForest</h2>
        </div>
        """, unsafe_allow_html=True
    )

    # Ввод данных
    satisfaction_level = st.slider("Уровень удовлетворенности сотрудника (0.1 - 1.0)", 0.1, 1.0, 0.5)
    last_evaluation = st.slider("Оценка работодателя (0.1 - 1.0)", 0.1, 1.0, 0.5)
    number_project = st.number_input("Количество проектов", min_value=1, max_value=10, value=3)
    average_montly_hours = st.number_input("Среднее количество часов в месяц", min_value=80, max_value=320, value=160)
    time_spend_company = st.number_input("Количество лет в компании", min_value=1, max_value=10, value=3)
    Work_accident = st.selectbox("Происшествия на рабочем месте", ("Нет", "Да"))
    promotion_last_5years = st.selectbox("Продвижение за последние 5 лет", ("Нет", "Да"))
    salary = st.selectbox("Уровень заработной платы", ("Низкий", "Средний", "Высокий"))
    sales = st.selectbox("Отдел", ["Отдел продаж", "Технический", "Поддержка", "Управление", "Маркетинг", "РандД", "Аккаунтинг", "HR", "ИТ"])

    # Преобразование данных
    Work_accident = 1 if Work_accident == "Да" else 0
    promotion_last_5years = 1 if promotion_last_5years == "Да" else 0
    salary_map = {"Низкий": 0, "Средний": 1, "Высокий": 2}
    salary = salary_map[salary]
    sales_map = {"Отдел продаж": 0, "Технический": 1, "Поддержка": 2, "Управление": 3, "Маркетинг": 4, "РандД": 5, "Аккаунтинг": 6, "HR": 7, "ИТ": 8}
    sales = sales_map[sales]

    # Кнопка предсказания
    if st.button("Предсказать"):
        result = predict_employee_turnover(satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales)
        st.success(f'Прогноз: {"Сотрудник уволится" if result[0] == 1 else "Сотрудник останется"}')

if __name__ == '__main__':
    main()
