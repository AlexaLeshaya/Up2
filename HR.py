import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

pickle_in = open("HR.pkl","rb")
rfc=pickle.load(pickle_in)

def predict_employee_turnover(satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales):
    
    prediction=rfc.predict([[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales]])
    print(prediction)
    return prediction




def main():
    st.title("Прогнозирование текучести кадров в компании")
    html_temp = """
    <div style = "background-color:maroon; padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting employee turnover in the company. Based on data from kaggle.com, algorithm RandomForest </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    satisfaction_level=st.radio("Уровень удовлетворенности сотрудника:  (0.1 - совсем не удовлетворен и 1.0 - высшая степень удовлетворенности)", 
               key="satisfaction_level",
               options=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], )
    if satisfaction_level=="0.1":
        satisfaction_level=0.1
    elif satisfaction_level=="0.2":
        satisfaction_level=0.2
    elif satisfaction_level=="0.3":
        satisfaction_level=0.3
    elif satisfaction_level=="0.4":
        satisfaction_level=0.4
    elif satisfaction_level=="0.5":
        satisfaction_level=0.5
    elif satisfaction_level=="0.6":
        satisfaction_level=0.6
    elif satisfaction_level=="0.7":
        satisfaction_level=0.7
    elif satisfaction_level=="0.8":
        satisfaction_level=0.8
    elif satisfaction_level=="0.9":
        satisfaction_level=0.9
    elif satisfaction_level=="1.0":
        satisfaction_level=1.0
       
    last_evaluation=st.radio("Оценка работодателя: (0.1 - совсем не доволен вами и 1.0 - очень доволен вами)",
               key="last_evaluation",
               options=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], )
    if last_evaluation=="0.1":
        last_evaluation=0.1
    elif last_evaluation=="0.2":
        last_evaluation=0.2
    elif last_evaluation=="0.3":
        last_evaluation=0.3
    elif last_evaluation=="0.4":
        last_evaluation=0.4
    elif last_evaluation=="0.5":
        last_evaluation=0.5
    elif last_evaluation=="0.6":
        last_evaluation=0.6
    elif last_evaluation=="0.7":
        last_evaluation=0.7
    elif last_evaluation=="0.8":
        last_evaluation=0.8
    elif last_evaluation=="0.9":
        last_evaluation=0.9
    elif last_evaluation=="1.0":
        last_evaluation=1.0
        
    number_project = st.text_input("Количество выполняемых проектов (от 2 до 7)","")
    average_montly_hours = st.text_input("Среднее количество рабочих часов за месяц (от 96 до 310)","")
    time_spend_company = st.text_input("Время работы в компании в годах (от 2 до 10)","")
    
    Work_accident=st.radio("Был ли несчастный случай на рабочем месте:",
               key="Work_accident",
               options=["Да", "Нет"], )
    if Work_accident=="Да":
        Work_accident=1
    elif Work_accident=="Нет":
        Work_accident=0
    
    promotion_last_5years=st.radio("Было повышение за последние 5 лет:",
               key="promotion_last_5years",
               options=["Да", "Нет"], )
    if promotion_last_5years=="Да":
        promotion_last_5years=1
    elif promotion_last_5years=="Нет":
        promotion_last_5years=0
    
    salary=st.radio("Уровень заработной платы:",
               key="salary",
               options=["Низкий", "Средний", "Высокий"], )
    if salary=="Низкий":
        salary=0
    elif salary=="Средний":
        salary=1
    elif salary=="Высокий":
        salary=2
        
    sales=st.radio("Отдел:", 
               key="sales",
               options=["Отдел IT","Отдел исследований и разработок","Бухгалтерия","Отдел кадров","Отдел управления","Отдел маркетинга","Отдел управления продуктом","Отдел продаж","Отдел поддержки","Технический отдел"], )
    if sales=="Отдел IT":
        sales=6
    elif sales=="Отдел исследований и разработок":
        sales=9
    elif sales=="Бухгалтерия":
        sales=1
    elif sales=="Отдел кадров":
        sales=2
    elif sales=="Отдел управления":
        sales=5
    elif sales=="Отдел маркетинга":
        sales=8
    elif sales=="Отдел управления продуктом":
        sales=7
    elif sales=="Отдел продаж":
        sales=0
    elif sales=="Отдел поддержки":
        sales=6
    elif sales=="Технический отдел":
        sales=3
        
        
    result={}
    if st.button("Predict"):
        result=int(predict_employee_turnover(satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, salary, sales))
    
    st.success('Forecasting employee turnover in the company is {}'. format(result))
       
    

if __name__=='__main__':
    main()
    
    
