import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title('👀 Прогнозирование оттока сотрудников компании')

# Загрузка данных
with st.expander('Исходные данные'):
    df = pd.read_csv('HR_gender.csv')
  
    st.write('**X**')
    X_raw = df.drop('left', axis=1)  # 'left' - целевая переменная (увольнение)
    X_raw

    st.write('**y**')
    y_raw = df['left']
    y_raw

# Визуализация данных
with st.expander('Визуализация данных'):
    st.scatter_chart(data=df, x='satisfaction_level', y='average_montly_hours', color='left')
    st.bar_chart(data=df['salary'].value_counts())

# Пользовательские параметры
with st.sidebar:
    st.header('Введите характеристики сотрудника')
    satisfaction_level = st.slider('Уровень удовлетворенности сотрудника', 0.0, 1.0, 0.5)
    last_evaluation = st.slider('Последняя оценка компанией', 0.0, 1.0, 0.7)
    number_project = st.slider('Количество проектов', 1, 7, 4)
    average_montly_hours = st.slider('Среднее количество рабочих часов', 96, 310, 200)
    time_spend_company = st.slider('Время в компании (лет)', 1, 10, 3)
    Work_accident = st.selectbox('Происходил ли несчастный случай на работе?', (0, 1))
    promotion_last_5years = st.selectbox('Повышение за последние 5 лет?', (0, 1))
    sales = st.selectbox('Отдел', df['sales'].unique())
    salary = st.selectbox('Уровень зарплаты', df['salary'].unique())
  
    # Собираем вводные данные в DataFrame
    data = {'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': Work_accident,
            'promotion_last_5years': promotion_last_5years,
            'sales': sales,
            'salary': salary}
    input_df = pd.DataFrame(data, index=[0])

    # Объединяем с исходными данными для корректного кодирования
    input_data = pd.concat([input_df, X_raw], axis=0)

# Кодирование категориальных переменных
encode = ['sales', 'salary']
input_data_encoded = pd.get_dummies(input_data, columns=encode)

# Отделяем строку с вводом пользователя
X_input = input_data_encoded[:1]

# Обработка категориальных переменных для основного набора данных
df_encoded = pd.get_dummies(X_raw, columns=encode)

# Разделение данных на обучающую и тестовую выборки
X = df_encoded
y = y_raw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели RandomForestClassifier с корректным параметром max_features
clf = RandomForestClassifier(n_estimators=7, max_features='sqrt', n_jobs=2, random_state=1)
clf.fit(X_train, y_train)

# Прогнозирование
prediction = clf.predict(X_input)
prediction_proba = clf.predict_proba(X_input)

# Отображение результата
st.subheader('Вероятность увольнения')
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Останется', 'Уволится'])
st.dataframe(df_prediction_proba)

# Вывод результата
if prediction[0] == 1:
    st.success("Сотрудник, вероятно, уволится.")
else:
    st.success("Сотрудник, вероятно, останется.")
