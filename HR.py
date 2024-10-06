from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

data = pd.read_csv('HR.csv')

# Обработка категориальных переменных
hr_data_encoded = data.copy()
label_encoder = LabelEncoder()

# Преобразуем категориальные переменные 'sales' и 'salary' в числовые
hr_data_encoded['sales'] = label_encoder.fit_transform(hr_data_encoded['sales'])
hr_data_encoded['salary'] = label_encoder.fit_transform(hr_data_encoded['salary'])

# Разделим данные на признаки (X) и целевую переменную (y)
X = hr_data_encoded.drop(columns=['left'])
y = hr_data_encoded['left']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Обучение модели RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=7, max_features='auto', n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)

# Сохраним модель в новом формате
new_model_path = '/mnt/data/HR.pkl'
with open(new_model_path, 'wb') as model_file:
    pickle.dump(rfc, model_file)

new_model_path
