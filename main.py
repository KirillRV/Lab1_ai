import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r'C:\Users\Kirill\PycharmProjects\lab1\Titanic-Dataset.csv')

print("Первые 5 строк датасета:")
print(df.head())

#Проверка пропущенных значений в каждом столбце
print("\nКоличество пропущенных значений в каждом столбце:")
print(df.isnull().sum())

#Заполнение пропущенных значений
#Для 'Age' используем медиану
df['Age'] = df['Age'].fillna(df['Age'].median())
#Для 'Cabin' используем 'Unknown'
df['Cabin'] = df['Cabin'].fillna('Unknown')
#Для 'Embarked' используем моду
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Проверка после заполнения
print("\nКоличество пропущенных значений после заполнения:")
print(df.isnull().sum())

#Нормализация данных (применяем к 'Age' и 'Fare')
scaler = MinMaxScaler()
numeric_cols = ['Age', 'Fare']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nНормализованные данные (Age и Fare в диапазоне [0, 1]):")
print(df[numeric_cols].head())

categorical_cols = ['Sex', 'Embarked', 'Pclass']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nДанные после преобразования категориальных признаков:")
print(df.head())

#Сохранение обработанного датасета
df.to_csv(r'C:\Users\Kirill\PycharmProjects\lab1\processed_Titanic-Dataset.csv', index=False)
print("\nОбработанный датасет сохранён как 'C:\\Users\\Kirill\\PycharmProjects\\lab1\\processed_Titanic-Dataset.csv'")