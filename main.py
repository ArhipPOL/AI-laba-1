"""
Лабораторная работа №1
Предобработка данных с использованием pandas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    return pd.read_csv(file_path)

def analyze_missing_values(df):
    """Анализ пропущенных значений"""
    print("Анализ пропущенных значений:")
    missing_info = pd.DataFrame({
        'Количество пропусков': df.isnull().sum(),
        'Процент пропусков': (df.isnull().sum() / len(df)) * 100
    })
    return missing_info

def fill_missing_values(df):
    """Заполнение пропущенных значений"""
    df_filled = df.copy()

    # Числовые столбцы
    numeric_columns = df_filled.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if df_filled[col].isnull().sum() > 0:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)

    # Категориальные столбцы
    categorical_columns = df_filled.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_filled[col].isnull().sum() > 0:
            mode_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
            df_filled[col].fillna(mode_value, inplace=True)

    return df_filled

def normalize_data(df, columns_to_normalize):
    """Нормализация данных"""
    df_normalized = df.copy()

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    # Нормализуем ВСЕ столбцы сразу для корректной работы
    minmax_data = minmax_scaler.fit_transform(df[columns_to_normalize])
    df_minmax = pd.DataFrame(minmax_data, columns=[f'{col}_minmax' for col in columns_to_normalize])

    # StandardScaler
    standard_scaler = StandardScaler()
    standard_data = standard_scaler.fit_transform(df[columns_to_normalize])
    df_standard = pd.DataFrame(standard_data, columns=[f'{col}_standard' for col in columns_to_normalize])

    # Объединяем все датафреймы
    df_normalized = pd.concat([df_normalized, df_minmax, df_standard], axis=1)

    return df_normalized

def encode_categorical_data(df, categorical_columns):
    """One-Hot Encoding категориальных данных"""
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def main():
    """Основная функция"""
    try:
        # Загрузка данных
        print("Загрузка данных...")
        df = load_data('titanic.csv')

        # Анализ исходных данных
        print("=== ИСХОДНЫЕ ДАННЫЕ ===")
        print(f"Размер датасета: {df.shape}")
        print(df.info())
        missing_info = analyze_missing_values(df)
        print(missing_info)

        # Заполнение пропущенных значений
        print("\n=== ЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
        df_filled = fill_missing_values(df)
        print("Проверка после заполнения:")
        print(f"Осталось пропусков: {df_filled.isnull().sum().sum()}")

        # Нормализация
        print("\n=== НОРМАЛИЗАЦИЯ ДАННЫХ ===")
        numeric_cols = ['Age', 'Fare']
        # Проверяем, что столбцы существуют в датасете
        available_numeric_cols = [col for col in numeric_cols if col in df_filled.columns]
        print(f"Нормализуемые столбцы: {available_numeric_cols}")

        df_normalized = normalize_data(df_filled, available_numeric_cols)
        print(f"Размер после нормализации: {df_normalized.shape}")

        # Кодирование категориальных данных
        print("\n=== КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ ===")
        cat_cols = ['Sex', 'Embarked', 'Pclass']
        # Проверяем, что столбцы существуют
        available_cat_cols = [col for col in cat_cols if col in df_normalized.columns]
        print(f"Кодируемые столбцы: {available_cat_cols}")

        df_encoded = encode_categorical_data(df_normalized, available_cat_cols)
        print(f"Размер после кодирования: {df_encoded.shape}")

        # Удаление исходных столбцов (ОПЦИОНАЛЬНО - можно оставить для анализа)
        print("\n=== ФИНАЛЬНАЯ ОБРАБОТКА ===")
        columns_to_drop = available_numeric_cols + available_cat_cols
        df_final = df_encoded.drop(columns=columns_to_drop, errors='ignore')
        print(f"Размер после удаления исходных столбцов: {df_final.shape}")

        # Разделение на train/test
        print("\n=== РАЗДЕЛЕНИЕ НА TRAIN/TEST ===")
        train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42)

        # Сохранение результатов
        print("\n=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")
        train_df.to_csv('processed_titanic_train.csv', index=False)
        test_df.to_csv('processed_titanic_test.csv', index=False)

        # Создание файла requirements.txt
        with open('requirements.txt', 'w') as f:
            f.write("""pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
""")

        print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
        print(f"Исходный размер: {df.shape}")
        print(f"Финальный размер: {df_final.shape}")
        print(f"Train set: {train_df.shape}")
        print(f"Test set: {test_df.shape}")
        print(f"Сохраненные файлы:")
        print("- processed_titanic_train.csv")
        print("- processed_titanic_test.csv")
        print("- requirements.txt")
        print("\n✅ Обработка данных завершена успешно!")

    except FileNotFoundError:
        print("❌ Ошибка: Файл 'titanic.csv' не найден!")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()

