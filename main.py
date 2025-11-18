"""
Лабораторная работа №1
Предобработка данных с использованием pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    return pd.read_csv(file_path)

def display_basic_info(df):
    """Вывод базовой информации о датасете"""
    print("=== БАЗОВАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")
    print(f"Размер датасета: {df.shape}")
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")

    print("\nПервые 5 строк датасета:")
    print(df.head())

    print("\nПоследние 5 строк датасета:")
    print(df.tail())

    print("\nНазвания столбцов:")
    print(df.columns.tolist())

    print("\nТипы данных:")
    print(df.dtypes)

    print("\nБазовая статистика числовых столбцов:")
    print(df.describe())

    print("\nБазовая статистика категориальных столбцов:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe())

def visualize_missing_values(df):
    """Визуализация пропущенных значений"""
    plt.figure(figsize=(12, 6))

    # Heatmap пропущенных значений
    plt.subplot(1, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Heatmap пропущенных значений')

    # Bar plot пропущенных значений по столбцам
    plt.subplot(1, 2, 2)
    missing_counts = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Количество пропусков': missing_counts,
        'Процент пропусков': missing_percent
    }).sort_values('Процент пропусков', ascending=False)

    missing_df[missing_df['Количество пропусков'] > 0]['Процент пропусков'].plot(kind='bar')
    plt.title('Процент пропусков по столбцам')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_missing_values(df):
    """Анализ пропущенных значений"""
    print("=== АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
    missing_info = pd.DataFrame({
        'Количество пропусков': df.isnull().sum(),
        'Процент пропусков': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Процент пропусков', ascending=False)

    # Выводим только столбцы с пропусками
    missing_columns = missing_info[missing_info['Количество пропусков'] > 0]
    if len(missing_columns) > 0:
        print("Столбцы с пропущенными значениями:")
        print(missing_columns)
    else:
        print("Пропущенных значений не обнаружено!")

    print(f"\nОбщее количество пропусков в датасете: {df.isnull().sum().sum()}")

    return missing_info

def fill_missing_values(df):
    """Заполнение пропущенных значений"""
    print("=== ЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
    df_filled = df.copy()

    # Анализ до заполнения
    missing_before = df.isnull().sum()
    columns_with_missing = missing_before[missing_before > 0].index.tolist()

    if not columns_with_missing:
        print("Нет пропущенных значений для заполнения.")
        return df_filled

    print("Столбцы с пропущенными значениями:")
    for col in columns_with_missing:
        print(f"  - {col}: {missing_before[col]} пропусков")

    # Числовые столбцы
    numeric_columns = df_filled.select_dtypes(include=['number']).columns
    numeric_with_missing = [col for col in numeric_columns if col in columns_with_missing]

    for col in numeric_with_missing:
        if df_filled[col].isnull().sum() > 0:
            median_val = df_filled[col].median()
            df_filled[col].fillna(median_val, inplace=True)
            print(f"Заполнен числовой столбец '{col}' медианой: {median_val:.2f}")

    # Категориальные столбцы
    categorical_columns = df_filled.select_dtypes(include=['object']).columns
    categorical_with_missing = [col for col in categorical_columns if col in columns_with_missing]

    for col in categorical_with_missing:
        if df_filled[col].isnull().sum() > 0:
            mode_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
            df_filled[col].fillna(mode_value, inplace=True)
            print(f"Заполнен категориальный столбец '{col}' модой: '{mode_value}'")

    # Проверка после заполнения
    missing_after = df_filled.isnull().sum().sum()
    print(f"\nПроверка после заполнения:")
    print(f"Осталось пропусков: {missing_after}")

    return df_filled

def normalize_data(df):
    """Нормализация данных"""
    print("=== НОРМАЛИЗАЦИЯ ДАННЫХ ===")
    df_normalized = df.copy()

    # Автоматическое определение числовых столбцов
    numeric_columns = df_normalized.select_dtypes(include=['number']).columns.tolist()

    # Исключаем целевые переменные если они есть
    exclude_columns = ['Survived', 'target']  # Добавьте другие целевые переменные при необходимости
    columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]

    if not columns_to_normalize:
        print("Нет числовых столбцов для нормализации.")
        return df_normalized

    print(f"Нормализуемые столбцы: {columns_to_normalize}")

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    minmax_data = minmax_scaler.fit_transform(df_normalized[columns_to_normalize])
    df_minmax = pd.DataFrame(minmax_data,
                           columns=[f'{col}_minmax' for col in columns_to_normalize],
                           index=df_normalized.index)

    # StandardScaler
    standard_scaler = StandardScaler()
    standard_data = standard_scaler.fit_transform(df_normalized[columns_to_normalize])
    df_standard = pd.DataFrame(standard_data,
                             columns=[f'{col}_standard' for col in columns_to_normalize],
                             index=df_normalized.index)

    # Объединяем все датафреймы
    df_normalized = pd.concat([df_normalized, df_minmax, df_standard], axis=1)

    print(f"Добавлено {len(columns_to_normalize) * 2} новых столбцов после нормализации")
    print(f"Размер после нормализации: {df_normalized.shape}")

    return df_normalized

def encode_categorical_data(df):
    """One-Hot Encoding категориальных данных"""
    print("=== КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ ===")

    # Автоматическое определение категориальных столбцов
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_columns:
        print("Нет категориальных столбцов для кодирования.")
        return df

    print(f"Категориальные столбцы для кодирования: {categorical_columns}")

    # Применяем One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, prefix_sep='_')

    print(f"Размер после кодирования: {df_encoded.shape}")
    print(f"Количество столбцов до кодирования: {len(df.columns)}")
    print(f"Количество столбцов после кодирования: {len(df_encoded.columns)}")

    # Показываем новые созданные столбцы
    new_columns = set(df_encoded.columns) - set(df.columns)
    print(f"Создано {len(new_columns)} новых бинарных столбцов")

    return df_encoded

def create_requirements_file():
    """Создание файла requirements.txt"""
    requirements = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("Создан файл requirements.txt")

def main():
    """Основная функция"""
    try:
        # Загрузка данных
        print("=" * 60)
        print("ЛАБОРАТОРНАЯ РАБОТА №1: ПРЕДОБРАБОТКА ДАННЫХ")
        print("=" * 60)

        print("\nЗагрузка данных...")
        df = load_data('titanic.csv')

        # Шаг 1: Вывод данных на экран
        display_basic_info(df)

        # Шаг 2: Визуализация пропущенных значений
        visualize_missing_values(df)

        # Шаг 3: Анализ пропущенных значений
        missing_info = analyze_missing_values(df)

        # Шаг 4: Заполнение пропущенных значений
        df_filled = fill_missing_values(df)

        # Шаг 5: Нормализация данных
        df_normalized = normalize_data(df_filled)

        # Шаг 6: Кодирование категориальных данных
        df_encoded = encode_categorical_data(df_normalized)

        # Финальная обработка
        print("\n=== ФИНАЛЬНАЯ ОБРАБОТКА ===")

        # Удаляем исходные столбцы, которые были нормализованы и закодированы
        numeric_columns = df_filled.select_dtypes(include=['number']).columns
        categorical_columns = df_filled.select_dtypes(include=['object']).columns

        # Исключаем целевые переменные из удаления
        exclude_from_drop = ['Survived', 'target']
        columns_to_drop = [col for col in numeric_columns if col not in exclude_from_drop] + categorical_columns.tolist()

        df_final = df_encoded.drop(columns=columns_to_drop, errors='ignore')
        print(f"Удалено исходных столбцов: {len(columns_to_drop)}")
        print(f"Финальный размер датасета: {df_final.shape}")

        # Разделение на train/test
        print("\n=== РАЗДЕЛЕНИЕ НА TRAIN/TEST ===")
        train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final.get('Survived', None))
        print(f"Train set: {train_df.shape}")
        print(f"Test set: {test_df.shape}")

        # Сохранение результатов
        print("\n=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")
        train_df.to_csv('processed_titanic_train.csv', index=False)
        test_df.to_csv('processed_titanic_test.csv', index=False)

        # Создание файла requirements.txt
        create_requirements_file()

        # Итоговый отчет
        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)
        print(f"Исходный размер: {df.shape}")
        print(f"Финальный размер: {df_final.shape}")
        print(f"Train set: {train_df.shape}")
        print(f"Test set: {test_df.shape}")
        print(f"Сохраненные файлы:")
        print("  - processed_titanic_train.csv")
        print("  - processed_titanic_test.csv")
        print("  - requirements.txt")

        # Информация о проделанной работе
        print("\nВЫПОЛНЕННЫЕ ЭТАПЫ ПРЕДОБРАБОТКИ:")
        print("✅ 1. Загрузка данных и вывод на экран")
        print("✅ 2. Визуализация пропущенных значений")
        print("✅ 3. Анализ пропущенных значений")
        print("✅ 4. Заполнение пропущенных значений (медиана/мода)")
        print("✅ 5. Нормализация данных (MinMaxScaler + StandardScaler)")
        print("✅ 6. One-Hot Encoding категориальных данных")
        print("✅ 7. Разделение на обучающую и тестовую выборки")
        print("✅ 8. Сохранение обработанных данных")

        print("\n🎉 ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 60)

    except FileNotFoundError:
        print("❌ Ошибка: Файл 'titanic.csv' не найден!")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
