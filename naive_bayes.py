import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def run_r_style_naive_bayes(file_path):
    print("="*70)
    print("ANÁLISIS DE NAIVE BAYES - ESTILO METODOLÓGICO R (DISCRETIZADO)")
    print("="*70)

    # 1. Carga de datos
    df = pd.read_csv(file_path)
    
    # 2. Ingeniería de Variables y Discretización (Factores en R)
    print("\n[1] Discretizando variables continuas en categorías...")
    
    # IMC
    df['IMC'] = df['Peso'] / ((df['Talla'] / 100) ** 2)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['IMC'])
    
    # Rango de Edad
    df['Cat_Edad'] = pd.cut(df['Edad'], bins=[0, 19, 34, 100], labels=['Adolescente', 'Adulta Joven', 'Adulta Mayor'])
    
    # Trimestre Gestacional
    df['Cat_Trimestre'] = pd.cut(df['Edad_Gestacional'], bins=[0, 13, 26, 50], labels=['1er Trimestre', '2do Trimestre', '3er Trimestre'])
    
    # Nivel de Altitud (Puno es alto, definimos rangos locales)
    df['Cat_Altitud'] = pd.cut(df['Altitud_Loc'], bins=[0, 3850, 4000, 6000], labels=['Baja (<3850)', 'Media (3850-4000)', 'Alta (>4000)'])
    
    # Estado Nutricional (IMC)
    df['Cat_Nutricion'] = pd.cut(df['IMC'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Bajo Peso', 'Normal', 'Sobrepeso', 'Obesidad'])

    # Variable Objetivo: Anemia (Binaria)
    df['Target'] = df['Dx_Anemia'].apply(lambda x: 'SANO' if x == 'NORMAL' else 'ANEMIA')

    # 3. Selección de Factores
    features = ['Cat_Edad', 'Cat_Trimestre', 'Cat_Altitud', 'Cat_Nutricion']
    df_model = df[features + ['Target']].dropna()

    print("\n[2] Probabilidades A Priori (Distribución de clases):")
    prior = df_model['Target'].value_counts(normalize=True)
    print(prior)

    # 4. Codificación para Scikit-Learn (Labels)
    le_dict = {}
    X = pd.DataFrame()
    for col in features:
        le = LabelEncoder()
        X[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(df_model['Target'])

    # 5. División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Entrenamiento con CategoricalNB (Ideal para factores)
    print("\n[3] Entrenando modelo Categorical Naive Bayes...")
    # min_categories asegura que el modelo conozca todas las categorías posibles
    model = CategoricalNB()
    model.fit(X_train, y_train)

    # 7. Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print(f"RESULTADOS DEL MODELO (Accuracy: {acc:.2%})")
    print("="*70)

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    target_labels = le_target.classes_
    
    # Formateo manual para claridad
    print(f"{'':<15} Predicho: {target_labels[0]:<10} Predicho: {target_labels[1]:<10}")
    print(f"Real: {target_labels[0]:<10} {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"Real: {target_labels[1]:<10} {cm[1][0]:<15} {cm[1][1]:<15}")

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_labels))

    # --- NUEVO: GRÁFICO DE MAPA DE CALOR ---
    print("\n[4] Generando Mapa de Calor de la Matriz de Confusión...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=['Pred: ANEMIA', 'Pred: SANO'], 
                yticklabels=['Real: ANEMIA', 'Real: SANO'])
    plt.title('Mapa de Calor - Matriz de Confusión (Naive Bayes R-Style)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    output_image = 'heatmap_confusion_matrix_r_style.png'
    plt.savefig(output_image)
    print(f"Gráfico guardado como: {output_image}")

    # 8. Ejemplo de inferencia (Probabilidades Condicionales)
    print("\n[5] Ejemplo de Análisis Probabilístico:")
    # Tomamos un caso: Adolescente, 3er Trimestre, Altitud Alta, Bajo Peso
    example_case = [[0, 2, 0, 0]] # Basado en los labels codificados
    probs = model.predict_proba(example_case)[0]
    print(f"Para una gestante con perfil: Adolescente, 3er Trimestre, Altura >4000m, Bajo Peso:")
    print(f" -> Probabilidad de ANEMIA: {probs[0]:.2%}")
    print(f" -> Probabilidad de SANO: {probs[1]:.2%}")

if __name__ == "__main__":
    input_file = 'Gestantes_Puno_Clean.csv'
    run_r_style_naive_bayes(input_file)

if __name__ == "__main__":
    input_file = 'Gestantes_Puno_Clean.csv'
    run_r_style_naive_bayes(input_file)
