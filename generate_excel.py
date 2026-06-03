import pandas as pd
import numpy as np

def export_categorized_excel(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Ingeniería de Variables
    df['IMC'] = df['Peso'] / ((df['Talla'] / 100) ** 2)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Categorización
    df['Cat_Edad'] = pd.cut(df['Edad'], bins=[0, 19, 34, 100], labels=['Adolescente', 'Adulta Joven', 'Adulta Mayor'])
    df['Cat_Trimestre'] = pd.cut(df['Edad_Gestacional'], bins=[0, 13, 26, 50], labels=['1er Trimestre', '2do Trimestre', '3er Trimestre'])
    df['Cat_Altitud'] = pd.cut(df['Altitud_Loc'], bins=[0, 3850, 4000, 6000], labels=['Baja', 'Media', 'Alta'])
    df['Cat_Nutricion'] = pd.cut(df['IMC'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Bajo Peso', 'Normal', 'Sobrepeso', 'Obesidad'])
    df['Target'] = df['Dx_Anemia'].apply(lambda x: 'SANO' if x == 'NORMAL' else 'ANEMIA')
    
    # Exportar a Excel
    df.to_excel(output_file, index=False)
    print(f"Archivo guardado como: {output_file}")

if __name__ == "__main__":
    export_categorized_excel('Gestantes_Puno_Clean.csv', 'Gestantes_Puno_Categorizada.xlsx')
