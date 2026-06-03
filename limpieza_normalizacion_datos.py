import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    print(f"Leyendo archivo: {input_file}...")
    # Leer el CSV
    df = pd.read_csv(input_file)

    print("Iniciando limpieza de datos...")

    # 1. Limpieza de strings (quitar espacios en blanco y estandarizar a mayúsculas)
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        # No convertimos fechas a mayúsculas aún, solo columnas categóricas
        if col not in ['Fecha', 'FechaHemoglobina']:
            df[col] = df[col].str.upper()

    # 2. Conversión de Fechas
    date_cols = ['Fecha', 'FechaHemoglobina']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # 3. Limpieza de valores numéricos
    # Convertir a numérico y manejar errores (ej. espacios vacíos se vuelven NaN)
    numeric_cols = ['Edad', 'Edad_Gestacional', 'Peso', 'Talla', 'PPG', 
                    'Altitud_Dist', 'Altitud_Loc', 'Hematocrito', 'Hemoglobina', 'Hbc']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Tratamiento específico de valores inconsistentes
    # Hemoglobina 0 suele ser error de registro o falta de dato
    df.loc[df['Hemoglobina'] == 0, 'Hemoglobina'] = np.nan
    
    # 5. Normalización de UBIGEO (asegurar que sea string de 6 dígitos)
    if 'UBIGEO' in df.columns:
        df['UBIGEO'] = df['UBIGEO'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)

    # 6. Manejo de duplicados
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    final_rows = len(df)
    if initial_rows > final_rows:
        print(f"Se eliminaron {initial_rows - final_rows} filas duplicadas.")

    # Guardar el archivo limpio
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Limpieza completada. Archivo guardado como: {output_file}")
    
    # Resumen rápido
    print("\nResumen de la limpieza:")
    print(df.info())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

if __name__ == "__main__":
    input_csv = 'Gestantes_Puno.csv'
    output_csv = 'Gestantes_Puno_Clean.csv'
    clean_data(input_csv, output_csv)
