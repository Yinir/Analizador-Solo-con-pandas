import pandas as pd
import os
from tqdm import tqdm
import chardet
import numpy as np

def detectar_codificacion(ruta_archivo, muestra_bytes=10000):
    """Detecta la codificación del archivo de manera más confiable usando chardet."""
    with open(ruta_archivo, 'rb') as f:
        resultado = chardet.detect(f.read(muestra_bytes))
    return resultado['encoding'] 

def inferir_tipo_oracle(serie):
    """Infiere el tipo de dato Oracle para una columna de pandas."""
    # Convertir a string si hay valores no string
    if serie.dtype == object:
        serie = serie.astype(str)
    
    # Intentar convertir a datetime
    try:
        # Probar con infer_datetime_format para mayor eficiencia
        datetime_serie = pd.to_datetime(serie, errors='coerce')
        if not datetime_serie.isna().all():
            # Verificar si tiene componente de hora
            if any(datetime_serie.dropna().apply(lambda x: x.time() != pd.Timestamp('00:00:00').time())):
                return "TIMESTAMP"
            else:
                return "DATE"
    except:
        pass
    
    # Intentar convertir a numérico
    try:
        numeric_serie = pd.to_numeric(serie, errors='coerce')
        if not numeric_serie.isna().all():
            # Verificar si es entero
            if (numeric_serie.dropna() % 1 == 0).all():
                max_val = numeric_serie.max()
                min_val = numeric_serie.min()
                # Determinar precisión necesaria
                if max_val < 10**10 and min_val > -10**10:
                    return "NUMBER(10)"
                else:
                    return "NUMBER(20)"
            else:
                return "NUMBER(15,2)"
    except:
        pass
    
    # Si no es fecha ni número, es texto
    max_len = serie.str.len().max()
    return f"VARCHAR2({int(max_len)})"

def analizar_archivo_pandas(ruta_archivo):
    """Analiza completamente el archivo usando pandas."""
    print("\nIniciando análisis con pandas...")
    
    # Paso 1: Detectar codificación
    encoding = detectar_codificacion(ruta_archivo)
    print(f"Codificación detectada: {encoding}")
    
    # Paso 2: Determinar si el archivo es grande
    file_size = os.path.getsize(ruta_archivo)
    es_grande = file_size > 100 * 1024 * 1024  # > 100MB
    
    # Paso 3: Leer el archivo con estrategia adecuada
    if es_grande:
        print("Archivo grande detectado. Usando muestreo y lectura por chunks...")
        
        # Primero leer solo los encabezados (sin usar with)
        reader = pd.read_csv(ruta_archivo, sep='|', quotechar='"', 
                           encoding=encoding, nrows=0)
        columnas = reader.columns.tolist()
        
        # Inicializar estructuras para análisis
        tipos_finales = {col: set() for col in columnas}
        max_longitudes = {col: 0 for col in columnas}
        total_filas = 0
        
        # Leer por chunks
        chunksize = 10**5  # 100,000 filas por chunk
        for chunk in tqdm(pd.read_csv(ruta_archivo, sep='|', quotechar='"', 
                                     encoding=encoding, chunksize=chunksize,
                                     low_memory=False, on_bad_lines='warn'),
                          desc="Procesando chunks"):
            total_filas += len(chunk)
            
            for col in columnas:
                # Limpiar nombre de columna
                col_limpia = col.strip().strip('"')
                
                # Procesar cada columna
                serie = chunk[col].astype(str)
                
                # Actualizar longitud máxima
                current_max = serie.str.len().max()
                if current_max > max_longitudes[col_limpia]:
                    max_longitudes[col_limpia] = current_max
                
                # Inferir tipo y agregar al conjunto
                tipo = inferir_tipo_oracle(serie)
                tipos_finales[col_limpia].add(tipo)
        
        print(f"\nTotal de filas procesadas: {total_filas:,}")
    else:
        print("Archivo pequeño detectado. Leyendo todo en memoria...")
        df = pd.read_csv(ruta_archivo, sep='|', quotechar='"', 
                         encoding=encoding, low_memory=False, 
                         on_bad_lines='warn')
        
        # Inicializar estructuras
        columnas = df.columns.tolist()
        tipos_finales = {}
        max_longitudes = {}
        
        for col in columnas:
            # Limpiar nombre de columna
            col_limpia = col.strip().strip('"')
            
            # Procesar cada columna
            serie = df[col].astype(str)
            
            # Calcular longitud máxima
            max_longitudes[col_limpia] = serie.str.len().max()
            
            # Inferir tipo
            tipos_finales[col_limpia] = {inferir_tipo_oracle(serie)}
    
    # Paso 4: Consolidar tipos (para chunks donde hay múltiples tipos)
    tipos_consolidados = {}
    for col, tipos in tipos_finales.items():
        if len(tipos) == 1:
            tipos_consolidados[col] = tipos.pop()
        else:
            # Priorizar tipos más específicos
            if 'TIMESTAMP' in tipos:
                tipos_consolidados[col] = 'TIMESTAMP'
            elif 'DATE' in tipos:
                tipos_consolidados[col] = 'DATE'
            elif any(t.startswith('NUMBER') for t in tipos):
                # Tomar el tipo NUMBER más abarcador
                has_decimal = any('NUMBER(15,2)' in t for t in tipos)
                tipos_consolidados[col] = 'NUMBER(15,2)' if has_decimal else 'NUMBER(10)'
            else:
                # Usar el VARCHAR más grande
                max_len = max(int(t.split('(')[1].split(')')[0]) for t in tipos if t.startswith('VARCHAR'))
                tipos_consolidados[col] = f'VARCHAR2({max_len})'
    
    return max_longitudes, tipos_consolidados

def generar_estructura_oracle(nombre_tabla, max_longitudes, tipos_dato):
    """Genera la estructura Oracle optimizada."""
    estructura = [f"-- ESTRUCTURA ORACLE GENERADA CON PANDAS\nCREATE TABLE {nombre_tabla.upper()} ("]
    
    for campo in sorted(max_longitudes.keys()):
        tipo = tipos_dato[campo]
        nombre_col = campo[:30].upper()  # Oracle tiene límite de 30 caracteres
        estructura.append(f"    {nombre_col} {tipo},")
    
    # Eliminar última coma y cerrar tabla
    estructura[-1] = estructura[-1].rstrip(',')
    estructura.append(");")
    
    return "\n".join(estructura)

if __name__ == "__main__":
    print("ANALIZADOR DE ARCHIVOS PARA ORACLE (VERSIÓN PANDAS)")
    print("=" * 70)
    
    ruta_archivo = input("Introduce la ruta completa del archivo TXT/CSV: ").strip('"')
    nombre_tabla = input("Nombre de la tabla Oracle (dejar vacío para 'usuarios'): ").strip() or "usuarios"
    
    if not os.path.exists(ruta_archivo):
        print(f"\nError: No se encontró el archivo en {ruta_archivo}")
    else:
        try:
            max_longitudes, tipos_dato = analizar_archivo_pandas(ruta_archivo)
            
            if max_longitudes and tipos_dato:
                print("\nRESULTADOS DEL ANÁLISIS:")
                print("-" * 70)
                for campo in sorted(max_longitudes.keys()):
                    print(f"{campo.ljust(25)}: {max_longitudes[campo]} chars | Tipo: {tipos_dato[campo]}")
                
                estructura = generar_estructura_oracle(nombre_tabla, max_longitudes, tipos_dato)
                print("\n" + "=" * 70)
                print(estructura)
                
                # Guardar en archivo
                nombre_archivo_salida = f"estructura_oracle_{nombre_tabla}.sql"
                with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
                    f.write(estructura)
                print(f"\nEstructura guardada en '{nombre_archivo_salida}'")
            
        except Exception as e:
            print(f"\nError: {str(e)}")