# ==========================================
# PROYECTO IA: PREDICCIÓN DE INGRESO UNIVERSITARIO
# Autores: Sebas Galdames, Nicolás Rodríguez, Constanza Toro
# ==========================================

import os
import glob
import re
import unicodedata
import difflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# ------------------------------------------
# PARTE 1: CARGA, CONSOLIDACIÓN Y CRUZAMIENTO DE DATOS
# ------------------------------------------
drive.mount('/content/drive')

try:
    # Punto clave: Ubicación automática de la carpeta raíz del proyecto en Google Drive
    carpeta_base = glob.glob('/content/drive/MyDrive/*admisi*')[0]
    print(f"📁 Carpeta encontrada: {carpeta_base}")
except IndexError:
    raise FileNotFoundError("⚠️ No se encontró la carpeta de admisión en tu Google Drive. Verifica el nombre.")

anios = ['2024', '2025', '2026']
lista_dataframes = []

for anio in anios:
    ruta_anio = os.path.join(carpeta_base, anio)
    print(f"\n📂 Procesando carpeta del año {anio}...")

    archivos_csv = glob.glob(os.path.join(ruta_anio, '*.csv'))
    ruta_consolidado = [f for f in archivos_csv if 'Consolidado' in os.path.basename(f)][0]
    ruta_mineduc = [f for f in archivos_csv if 'Consolidado' not in os.path.basename(f)][0]

    # Leer consolidado DEMRE (Contiene postulaciones y resultados reales)
    df_demre = pd.read_csv(ruta_consolidado, sep=';')

    # Detección automática del separador del archivo MINEDUC para evitar errores de lectura
    with open(ruta_mineduc, 'r', encoding='latin1') as f:
        primera_linea = f.readline()
    sep_mineduc = ';' if ';' in primera_linea else (',' if ',' in primera_linea else '\t')

    cols_reales = pd.read_csv(ruta_mineduc, sep=sep_mineduc, encoding='latin1', nrows=0).columns.tolist()
    col_inst = [c for c in cols_reales if 'NOMB_INST' in c.upper()][0]
    col_tipo = [c for c in cols_reales if 'TIPO_INST_3' in c.upper()][0]

    df_mineduc = pd.read_csv(
        ruta_mineduc,
        sep=sep_mineduc,
        usecols=[col_inst, col_tipo],
        encoding='latin1'
    ).drop_duplicates(subset=[col_inst])

    # Cruce de fuentes (DEMRE + MINEDUC) utilizando llaves normalizadas por nombre de universidad
    df_demre['uni_norm'] = df_demre['universidad'].str.upper().str.strip()
    df_mineduc['uni_norm'] = df_mineduc[col_inst].astype(str).str.upper().str.strip()
    df_mineduc['TIPO_INST_3'] = df_mineduc[col_tipo].astype(str).str.strip()

    df_cruzado = pd.merge(
        df_demre,
        df_mineduc[['uni_norm', 'TIPO_INST_3']],
        on='uni_norm',
        how='left'
    ).drop(columns=['uni_norm'])

    df_cruzado['TIPO_INST_3'] = df_cruzado['TIPO_INST_3'].fillna('Universidades CRUCH')
    lista_dataframes.append(df_cruzado)
    print(f"   ✅ Año {anio} listo: {len(df_cruzado)} registros cruzados.")

# Consolidación final de los 3 años de datos históricos
df_final = pd.concat(lista_dataframes, ignore_index=True)

# Muestreo estratificado para balancear el dataset y agilizar el entrenamiento en la nube
df_modelo = df_final.groupby('anio', group_keys=False).apply(
    lambda x: x.sample(n=min(30000, len(x)), random_state=42)
).reset_index(drop=True)

print(f"\n🎉 ¡DATASET MAESTRO LISTO! Total filas para entrenar: {len(df_modelo)}")


# ------------------------------------------
# PARTE 2: ENTRENAMIENTO Y COMPARACIÓN DE 3 ALGORITMOS
# ------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

features = ['anio', 'region', 'TIPO_INST_3', 'universidad', 'carrera', 'orden_preferencia', 'puntaje_ponderado']
X = df_modelo[features]
y = df_modelo['ingresado']

# División del dataset: 80% entrenamiento, 20% prueba con estratificación por la clase objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = ['region', 'TIPO_INST_3', 'universidad', 'carrera']
num_cols = ['anio', 'orden_preferencia', 'puntaje_ponderado']

# Preprocesador para transformar variables categóricas mediante One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

# Definición de los 3 modelos solicitados para competencia técnica
modelos = {
    "Naive Bayes": GaussianNB(),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
}

resultados = []
pipelines_entrenados = {}

print("⏳ Entrenando y evaluando los 3 modelos predictivos...")
for nombre, clf in modelos.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    pipelines_entrenados[nombre] = pipe

    y_pred = pipe.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4)
    })

df_metricas = pd.DataFrame(resultados)
print("\n📊 TABLA COMPARATIVA DE MÉTRICAS:")
print(df_metricas.to_string(index=False))

# Selección automática del modelo ganador basado estrictamente en el mayor F1-Score
mejor_nombre = df_metricas.sort_values(by="F1-Score", ascending=False).iloc[0]["Modelo"]
modelo_ganador = pipelines_entrenados[mejor_nombre]
print(f"\n🏆 Modelo seleccionado para el Chatbot: {mejor_nombre}")


# ------------------------------------------
# PARTE 3: INTERFAZ DEL CHATBOT CON GRADIO (10 PASOS Y EXCEPCIONES)
# ------------------------------------------
import gradio as gr

def limpiar_texto(texto):                                                                # Normaliza cadenas de texto eliminando tildes y convirtiendo a mayúsculas para estandarizar búsquedas
    texto = str(texto).upper().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

df_final['reg_clean'] = df_final['region'].apply(limpiar_texto)
df_final['uni_clean'] = df_final['universidad'].apply(limpiar_texto)
df_final['car_clean'] = df_final['carrera'].apply(limpiar_texto)

cortes_reales = df_final[df_final['ingresado'] == 1].groupby(['region', 'universidad', 'carrera'])['puntaje_ponderado'].min().reset_index()
cortes_reales.rename(columns={'puntaje_ponderado': 'puntaje_corte'}, inplace=True)

tabla_maestra = df_final[['region', 'universidad', 'carrera', 'reg_clean', 'uni_clean', 'car_clean', 'TIPO_INST_3', '%_NOTAS', '%_Ranking', '%_LENG', '%_MATE1']].drop_duplicates(subset=['region', 'universidad', 'carrera'])
tabla_maestra = pd.merge(tabla_maestra, cortes_reales, on=['region', 'universidad', 'carrera'], how='left')

modelo_chatbot = pipelines_entrenados[mejor_nombre]

def buscar_coincidencia(texto_usuario, df_subset, col_clean, col_oficial, limite_sugerencias=5): # Realiza búsqueda inteligente (exacta, parcial y difusa/fuzzy matching) sobre los catálogos institucionales
    busq = limpiar_texto(texto_usuario)
    if len(busq) < 3:
        return None, df_subset[col_oficial].unique()[:limite_sugerencias].tolist()

    exacta = df_subset[df_subset[col_clean] == busq]
    if not exacta.empty:
        return exacta.iloc[0][col_oficial], []

    parcial = df_subset[df_subset[col_clean].str.contains(re.escape(busq), na=False)]
    unicos_parcial = parcial[col_oficial].unique().tolist()
    if len(unicos_parcial) == 1:
        return unicos_parcial[0], []
    elif len(unicos_parcial) > 1:
        return None, unicos_parcial[:limite_sugerencias]

    opciones_clean = df_subset[col_clean].unique().tolist()
    mapa_oficial = dict(zip(df_subset[col_clean], df_subset[col_oficial]))
    similares_clean = difflib.get_close_matches(busq, opciones_clean, n=limite_sugerencias, cutoff=0.45)
    sugerencias = [mapa_oficial[c] for c in similares_clean]

    if not sugerencias:
        sugerencias = df_subset[col_oficial].unique()[:limite_sugerencias].tolist()

    return None, sugerencias

with gr.Blocks(title="Simulador Admisión IA (10 Pasos)") as demo:
    gr.Markdown("## 🎓 Asistente Vocacional de Admisión Universitaria (DEMRE + MINEDUC)")
    gr.Markdown("*Herramienta referencial basada en registros históricos reales. No vinculante.*")

    chatbot = gr.Chatbot(value=[
        {"role": "assistant", "content": "¡Hola! Soy tu asistente de admisión universitaria.\n\n**Paso 1/10:** Para adaptar la orientación, indícame tu situación actual: ¿Estás cursando **4° Medio** o ya eres **Egresado**?"}
    ])
    msg = gr.Textbox(placeholder="Escribe tu respuesta aquí y presiona Enter...", label="Tu mensaje")
    clear = gr.Button("🗑️ Reiniciar Conversación")
    estado = gr.State({"paso": 1})

    def procesar_mensaje(mensaje_usuario, historial, datos):                             # Gestiona la máquina de estados secuencial de los 10 pasos del chatbot e invoca la inferencia del modelo
        if not mensaje_usuario.strip():
            return "", historial, datos

        historial.append({"role": "user", "content": mensaje_usuario})
        paso = datos.get("paso", 1)
        texto_clean = limpiar_texto(mensaje_usuario)

        try:
            # PASO 1: Situación académica
            if paso == 1:
                if "4" in texto_clean or "CUARTO" in texto_clean or "MEDIO" in texto_clean:
                    datos["situacion"] = "4° Medio"
                    nota_situacion = "Como estás en **4° Medio**, recuerda que tu NEM y Ranking se cerrarán al terminar el año escolar."
                elif "EGRESADO" in texto_clean or "SALI" in texto_clean or "ANTERIOR" in texto_clean:
                    datos["situacion"] = "Egresado"
                    nota_situacion = "Como **Egresado**, ya cuentas con tu NEM y Ranking definitivos."
                else:
                    historial.append({"role": "assistant", "content": "⚠️ Por favor indica si estás en **4° Medio** o eres **Egresado**."})
                    return "", historial, datos

                datos["paso"] = 2
                respuesta = f"{nota_situacion}\n\n**Paso 2/10:** ¿En qué **Región** deseas estudiar? (Ej: Metropolitana, Valparaíso)"

            # PASO 2: Selección de Región
            elif paso == 2:
                reg_encontrada, sugerencias = buscar_coincidencia(mensaje_usuario, tabla_maestra, 'reg_clean', 'region', limite_sugerencias=5)
                if not reg_encontrada:
                    lista_sug = "\n".join([f"• {s}" for s in sugerencias])
                    respuesta = f"⚠️ No identifiqué una región exacta para **'{mensaje_usuario}'**.\nElige una opción:\n{lista_sug}"
                else:
                    datos["region"] = reg_encontrada
                    datos["paso"] = 3
                    respuesta = (f"✅ Región: **{reg_encontrada}**.\n\n"
                                 f"**Paso 3/10:** ¿Qué tipo de institución prefieres?\n"
                                 f"1. **CRUCH**\n2. **Privadas**\n3. **Todas**")

            # PASO 3: Filtro Tipo de Institución
            elif paso == 3:
                df_reg = tabla_maestra[tabla_maestra['region'] == datos['region']]
                if "1" in texto_clean or "CRUCH" in texto_clean:
                    df_filtrado = df_reg[df_reg['TIPO_INST_3'].str.contains('CRUCH', case=False, na=False)]
                    datos["filtro_inst"] = "Universidades CRUCH"
                elif "2" in texto_clean or "PRIVADA" in texto_clean:
                    df_filtrado = df_reg[~df_reg['TIPO_INST_3'].str.contains('CRUCH', case=False, na=False)]
                    datos["filtro_inst"] = "Universidades Privadas"
                else:
                    df_filtrado = df_reg
                    datos["filtro_inst"] = "Todas las Universidades"

                if df_filtrado.empty:
                    df_filtrado = df_reg

                unis_disp = "\n".join([f"• {u}" for u in df_filtrado['universidad'].unique()[:6]])
                datos["paso"] = 4
                respuesta = f"✅ Filtro: **{datos['filtro_inst']}**.\n\n**Paso 4/10:** Escribe el nombre de la **Universidad**:\n{unis_disp}"

            # PASO 4: Selección de Universidad
            elif paso == 4:
                df_reg = tabla_maestra[tabla_maestra['region'] == datos['region']]
                uni_encontrada, sugerencias = buscar_coincidencia(mensaje_usuario, df_reg, 'uni_clean', 'universidad', limite_sugerencias=6)
                if not uni_encontrada:
                    lista_sug = "\n".join([f"• {s}" for s in sugerencias])
                    respuesta = f"❌ Universidad no encontrada.\n🔍 **Sugerencias:**\n{lista_sug}"
                else:
                    datos["universidad"] = uni_encontrada
                    datos["paso"] = 5
                    respuesta = f"✅ Universidad: **{uni_encontrada}**.\n\n**Paso 5/10:** ¿A qué **Carrera** deseas postular?"

            # PASO 5: Selección de Carrera
            elif paso == 5:
                df_uni = tabla_maestra[
                    (tabla_maestra['region'] == datos['region']) &
                    (tabla_maestra['universidad'] == datos['universidad'])
                ]
                car_encontrada, sugerencias = buscar_coincidencia(mensaje_usuario, df_uni, 'car_clean', 'carrera', limite_sugerencias=8)
                if not car_encontrada:
                    lista_sug = "\n".join([f"• {s}" for s in sugerencias])
                    respuesta = f"❌ Carrera no encontrada.\n🔍 **Carreras similares:**\n{lista_sug}"
                else:
                    datos["carrera"] = car_encontrada
                    datos["paso"] = 6
                    respuesta = f"✅ Carrera: **{car_encontrada}**.\n\n**Paso 6/10:** Orden de preferencia (**1 al 20**):"

            # PASO 6: Orden de Preferencia
            elif paso == 6:
                nums = re.findall(r'\d+', mensaje_usuario)
                if not nums or not (1 <= int(nums[0]) <= 20):
                    respuesta = "⚠️ Ingresa un número válido entre **1 y 20**."
                else:
                    datos["orden_pref"] = int(nums[0])
                    datos["paso"] = 7
                    respuesta = f"✅ Preferencia N° **{datos['orden_pref']}**.\n\n**Paso 7/10:** Ingresa **NEM, Ranking** separados por coma (Ej: `710, 740`):"

            # PASO 7: Puntajes NEM y Ranking
            elif paso == 7:
                nums = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', mensaje_usuario)]
                if len(nums) != 2 or any(p < 100 or p > 1000 for p in nums):
                    respuesta = "⚠️ Ingresa exactamente 2 puntajes válidos entre 100 y 1000 (Ej: `710, 740`)."
                else:
                    datos["nem"], datos["ranking"] = nums[0], nums[1]
                    datos["paso"] = 8
                    respuesta = f"✅ Guardado.\n\n**Paso 8/10:** Ingresa **Lectora, Matemática M1** separados por coma (Ej: `680, 750`):"

            # PASO 8: Pruebas Obligatorias
            elif paso == 8:
                nums = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', mensaje_usuario)]
                if len(nums) != 2 or any(p < 100 or p > 1000 for p in nums):
                    respuesta = "⚠️ Ingresa exactamente 2 puntajes válidos entre 100 y 1000 (Ej: `680, 750`)."
                else:
                    datos["lectora"], datos["m1"] = nums[0], nums[1]
                    datos["paso"] = 9
                    respuesta = f"✅ Guardado.\n\n**Paso 9/10:** ¿Rendiste **Prueba Electiva / M2**? Escribe el puntaje o `No`:"

            # PASO 9: Electiva + Ejecución del Modelo Predictivo
            elif paso == 9:
                if "NO" in texto_clean:
                    datos["electiva"] = (datos["m1"] + datos["lectora"]) / 2.0
                    nota_elec = "Sin electiva (estimada proporcional)."
                else:
                    nums = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', mensaje_usuario)]
                    if not nums or not (100 <= nums[0] <= 1000):
                        historial.append({"role": "assistant", "content": "⚠️ Ingresa un puntaje válido entre 100 y 1000 o escribe **No**."})
                        return "", historial, datos
                    datos["electiva"] = nums[0]
                    nota_elec = f"Electiva incorporada ({datos['electiva']} pts)."

                fila = tabla_maestra[
                    (tabla_maestra['region'] == datos['region']) &
                    (tabla_maestra['universidad'] == datos['universidad']) &
                    (tabla_maestra['carrera'] == datos['carrera'])
                ].iloc[0]

                p_nem = (fila['%_NOTAS'] / 100.0) * datos["nem"]
                p_rk = (fila['%_Ranking'] / 100.0) * datos["ranking"]
                p_lec = (fila['%_LENG'] / 100.0) * datos["lectora"]
                p_m1 = (fila['%_MATE1'] / 100.0) * datos["m1"]
                resto_pct = max(0, 100 - (fila['%_NOTAS'] + fila['%_Ranking'] + fila['%_LENG'] + fila['%_MATE1'])) / 100.0
                p_elec = resto_pct * datos["electiva"]
                ponderado = round(p_nem + p_rk + p_lec + p_m1 + p_elec, 2)

                fila_pred = pd.DataFrame({
                    'anio': [2026],
                    'region': [fila['region']],
                    'TIPO_INST_3': [fila['TIPO_INST_3']],
                    'universidad': [fila['universidad']],
                    'carrera': [fila['carrera']],
                    'orden_preferencia': [datos["orden_pref"]],
                    'puntaje_ponderado': [ponderado]
                })

                # Punto clave: Ejecución de la inferencia matemática usando el modelo ganador
                prob = float(modelo_chatbot.predict_proba(fila_pred)[0][1]) * 100.0
                corte = fila['puntaje_corte']
                info_corte = f"{corte:.1f} pts" if pd.notna(corte) else "Sin registro"

                if prob >= 50:
                    veredicto = "🟢 ALTA PROBABILIDAD"
                    consejo = "Tu puntaje supera el umbral histórico. Ubica esta carrera en tus primeras preferencias."
                else:
                    veredicto = "🟡 BAJA PROBABILIDAD"
                    consejo = "Existe riesgo de lista de espera. Te sugerimos tener opciones de respaldo."

                datos["paso"] = 10
                respuesta = (f"📊 **DIAGNÓSTICO FINAL (Paso 10/10)**\n"
                             f"• **Institución:** {fila['universidad']}\n"
                             f"• **Carrera:** {fila['carrera']}\n"
                             f"• **Puntaje Ponderado:** **{ponderado} pts** (Corte mín: {info_corte})\n"
                             f"• **Resultado IA ({mejor_nombre}):** {veredicto} (**{prob:.1f}%**)\n"
                             f"• **Recomendación:** {consejo}\n\n"
                             f"🔄 **¿Deseas simular otra carrera con tus mismos puntajes?** Responde **Sí** o **No**.")

            # PASO 10: Continuidad o Cierre
            elif paso == 10:
                if "SI" in texto_clean or "SÍ" in texto_clean:
                    datos["paso"] = 2
                    respuesta = "¡Perfecto! Conservaremos tus puntajes. Escribe la **Región** para tu nueva simulación:"
                else:
                    datos = {"paso": 1}
                    respuesta = "✅ ¡Gracias por utilizar el Simulador! Indica si estás en **4° Medio** o eres **Egresado** para reiniciar."

        except Exception as e:                                                            # Manejo global de excepciones para evitar que el bot colapse ante entradas inesperadas
            respuesta = f"❌ Ocurrió un error inesperado en el flujo: {str(e)}. Reiniciando al Paso 1."
            datos = {"paso": 1}

        historial.append({"role": "assistant", "content": respuesta})
        return "", historial, datos

    def reiniciar():                                                                     # Restablece por completo el historial de chat y el estado inicial al Paso 1
        return [{"role": "assistant", "content": "¡Hola! Conversación reiniciada.\n\n**Paso 1/10:** ¿Estás cursando **4° Medio** o ya eres **Egresado**?"}], {"paso": 1}

    msg.submit(procesar_mensaje, [msg, chatbot, estado], [msg, chatbot, estado])
    clear.click(reiniciar, None, [chatbot, estado])

demo.launch(share=True)
