import os
# Evita el warning de Ultralytics en Streamlit Cloud
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# -----------------------
# Diccionario de Traducción
# -----------------------
# Traduce las clases originales del modelo al español
TRADUCCION_CLASES = {
    "boots": "Botas",
    "earmuffs": "Orejeras",
    "glasses": "Gafas",
    "gloves": "Guantes",
    "helmet": "Casco",
    "person": "Persona",
    "vest": "Chaleco"
}

# -----------------------
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# UI y Configuración
# -----------------------
st.set_page_config(page_title="Detección de PPE", layout="wide")

# Título e Instrucciones
st.title("🏭 Sistema Inteligente de Detección de EPP")
st.markdown("""
**📌 Instrucciones de uso:**
1. Sube una fotografía del trabajador.
2. El sistema detectará automáticamente a las personas en la imagen.
3. Se verificará si portan el Equipo de Protección Personal obligatorio (**Casco y Chaleco**).
4. El semáforo indicará si la persona está autorizada para ingresar a la planta.
""")
st.markdown("---")

foto = st.file_uploader("Sube una imagen para analizar", type=["jpg", "png", "jpeg"])

if foto:
    imagen_original = Image.open(foto).convert("RGB")
    
    with st.expander("Ver imagen original", expanded=False):
        st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    # Convertir a numpy (YOLO usa esto)
    img_np = np.array(imagen_original)

    # -----------------------
    # Detectar personas (YOLOv8n)
    # -----------------------
    resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # id 0 es 'persona' en YOLOv8 normal
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    st.subheader(f"👥 Se han detectado {len(personas)} persona(s) en la imagen")
    st.markdown("---")

    # -----------------------
    # Procesar cada persona con modelo PPE
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):
        st.markdown(f"### 👤 Trabajador {i}")
        
        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)

        # Predecir PPE en el recorte
        resultados_ppe = modelo_ppe(persona_np)[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []
        datos_analitica = [] # Para guardar datos para la tabla

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_ppe.names[cls]
            
            # Omitimos la etiqueta "person" si el modelo PPE la vuelve a detectar
            if label_ingles == "person":
                continue

            # Traducir al español usando el diccionario
            label_espanol = TRADUCCION_CLASES.get(label_ingles, label_ingles.capitalize())
            conf = float(box.conf[0])
            
            etiquetas.append(label_espanol)
            datos_analitica.append({"Equipo Detectado": label_espanol, "Confianza": f"{conf*100:.2f}%"})

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])

            # Dibujar caja de predicción con el nombre en español
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#00FF00", width=3)
            draw.text((x1o, max(0, y1o - 15)), f"{label_espanol} {conf:.2f}", fill="#00FF00")

        # Layout en 2 columnas para mostrar foto y resultados lado a lado
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption=f"Recorte Trabajador {i}", use_container_width=True)

        with col2:
            # -----------------------
            # Semáforo y Validación PPE
            # -----------------------
            st.markdown("#### 🚥 Control de Acceso a Planta")
            
            # Requisitos OBLIGATORIOS (solo Casco y Chaleco)
            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("🟢 **ACCESO PERMITIDO:** El trabajador cumple con el equipo de seguridad obligatorio (Casco y Chaleco).")
            else:
                faltantes = requeridos - presentes
                st.error(f"🔴 **ACCESO DENEGADO:** Riesgo crítico de seguridad. Faltan los siguientes equipos obligatorios: **{', '.join(faltantes)}**")

            # -----------------------
            # Analítica Predictiva
            # -----------------------
            st.markdown("#### 📊 Analítica del Modelo")
            if datos_analitica:
                df_analitica = pd.DataFrame(datos_analitica)
                # Ordenar por porcentaje de confianza
                df_analitica = df_analitica.sort_values(by="Confianza", ascending=False).reset_index(drop=True)
                st.dataframe(df_analitica, use_container_width=True)
            else:
                st.warning("⚠️ El modelo no detectó ningún equipo de protección en este trabajador.")
        
        st.markdown("---")

# -----------------------
# Pie de página
# -----------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #888888; font-size: 14px;'>"
    "© Alfredo Diaz UNAB 2026. Todos los derechos reservados."
    "</p>", 
    unsafe_allow_html=True
)
