import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# -----------------------
# Configuración de página
# -----------------------
st.set_page_config(page_title="Detección de PPE", layout="wide")

# -----------------------
# Estilos simples (limpio)
# -----------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #2E86C1;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #F8F9F9;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Diccionario traducción
# -----------------------
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
# Header
# -----------------------
st.markdown("<h1 class='main-title'>🏭 Sistema de Detección de EPP</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Verificación automática de seguridad industrial</p>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Selector de entrada
# -----------------------
opcion = st.radio("Selecciona la fuente de imagen:", ["📁 Subir imagen", "📸 Usar cámara"])

foto = None

if opcion == "📁 Subir imagen":
    foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
else:
    foto = st.camera_input("Toma una foto")

# -----------------------
# Procesamiento
# -----------------------
if foto:
    imagen_original = Image.open(foto).convert("RGB")

    colA, colB = st.columns([1, 2])

    with colA:
        st.image(imagen_original, caption="Imagen cargada", use_container_width=True)

    img_np = np.array(imagen_original)

    with st.spinner("Analizando imagen..."):
        resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    with colB:
        st.subheader(f"👥 Personas detectadas: {len(personas)}")

    st.markdown("---")

    # -----------------------
    # Procesar personas
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):

        st.markdown(f"### 👤 Trabajador {i}")

        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np = np.array(persona_crop)

        resultados_ppe = modelo_ppe(persona_np)[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas = []
        datos_analitica = []

        for box in resultados_ppe.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_ppe.names[cls]

            if label_ingles == "person":
                continue

            label_espanol = TRADUCCION_CLASES.get(label_ingles, label_ingles.capitalize())
            conf = float(box.conf[0])

            etiquetas.append(label_espanol)
            datos_analitica.append({
                "Equipo": label_espanol,
                "Confianza": f"{conf*100:.2f}%"
            })

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
            draw.rectangle([x1o, y1o, x2o, y2o], outline="#00FF00", width=3)
            draw.text((x1o, max(0, y1o - 15)), f"{label_espanol}", fill="#00FF00")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, caption="Detección", use_container_width=True)

        with col2:
            st.markdown("#### 🚥 Estado de seguridad")

            requeridos = {"Casco", "Chaleco"}
            presentes = set(etiquetas)

            if requeridos.issubset(presentes):
                st.success("🟢 Acceso permitido")
            else:
                faltantes = requeridos - presentes
                st.error(f"🔴 Acceso denegado - Faltan: {', '.join(faltantes)}")

            st.markdown("#### 📊 Confianza de detección")

            if datos_analitica:
                for d in datos_analitica:
                    st.write(d["Equipo"])
                    valor = float(d["Confianza"].replace("%", "")) / 100
                    st.progress(valor)
            else:
                st.warning("No se detectó EPP")

        st.markdown("---")

# -----------------------
# Footer
# -----------------------
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 13px;'>
© Alfredo Diaz UNAB 2026
</p>
""", unsafe_allow_html=True)
