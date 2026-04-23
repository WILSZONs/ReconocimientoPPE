import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Configuración de entorno
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -----------------------
# CONFIGURACIÓN Y CONSTANTES
# -----------------------
TRADUCCION_CLASES = {
    "boots": "Botas", "earmuffs": "Orejeras", "glasses": "Gafas",
    "gloves": "Guantes", "helmet": "Casco", "person": "Persona", "vest": "Chaleco"
}
EPP_REQUERIDO = {"Casco", "Chaleco"}

# -----------------------
# LÓGICA DE NEGOCIO (PROCESAMIENTO)
# -----------------------

@st.cache_resource
def load_models():
    """Carga los modelos de YOLO una sola vez."""
    return YOLO("yolov8n.pt"), YOLO("best.pt")

def get_person_boxes(results):
    """Extrae coordenadas de personas detectadas."""
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Clase 0 es persona en COCO
            boxes.append(map(int, box.xyxy[0]))
    return boxes

def process_ppe_detection(image_crop, model_ppe):
    """Analiza el EPP en un recorte de imagen y devuelve la imagen dibujada y datos."""
    results = model_ppe(np.array(image_crop))[0]
    draw = ImageDraw.Draw(image_crop)
    detecciones = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label_en = model_ppe.names[cls]
        
        if label_en == "person": continue

        label_es = TRADUCCION_CLASES.get(label_en, label_en.capitalize())
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dibujo estético
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
        draw.text((x1, max(0, y1 - 20)), f"{label_es} {conf:.2%}", fill="#00FF00")
        
        detecciones.append({"label": label_es, "conf": conf})
        
    return image_crop, detecciones

# -----------------------
# INTERFAZ DE USUARIO (UI)
# -----------------------

def inject_custom_css():
    st.markdown("""
        <style>
        .main-title { text-align: center; color: #2E86C1; margin-bottom: 0px; }
        .subtitle { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
        hr { margin: 1rem 0; }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Seguridad Industrial IA", layout="wide")
    inject_custom_css()
    
    modelo_personas, modelo_ppe = load_models()

    # Header
    st.markdown("<h1 class='main-title'>🏭 Sistema de Detección de EPP</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Monitoreo de Seguridad en Tiempo Real</p>", unsafe_allow_html=True)

    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración")
        fuente = st.radio("Fuente de entrada:", ["📁 Archivo local", "📸 Cámara"])
        foto = st.file_uploader("Subir imagen", type=["jpg", "png"]) if fuente == "📁 Archivo local" else st.camera_input("Capturar")

    if not foto:
        st.info("Esperando entrada de imagen...")
        return

    # Procesamiento Principal
    img_original = Image.open(foto).convert("RGB")
    
    with st.spinner("Detectando personal..."):
        res_personas = modelo_personas(np.array(img_original))[0]
        coords_personas = get_person_boxes(res_personas)

    st.success(f"Detección finalizada: {len(coords_personas)} personas encontradas.")
    
    # Grid de resultados
    for i, coords in enumerate(coords_personas, 1):
        with st.container():
            col_img, col_info = st.columns([1, 1.5])
            
            # Recorte y detección de EPP
            crop = img_original.crop(coords)
            img_detectada, lista_epp = process_ppe_detection(crop, modelo_ppe)
            
            with col_img:
                st.image(img_detectada, caption=f"Trabajador {i}", use_container_width=True)

            with col_info:
                st.markdown(f"### Análisis Trabajador {i}")
                
                # Validación de Reglas
                presentes = {d['label'] for d in lista_epp}
                faltantes = EPP_REQUERIDO - presentes
                
                if not faltantes:
                    st.success("✅ **ACCESO AUTORIZADO**: EPP Completo")
                else:
                    st.error(f"❌ **ACCESO DENEGADO**: Faltan {', '.join(faltantes)}")

                # Métricas de confianza
                if lista_epp:
                    cols_m = st.columns(len(lista_epp))
                    for idx, d in enumerate(lista_epp):
                        cols_m[idx % 3].metric(d['label'], f"{d['conf']:.1%}")
                
        st.divider()

    # Footer
    st.caption("© 2026 UNAB - Ingeniería de Sistemas")

if __name__ == "__main__":
    main()
