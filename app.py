import os
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Configuración de entorno para Ultralytics
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -----------------------
# CONFIGURACIÓN Y TRADUCCIÓN
# -----------------------
# He añadido las clases específicas que mostró tu captura de pantalla
TRADUCCION_CLASES = {
    "No-hardhat": "Sin Casco",
    "No-safety vest": "Sin Chaleco",
    "No-mask": "Sin Mascarilla",
    "Worker": "Trabajador",
    "hardhat": "Casco",
    "safety-vest": "Chaleco",
    "boots": "Botas",
    "earmuffs": "Orejeras",
    "glasses": "Gafas",
    "gloves": "Guantes",
    "person": "Persona",
    "vest": "Chaleco"
}

# -----------------------
# LÓGICA DE PROCESAMIENTO
# -----------------------

@st.cache_resource
def load_models():
    """Carga los modelos de detección."""
    return YOLO("yolov8n.pt"), YOLO("best.pt")

def get_person_boxes(results):
    """Extrae coordenadas de personas y las convierte a lista para PIL."""
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Clase 0 es persona en COCO
            coords = list(map(int, box.xyxy[0]))
            boxes.append(coords)
    return boxes

def process_ppe_detection(image_crop, model_ppe):
    """Analiza el EPP en un recorte y devuelve datos traducidos."""
    temp_img = image_crop.copy()
    results = model_ppe(np.array(temp_img))[0]
    draw = ImageDraw.Draw(temp_img)
    detecciones = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label_en = model_ppe.names[cls]
        
        if label_en == "person": continue

        # Traducir usando el diccionario o capitalizar si no existe
        label_es = TRADUCCION_CLASES.get(label_en, label_en.replace("-", " ").capitalize())
        conf = float(box.conf[0])
        x1, y1, x2, y2 = list(map(int, box.xyxy[0]))

        # Dibujo de caja sobre el recorte
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
        
        detecciones.append({"label": label_es, "conf": conf})
        
    return temp_img, detecciones

# -----------------------
# INTERFAZ DE USUARIO (UI)
# -----------------------

def main():
    st.set_page_config(page_title="Detección de EPP", layout="wide")
    
    # CSS para aumentar el tamaño de las letras y mejorar el estilo
    st.markdown("""
        <style>
        /* Título y subtítulo */
        .main-title { text-align: center; color: #2E86C1; font-size: 42px !important; font-weight: bold; }
        .subtitle { text-align: center; color: #555; font-size: 22px !important; margin-bottom: 30px; }
        
        /* Aumentar tamaño de letra general de la app */
        html, body, [class*="st-"] {
            font-size: 18px;
        }
        
        /* Letras de los mensajes de error/éxito */
        .stAlert p {
            font-size: 22px !important;
            font-weight: 500;
        }

        /* Títulos de secciones */
        h1, h2, h3 {
            font-size: 28px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    modelo_personas, modelo_ppe = load_models()

    st.markdown("<h1 class='main-title'>🏭 Sistema de Detección de EPP</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Verificación automática de seguridad industrial</p>", unsafe_allow_html=True)

    # Selector de entrada
    with st.sidebar:
        st.header("Entrada de Imagen")
        opcion = st.radio("Selecciona fuente:", ["📁 Subir imagen", "📸 Usar cámara"])
        foto = st.file_uploader("Archivo", type=["jpg", "png", "jpeg"]) if "Subir" in opcion else st.camera_input("Capturar")

    if foto:
        img_original = Image.open(foto).convert("RGB")
        
        with st.spinner("Buscando trabajadores..."):
            res_personas = modelo_personas(np.array(img_original))[0]
            coords_personas = get_person_boxes(res_personas)

        if not coords_personas:
            st.warning("⚠️ No se detectaron personas en la imagen.")
            return

        st.markdown(f"### 👥 Personas en escena: {len(coords_personas)}")
        st.divider()

        # Procesar cada persona detectada
        for i, coords in enumerate(coords_personas, 1):
            with st.container():
                col_img, col_info = st.columns([1, 1.5])
                
                # Recorte de la persona
                crop = img_original.crop(coords)
                img_analizada, lista_epp = process_ppe_detection(crop, modelo_ppe)
                
                with col_img:
                    st.image(img_analizada, caption=f"Trabajador {i}", use_container_width=True)

                with col_info:
                    st.markdown(f"#### Estado Trabajador {i}")
                    
                    # Lógica de seguridad
                    labels_detectados = {d['label'] for d in lista_epp}
                    
                    # Verificamos si hay faltantes críticos basándonos en tu modelo
                    tiene_faltas = any(x in labels_detectados for x in ["Sin Casco", "Sin Chaleco"])
                    
                    if tiene_faltas:
                        faltantes = [x for x in ["Sin Casco", "Sin Chaleco"] if x in labels_detectados]
                        st.error(f"🔴 **ACCESO DENEGADO** - Detectado: {', '.join(faltantes)}")
                    elif "Casco" in labels_detectados or "Chaleco" in labels_detectados:
                        st.success("🟢 **ACCESO PERMITIDO** - EPP Detectado")
                    else:
                        st.warning("⚠️ **ATENCIÓN**: No se detectó equipo de protección.")

                    # Detalle de confianza con letras más grandes
                    if lista_epp:
                        st.markdown("**Análisis de Confianza:**")
                        for d in lista_epp:
                            st.write(f"🔍 {d['label']}: **{d['conf']:.2%}**")
                            st.progress(d['conf'])
                
                st.divider()

    else:
        st.info("Por favor, sube una imagen o usa la cámara para comenzar el análisis.")

    # Footer
    st.markdown("""
        <br><br><hr>
        <p style='text-align: center; color: gray; font-size: 14px;'>
        © 2026 UNAB - Ingeniería de Sistemas
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
