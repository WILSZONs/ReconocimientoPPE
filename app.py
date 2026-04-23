import os
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Configuración de entorno
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# -----------------------
# CONFIGURACIÓN Y CONSTANTES
# -----------------------
TRADUCCION_CLASES = {
    "No-hardhat": "Sin Casco",
    "No-safety vest": "Sin Chaleco",
    "No-mask": "Sin Mascarilla",
    "Worker": "Trabajador",
    "hardhat": "Casco",
    "safety-vest": "Chaleco",
    "boots": "Botas",
    "glasses": "Gafas",
    "gloves": "Guantes"
}
EPP_REQUERIDO = {"Casco", "Chaleco"}

# -----------------------
# LÓGICA DE PROCESAMIENTO
# -----------------------

@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("best.pt")

def get_person_boxes(results):
    """Extrae coordenadas de personas y las convierte a LISTA (importante)"""
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            # Convertimos a lista de enteros para que PIL pueda procesarlo
            coords = list(map(int, box.xyxy[0]))
            boxes.append(coords)
    return boxes

def process_ppe_detection(image_crop, model_ppe):
    """Analiza el EPP en un recorte de imagen."""
    # Hacemos una copia para no dibujar sobre la original accidentalmente
    temp_img = image_crop.copy()
    results = model_ppe(np.array(temp_img))[0]
    draw = ImageDraw.Draw(temp_img)
    detecciones = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label_en = model_ppe.names[cls]
        
        if label_en == "person": continue

        label_es = TRADUCCION_CLASES.get(label_en, label_en.capitalize())
        conf = float(box.conf[0])
        x1, y1, x2, y2 = list(map(int, box.xyxy[0]))

        # Dibujo
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
        draw.text((x1, max(0, y1 - 20)), f"{label_es}", fill="#00FF00")
        
        detecciones.append({"label": label_es, "conf": conf})
        
    return temp_img, detecciones

# -----------------------
# INTERFAZ DE USUARIO (UI)
# -----------------------

def main():
    st.set_page_config(page_title="Detección de EPP", layout="wide")
    
    # CSS inyectado
    st.markdown("""
        <style>
        .main-title { text-align: center; color: #2E86C1; }
        .subtitle { text-align: center; color: #555; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    modelo_personas, modelo_ppe = load_models()

    st.markdown("<h1 class='main-title'>🏭 Sistema de Detección de EPP</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Verificación automática de seguridad industrial</p>", unsafe_allow_html=True)

    # Entrada de datos
    opcion = st.radio("Fuente:", ["📁 Subir imagen", "📸 Usar cámara"], horizontal=True)
    foto = st.file_uploader("Imagen", type=["jpg", "png"]) if "Subir" in opcion else st.camera_input("Foto")

    if foto:
        img_original = Image.open(foto).convert("RGB")
        
        with st.spinner("Analizando..."):
            res_personas = modelo_personas(np.array(img_original))[0]
            coords_personas = get_person_boxes(res_personas)

        if not coords_personas:
            st.warning("No se detectaron personas en la imagen.")
            return

        st.subheader(f"👥 Trabajadores detectados: {len(coords_personas)}")
        
        # Iterar sobre cada persona detectada
        for i, coords in enumerate(coords_personas, 1):
            with st.expander(f"👤 Ver detalles del Trabajador {i}", expanded=True):
                col_img, col_info = st.columns([1, 2])
                
                # RECORTE SEGURO
                crop = img_original.crop(coords) 
                img_detectada, lista_epp = process_ppe_detection(crop, modelo_ppe)
                
                with col_img:
                    st.image(img_detectada, use_container_width=True)

                with col_info:
                    presentes = {d['label'] for d in lista_epp}
                    faltantes = EPP_REQUERIDO - presentes
                    
                    if not faltantes:
                        st.success("🟢 **ACCESO PERMITIDO**")
                    else:
                        st.error(f"🔴 **ACCESO DENEGADO** - Faltan: {', '.join(faltantes)}")

                    # Mostrar confianza
                    if lista_epp:
                        for d in lista_epp:
                            st.text(f"{d['label']}: {d['conf']:.2%}")
                            st.progress(d['conf'])

    # Footer
    st.markdown("---")
    st.caption("© 2026 UNAB - Reconocimiento de EPP")

if __name__ == "__main__":
    main()
