import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from ultralytics import YOLO
import numpy as np
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------
# Configuración
# -----------------------
st.set_page_config(page_title="Detección PPE", layout="wide")

# -----------------------
# Traducción completa
# -----------------------
TRADUCCION_CLASES = {
    "boots": "Botas",
    "earmuffs": "Orejeras",
    "glasses": "Gafas",
    "gloves": "Guantes",
    "helmet": "Casco",
    "mask": "Mascarilla",
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
st.title("🏭 Detección de EPP en Tiempo Real")
st.markdown("Sistema de monitoreo de seguridad industrial con IA")

st.markdown("---")

# -----------------------
# Procesador de video
# -----------------------
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detectar personas
        resultados_personas = modelo_personas(img)[0]

        for box in resultados_personas.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                persona = img[y1:y2, x1:x2]

                # Detectar EPP en la persona
                resultados_ppe = modelo_ppe(persona)[0]

                etiquetas = []

                for box_ppe in resultados_ppe.boxes:
                    cls_ppe = int(box_ppe.cls[0])
                    label_ing = modelo_ppe.names[cls_ppe]

                    if label_ing == "person":
                        continue

                    label_esp = TRADUCCION_CLASES.get(label_ing, label_ing)
                    etiquetas.append(label_esp)

                    xp1, yp1, xp2, yp2 = map(int, box_ppe.xyxy[0])

                    # Dibujar cajas PPE
                    cv2.rectangle(persona, (xp1, yp1), (xp2, yp2), (0,255,0), 2)
                    cv2.putText(persona, label_esp, (xp1, yp1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Validación
                requeridos = {"Casco", "Chaleco"}
                presentes = set(etiquetas)

                if requeridos.issubset(presentes):
                    color = (0,255,0)
                    texto = "PERMITIDO"
                else:
                    color = (0,0,255)
                    texto = "DENEGADO"

                # Caja persona
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, texto, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------
# Modo de uso
# -----------------------
modo = st.radio("Modo de uso:", ["📸 Cámara en vivo", "📁 Imagen"])

# -----------------------
# Cámara en vivo
# -----------------------
if modo == "📸 Cámara en vivo":
    st.info("Activa tu cámara y el sistema analizará en tiempo real")

    webrtc_streamer(
        key="ppe",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# -----------------------
# Imagen
# -----------------------
else:
    from PIL import Image, ImageDraw
    import pandas as pd

    foto = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    if foto:
        imagen = Image.open(foto).convert("RGB")
        st.image(imagen, use_container_width=True)

        img_np = np.array(imagen)

        resultados_personas = modelo_personas(img_np)[0]

        personas = []
        for box in resultados_personas.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                personas.append((x1, y1, x2, y2))

        st.subheader(f"Personas detectadas: {len(personas)}")

        for i, (x1, y1, x2, y2) in enumerate(personas, 1):
            st.markdown(f"### Trabajador {i}")

            crop = imagen.crop((x1, y1, x2, y2))
            crop_np = np.array(crop)

            resultados_ppe = modelo_ppe(crop_np)[0]

            draw = ImageDraw.Draw(crop)
            etiquetas = []

            for box in resultados_ppe.boxes:
                cls = int(box.cls[0])
                label_ing = modelo_ppe.names[cls]

                if label_ing == "person":
                    continue

                label_esp = TRADUCCION_CLASES.get(label_ing, label_ing)
                etiquetas.append(label_esp)

                x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
                draw.rectangle([x1o, y1o, x2o, y2o], outline="green", width=3)
                draw.text((x1o, y1o-10), label_esp, fill="green")

            col1, col2 = st.columns(2)

            with col1:
                st.image(crop, use_container_width=True)

            with col2:
                requeridos = {"Casco", "Chaleco"}

                if requeridos.issubset(set(etiquetas)):
                    st.success("🟢 Acceso permitido")
                else:
                    st.error("🔴 Acceso denegado")

            st.markdown("---")
