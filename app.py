import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# -----------------------
# Configuración de página
# -----------------------
st.set_page_config(
    page_title="Sistema EPP · UNAB",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# Estilos mejorados
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section.main > div { padding-top: 1rem; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 55%, #1565C0 100%);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
}
.hero-icon { font-size: 3.2rem; line-height: 1; }
.hero-title {
    color: #ffffff;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.4px;
}
.hero-sub { color: #90CAF9; font-size: 0.97rem; margin: 0; }

/* ── Tarjeta de trabajador ── */
.worker-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a3a5c;
    margin-bottom: 0.9rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #EBF2FF;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Badge de estado ── */
.badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.3rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    background: #D1FAE5;
    color: #065F46;
    border: 1.5px solid #34D399;
    margin-bottom: 1.1rem;
}
.badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.3rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    background: #FEE2E2;
    color: #7F1D1D;
    border: 1.5px solid #F87171;
    margin-bottom: 0.5rem;
}
.faltantes-label {
    font-size: 0.85rem;
    color: #B91C1C;
    font-weight: 500;
    margin-bottom: 1rem;
    padding-left: 0.3rem;
}

/* ── Ítems EPP ── */
.ppe-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.6rem;
    margin-top: 0.4rem;
}
.ppe-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.52rem 0.85rem;
    border-radius: 10px;
    margin-bottom: 0.38rem;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
}
.ppe-item-name { font-size: 0.9rem; font-weight: 500; color: #1E293B; }
.ppe-item-conf {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1565C0;
    background: #EBF5FB;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
}
.ppe-item-neg { background: #FFF7ED; border-color: #FDBA74; }
.ppe-item-neg .ppe-item-name { color: #9A3412; }
.ppe-item-neg .ppe-item-conf { color: #9A3412; background: #FEE2E2; }

/* ── Stats chips ── */
.stat-chip {
    background: #EBF5FB;
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    border: 1px solid #BFDBFE;
    display: inline-block;
    margin-bottom: 0.8rem;
}
.stat-chip .num { font-size: 1.6rem; font-weight: 700; color: #1565C0; }
.stat-chip .lbl { font-size: 0.78rem; color: #64748B; font-weight: 500; margin-left: 0.4rem; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E2E8F0;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Diccionario de traducción COMPLETO
# Cubre todas las clases positivas y negativas típicas de modelos PPE/YOLOv8
# -----------------------
TRADUCCION_CLASES = {
    # ── Positivos ──
    "boots":            "Botas de seguridad",
    "earmuffs":         "Orejeras",
    "glasses":          "Gafas de seguridad",
    "gloves":           "Guantes",
    "helmet":           "Casco",
    "hardhat":          "Casco de seguridad",
    "mask":             "Mascarilla",
    "person":           "Persona",
    "vest":             "Chaleco reflectivo",
    "safety vest":      "Chaleco reflectivo",
    "safety-vest":      "Chaleco reflectivo",
    "worker":           "Trabajador",
    # ── Negativos (sin equipo) ──
    "no-hardhat":       "Sin casco",
    "no hardhat":       "Sin casco",
    "no-helmet":        "Sin casco",
    "no-mask":          "Sin mascarilla",
    "no mask":          "Sin mascarilla",
    "no-vest":          "Sin chaleco",
    "no-safety vest":   "Sin chaleco",
    "no safety vest":   "Sin chaleco",
    "no-safety-vest":   "Sin chaleco",
    "no-gloves":        "Sin guantes",
    "no-glasses":       "Sin gafas",
    "no-boots":         "Sin botas",
    "no-goggles":       "Sin gafas",
    "no-ear-muffs":     "Sin orejeras",
}

# Prefijos de clases negativas
CLASES_IGNORAR   = {"person", "worker"}
CLASES_NEGATIVAS = {k for k in TRADUCCION_CLASES if k.startswith("no")}

# -----------------------
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe      = YOLO("best.pt")
    return modelo_personas, modelo_ppe

modelo_personas, modelo_ppe = load_models()

# -----------------------
# Hero Banner
# -----------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🦺</div>
    <div>
        <div class="hero-title">Sistema de Detección de EPP</div>
        <div class="hero-sub">Verificación automática de equipos de protección personal · UNAB 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Selector de fuente
# -----------------------
col_r, _ = st.columns([1, 3])
with col_r:
    opcion = st.radio("**Fuente de imagen:**", ["📁 Subir imagen", "📸 Usar cámara"])

foto = None
if opcion == "📁 Subir imagen":
    foto = st.file_uploader("Sube una imagen (JPG / PNG)", type=["jpg", "png", "jpeg"],
                            label_visibility="collapsed")
else:
    foto = st.camera_input("Toma una foto", label_visibility="collapsed")

st.markdown("---")

# -----------------------
# Procesamiento principal
# -----------------------
if foto:
    imagen_original = Image.open(foto).convert("RGB")
    img_np          = np.array(imagen_original)

    col_img, col_info = st.columns([1, 2])

    with col_img:
        st.markdown("**📷 Imagen analizada**")
        st.image(imagen_original, use_container_width=True)

    with st.spinner("🔍 Detectando personas y EPP..."):
        resultados_personas = modelo_personas(img_np)[0]

    personas = []
    for box in resultados_personas.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            personas.append((x1, y1, x2, y2))

    with col_info:
        st.markdown(f"""
        <div class="stat-chip">
            <span class="num">{len(personas)}</span>
            <span class="lbl">persona(s) detectada(s)</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Desplázate para ver el análisis individual de cada trabajador.")

    st.markdown("---")

    # -----------------------
    # Análisis por trabajador
    # -----------------------
    for i, (x1, y1, x2, y2) in enumerate(personas, 1):

        persona_crop = imagen_original.crop((x1, y1, x2, y2))
        persona_np   = np.array(persona_crop)

        resultados_ppe = modelo_ppe(persona_np)[0]

        draw = ImageDraw.Draw(persona_crop)
        etiquetas_presentes = []   # nombres en español de EPP positivos detectados
        items_tabla = []           # lo que se muestra en la tabla

        for box in resultados_ppe.boxes:
            cls       = int(box.cls[0])
            label_raw = modelo_ppe.names[cls].lower().strip()

            if label_raw in CLASES_IGNORAR:
                continue

            es_negativo = label_raw in CLASES_NEGATIVAS
            label_es    = TRADUCCION_CLASES.get(label_raw, label_raw.replace("-", " ").title())
            conf        = float(box.conf[0])

            if not es_negativo:
                etiquetas_presentes.append(label_es.lower())

            items_tabla.append({"label": label_es, "conf": conf, "negativo": es_negativo})

            x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
            color = "#FF4444" if es_negativo else "#00FF66"
            draw.rectangle([x1o, y1o, x2o, y2o], outline=color, width=3)
            draw.text((x1o + 2, max(0, y1o - 16)), label_es, fill=color)

        # ── Layout ──
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(persona_crop, use_container_width=True, caption=f"Trabajador {i}")

        with col2:
            st.markdown(f'<div class="worker-title">👤 Trabajador {i}</div>', unsafe_allow_html=True)

            # Lógica de acceso: busca casco y chaleco (cualquier variante)
            tiene_casco   = any("casco" in e or "hardhat" in e or "helmet" in e
                                for e in etiquetas_presentes)
            tiene_chaleco = any("chaleco" in e or "vest" in e
                                for e in etiquetas_presentes)

            if tiene_casco and tiene_chaleco:
                st.markdown('<div class="badge-ok">✅ &nbsp;Acceso permitido</div>',
                            unsafe_allow_html=True)
            else:
                faltantes = []
                if not tiene_casco:   faltantes.append("🔴 Casco")
                if not tiene_chaleco: faltantes.append("🔴 Chaleco")
                st.markdown('<div class="badge-warn">🚫 &nbsp;Acceso denegado</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="faltantes-label">Falta: {" &nbsp;·&nbsp; ".join(faltantes)}</div>',
                    unsafe_allow_html=True)

            # Tabla de EPP detectado
            if items_tabla:
                st.markdown('<div class="ppe-section-title">📋 Equipos detectados</div>',
                            unsafe_allow_html=True)
                html_items = ""
                for it in items_tabla:
                    clase_extra = "ppe-item-neg" if it["negativo"] else ""
                    icono       = "⚠️" if it["negativo"] else "✔️"
                    pct         = f'{it["conf"] * 100:.1f}%'
                    html_items += f"""
                    <div class="ppe-item {clase_extra}">
                        <span class="ppe-item-name">{icono} &nbsp;{it['label']}</span>
                        <span class="ppe-item-conf">{pct}</span>
                    </div>"""
                st.markdown(html_items, unsafe_allow_html=True)
            else:
                st.warning("No se detectó ningún EPP en esta persona.")

        st.markdown("---")

# -----------------------
# Footer
# -----------------------
st.markdown("""
<div class="footer">© Alfredo Diaz · UNAB 2026 · Sistema de Verificación de EPP</div>
""", unsafe_allow_html=True)
