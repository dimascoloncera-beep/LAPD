import numpy as np
import pandas as pd
import streamlit as st
import folium
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================
# MAPA PRINCIPAL POR ÁREA
# =========================
def build_area_map(df, model, le, hour_for_map,
                   col_lat="latitud", col_lon="longitud",
                   col_area="area_nombre", topk=5,
                   min_n=1, use_n_in_radius=True):

    for c in [col_area, col_lat, col_lon]:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en el dataframe.")

    # limpiar
    d = df.dropna(subset=[col_area, col_lat, col_lon]).copy()
    d[col_lat] = pd.to_numeric(d[col_lat], errors="coerce")
    d[col_lon] = pd.to_numeric(d[col_lon], errors="coerce")
    d = d.dropna(subset=[col_lat, col_lon])

    if len(d) == 0:
        raise ValueError("No hay filas válidas después de limpiar lat/long.")

    # conteo de filas por área
    counts = d.groupby(col_area).size().rename("n_obs")

    # filtrar áreas con pocos datos
    if min_n > 1:
        valid_areas = counts[counts >= min_n].index
        d = d[d[col_area].isin(valid_areas)].copy()
        counts = d.groupby(col_area).size().rename("n_obs")

    if len(d) == 0:
        raise ValueError("No quedan áreas después de aplicar min_n.")

    # TOTAL de filas usadas para inferencia (ESA HORA, en tu lógica actual)
    n_total = len(d)

    # inferencia a hora fija
    X_inf = pd.DataFrame({
        "latitud": d[col_lat].values,
        "longitud": d[col_lon].values,
        "hora_entera": np.full(len(d), int(hour_for_map))
    })

    probs = model.predict_proba(X_inf[["latitud", "longitud", "hora_entera"]])
    probs_df = pd.DataFrame(probs, columns=le.classes_)
    probs_df[col_area] = d[col_area].values

    prob_area = probs_df.groupby(col_area).mean()
    centroids = d.groupby(col_area)[[col_lat, col_lon]].mean()

    dominant = prob_area.idxmax(axis=1)
    maxprob = prob_area.max(axis=1)

    summary = (
        centroids
        .join(prob_area, how="inner")
        .join(counts, how="inner")
        .assign(dominant=dominant, max_prob=maxprob)
        .reset_index()
        .rename(columns={col_area: "area"})
    )

    m = folium.Map(
        location=[d[col_lat].mean(), d[col_lon].mean()],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    base_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    color_map = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(le.classes_)}

    for _, row in summary.iterrows():
        area = row["area"]
        lat = float(row[col_lat])
        lon = float(row[col_lon])
        dom = row["dominant"]
        pmax = float(row["max_prob"])
        n_obs = int(row["n_obs"])

        top_series = prob_area.loc[area].sort_values(ascending=False).head(topk)

        table = "<table style='width:280px'>"
        table += f"<tr><th align='left'>Top {topk}</th><th align='right'>Prob</th></tr>"
        for lab, p in top_series.items():
            table += f"<tr><td>{lab}</td><td align='right'>{p*100:.1f}%</td></tr>"
        table += "</table>"

        popup_html = f"""
        <div style="font-family:Arial; font-size:13px;">
          <b>Área:</b> {area}<br>
          <b>Hora:</b> {hour_for_map}:00<br>
          <b>Registros usados (N área):</b> {n_obs}<br>
          <b>Categoría dominante:</b> {dom}<br>
          <b>Prob. dominante:</b> {pmax*100:.0f}%<br><br>
          {table}
        </div>
        """

        if use_n_in_radius:
            radius = 6 + (np.log1p(n_obs) * 2.0) + (pmax * 14.0)
        else:
            radius = 8 + pmax * 18

        folium.CircleMarker(
            location=[lat, lon],
            radius=float(radius),
            color=color_map.get(dom, "#000000"),
            fill=True,
            fill_opacity=0.60,
            popup=folium.Popup(popup_html, max_width=330),
            tooltip=f"{area} | N_area={n_obs} | Dom={dom} | {pmax*100:.0f}%"
        ).add_to(m)

    # ✅ DEVOLVER MAPA + TOTAL + SUMMARY PARA TABLA
    return m, n_total, summary


# =========================
# APP (LAYOUT SIDEBAR)
# =========================
st.set_page_config(page_title="Tesis:Análisis y visualización espacio-temporal del riesgo delictivo mediante aprendizaje automático. ", layout="wide")  
st.markdown("## 🗺️ Análisis y Visualización  del riesgo delictivo en Los Angeles, California.")
st.write("")

# ---- SIDEBAR
st.sidebar.markdown("### 1) Cargar los datos")
file = st.sidebar.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("### 2) Configurar la aplicacion")
hour = st.sidebar.selectbox("Hora (0–23)", list(range(24)), index=22)
topk = st.sidebar.selectbox("Top N de delitos a observar", [3, 5, 8, 10], index=1)
min_n = st.sidebar.number_input("Mínimo N registros por área", min_value=1, value=50, step=10)
use_n_in_radius = st.sidebar.checkbox("Usar N en el tamaño del círculo", value=True)

st.sidebar.write("")
run_btn = st.sidebar.button("Entrenar el  modelo y generar resultados", type="primary", use_container_width=True)

# ---- PANEL CENTRAL
if file is None:
    st.info("Sube el archivo para empezar.")
    st.stop()

# cargar
name = file.name.lower()
df = pd.read_csv(file) if name.endswith(".csv") else pd.read_excel(file)

# validar columnas
required = {"latitud", "longitud", "hora_entera", "Categoria_Delictiva", "area_nombre"}
missing = required - set(df.columns)
if missing:
    st.error(f"Faltan columnas: {sorted(missing)}")
    st.stop()

with st.expander("👀 Vista previa de datos", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

if not run_btn:
    st.warning("Configura los parámetros y pulsa **Entrenar modelo y generar mapa**.")
    st.stop()

# ---- ENTRENAR
X = df[["latitud", "longitud", "hora_entera"]].copy()
y = df["Categoria_Delictiva"].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    n_estimators=250,
    max_depth=6,
    learning_rate=0.12,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)

with st.spinner("Procesando los datos para obtener informacion"):
    model.fit(X_train, y_train)

st.success(f"Modelo entrenado. Clases: {len(le.classes_)}")

# ---- MAPA + ETIQUETA TOTAL + TABLA RESUMEN
with st.spinner("Generando mapa por área..."):
    m_area, n_total, summary = build_area_map(
        df=df,
        model=model,
        le=le,
        hour_for_map=hour,
        col_area="area_nombre",
        topk=int(topk),
        min_n=int(min_n),       
        use_n_in_radius=use_n_in_radius        
    )

# etiqueta/metric arriba del mapa
st.metric(label=f"Total de registros usados (hora {hour:02d}:00)", value=f"{n_total:,}")

components.html(m_area.get_root().render(), height=720, scrolling=False)

# =========================
# ✅ TABLA RESUMEN ABAJO
# =========================
st.markdown("### 📊 Resumen por área (hora seleccionada)")

tabla_resumen = (
    summary[["area", "dominant", "max_prob"]]
    .rename(columns={
        "area": "Área",
        "dominant": "Categoría dominante",
        "max_prob": "Probabilidad"
    })
    .copy()
)

tabla_resumen["Probabilidad"] = (tabla_resumen["Probabilidad"] * 100).round(1).astype(str) + "%"

# ordenar por mayor probabilidad (numérica)
tabla_resumen["_p"] = summary["max_prob"].values
tabla_resumen = tabla_resumen.sort_values("_p", ascending=False).drop(columns=["_p"])

st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)
