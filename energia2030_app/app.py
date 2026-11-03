from __future__ import annotations

from io import BytesIO
from pathlib import Path
import html
import unicodedata
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuración general y rutas. El notebook original se lee como insumo, pero
# nunca se modifica desde la aplicación.
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="ENERGÍA 2030",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#061329",
    "panel": "#102744",
    "panel_2": "#173657",
    "text": "#F2F7FF",
    "muted": "#A8B9D0",
    "turquoise": "#39C8FF",
    "green": "#47E7B2",
    "blue": "#4E8DFF",
    "amber": "#FFD166",
    "coral": "#FF7F9F",
}

SOURCE_COLORS = {
    "EOLICO": "#39C8FF",
    "SOLAR": "#FFC857",
    "HIDRO <=50MW": "#4DA3FF",
    "BIOMASA": "#63E6A6",
    "BIOGAS": "#C58BFF",
}

SOURCE_ICONS = {
    "EOLICO": "🌬️",
    "SOLAR": "☀️",
    "HIDRO <=50MW": "💧",
    "BIOMASA": "🌿",
    "BIOGAS": "♻️",
}


# -----------------------------------------------------------------------------
# Estilos visuales. Se mantienen en un único bloque para facilitar el aprendizaje
# y permitir cambiar la identidad visual sin tocar la lógica de datos.
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{ --turq: {COLORS['turquoise']}; --night: {COLORS['bg']}; }}
    .stApp {{ background: radial-gradient(circle at 82% 0%, #123A69 0, {COLORS['bg']} 42%); }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #08172D, #102744); border-right: 1px solid rgba(57,200,255,.18); }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(22,53,88,.96), rgba(10,25,48,.96));
        border: 1px solid rgba(57,200,255,.28); border-radius: 16px; padding: 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }}
    [data-testid="stMetricValue"] {{ color: {COLORS['turquoise']}; }}
    .hero {{
        padding: 28px 30px; border-radius: 22px; margin-bottom: 20px;
        background: linear-gradient(120deg, rgba(10,29,57,.98), rgba(17,74,116,.88));
        border: 1px solid rgba(57,200,255,.32);
    }}
    .hero h1 {{ margin: 0; color: white; font-size: clamp(2rem, 5vw, 4.1rem); line-height: .95; }}
    .hero p {{ color: #BDE6E2; max-width: 760px; margin: 14px 0 0; }}
    .eyebrow {{ color: {COLORS['turquoise']}; letter-spacing: .18em; font-size: .75rem; font-weight: 800; }}
    .learn {{
        border-left: 3px solid {COLORS['turquoise']}; padding: 9px 13px; margin: 8px 0 18px;
        color: {COLORS['muted']}; background: rgba(57,200,255,.065); border-radius: 0 9px 9px 0;
    }}
    .empty-map {{
        border: 1px dashed rgba(57,200,255,.45); border-radius: 18px; padding: 32px;
        background: rgba(11,35,56,.72); text-align: center;
    }}
    .source-chip {{ display:inline-block; padding:6px 10px; margin:3px; border-radius:999px;
        background:#103047; border:1px solid rgba(35,213,195,.25); color:#DDF8F4; }}
    div[data-testid="stDownloadButton"] button {{ border-color: {COLORS['turquoise']}; }}
    @media (max-width: 720px) {{ .hero {{ padding: 22px 18px; }} .block-container {{ padding-top: 1.2rem; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_gwh(value: float) -> str:
    """Formato numérico argentino compacto para indicadores."""
    return f"{value:,.0f}".replace(",", ".") + " GWh"


def explain(text: str) -> None:
    """Muestra una explicación breve debajo de cada bloque analítico."""
    st.markdown(f'<div class="learn">💡 <b>Qué aprendemos:</b> {text}</div>', unsafe_allow_html=True)


def chart_layout(fig: go.Figure, height: int = 430) -> go.Figure:
    """Aplica el mismo estilo a todos los gráficos Plotly."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=COLORS["panel"], font_color=COLORS["text"]),
    )
    fig.update_xaxes(gridcolor="rgba(148,184,186,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,184,186,.12)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    """Carga y tipa los CSV una sola vez por sesión."""
    files = {
        "annual": "historico_anual_fuentes.csv",
        "monthly": "historico_mensual_fuentes.csv",
        "models": "comparacion_modelos_ml.csv",
        "scenarios": "escenarios_energia_2030.csv",
        "backtest_scenarios": "backtest_escenarios_48m.csv",
        "backtest_sources": "backtest_fuentes_48m.csv",
        "provincial": "historico_provincial_fuentes.csv",
    }
    data = {key: pd.read_csv(DATA_DIR / name) for key, name in files.items()}
    data["annual"]["Año"] = pd.to_numeric(data["annual"]["Año"], errors="coerce")
    data["monthly"]["MES"] = pd.to_datetime(data["monthly"]["MES"], errors="coerce")
    data["monthly"]["Generacion_GWh"] = pd.to_numeric(data["monthly"]["Generacion_GWh"], errors="coerce")
    data["scenarios"]["Año"] = pd.to_numeric(data["scenarios"]["Año"], errors="coerce")
    data["provincial"]["Año"] = pd.to_numeric(data["provincial"]["Año"], errors="coerce")
    data["provincial"]["Generacion_GWh"] = pd.to_numeric(data["provincial"]["Generacion_GWh"], errors="coerce")
    return data



@st.cache_data(ttl=86400, show_spinner=False)
def load_province_geojson() -> dict:
    """Descarga una vez al día los límites provinciales oficiales de Georef Argentina."""
    response = requests.get("https://apis.datos.gob.ar/georef/api/v2.0/provincias.geojson", timeout=8)
    response.raise_for_status()
    return response.json()


def build_report(data: dict[str, pd.DataFrame]) -> bytes:
    """Genera un reporte HTML autocontenido, fácil de abrir o imprimir como PDF."""
    annual = data["annual"]
    scenarios = data["scenarios"]
    last_year = int(annual["Año"].max())
    last = annual.loc[annual["Año"] == last_year]
    leader = last.loc[last["Generacion_GWh"].idxmax()]
    total_2030 = scenarios.loc[scenarios["Año"] == 2030].groupby("Escenario")["Generacion_GWh"].sum()
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{fmt_gwh(v)}</td></tr>" for k, v in total_2030.items()
    )
    body = f"""<!doctype html><html lang='es'><meta charset='utf-8'><title>ENERGÍA 2030</title>
    <style>body{{font:16px Arial;max-width:900px;margin:40px auto;color:#102940}}h1{{color:#087f78}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccd}}small{{color:#587}}
    </style><h1>ENERGÍA 2030 — Reporte ejecutivo</h1><p>Generado desde los CSV del proyecto.</p>
    <h2>Último año histórico ({last_year})</h2><p>Total renovable: <b>{fmt_gwh(last['Generacion_GWh'].sum())}</b>.<br>
    Fuente líder: <b>{html.escape(str(leader['Fuente']))}</b> ({fmt_gwh(leader['Generacion_GWh'])}).</p>
    <h2>Escenarios 2030</h2><table><tr><th>Escenario</th><th>Generación total</th></tr>{rows}</table>
    <h2>Alcance</h2><p>El proyecto contiene datos nacionales agregados. Los CSV no incluyen provincia ni coordenadas;
    por ello no se presenta un mapa provincial. Las bandas de proyección se derivan del error porcentual de los backtests.</p>
    <small>Fuente: archivos exportados por analisis_energias_renovables(4).ipynb.</small></html>"""
    return body.encode("utf-8")


def model_frame(monthly: pd.DataFrame, source: str) -> pd.DataFrame:
    """Crea rezagos y variables estacionales sin usar información futura."""
    series = (
        monthly.loc[monthly["Fuente"] == source, ["MES", "Generacion_GWh"]]
        .dropna()
        .sort_values("MES")
        .set_index("MES")["Generacion_GWh"]
        .asfreq("MS")
    )
    frame = series.rename("y").to_frame()
    for lag in (1, 2, 3):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["mes_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["mes_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    return frame.dropna()


def run_lab(monthly: pd.DataFrame, source: str, model_name: str, test_months: int):
    """Entrena con corte temporal y devuelve predicciones y métricas comparables."""
    frame = model_frame(monthly, source)
    train, test = frame.iloc[:-test_months], frame.iloc[-test_months:]
    features = ["lag_1", "lag_2", "lag_3"]
    if model_name == "Persistencia mensual":
        pred = test["lag_1"].to_numpy()
    else:
        if model_name == "Ridge con estacionalidad":
            features += ["mes_sin", "mes_cos"]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            model = LinearRegression()
        model.fit(train[features], train["y"])
        pred = model.predict(test[features])
    result = test[["y"]].rename(columns={"y": "Real_GWh"}).copy()
    result["Prediccion_GWh"] = pred
    metrics = {
        "MAE": mean_absolute_error(result["Real_GWh"], pred),
        "RMSE": mean_squared_error(result["Real_GWh"], pred) ** 0.5,
        "R²": r2_score(result["Real_GWh"], pred),
    }
    return frame, train, result, metrics


data = load_data()
annual = data["annual"]
monthly = data["monthly"]
sources = sorted(annual["Fuente"].dropna().unique().tolist())
years = sorted(annual["Año"].dropna().astype(int).unique().tolist())


# -----------------------------------------------------------------------------
# Navegación y filtros globales. Los filtros se aplican en Inicio e Histórico;
# cada módulo especializado conserva sus propios controles.
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ ENERGÍA 2030")
    st.caption("Observatorio renovable · Argentina")
    section = st.radio(
        "Navegación",
        ["Inicio", "Storytelling", "Análisis histórico", "Machine Learning", "Proyecciones 2030", "Mapa provincial", "Descargas"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Filtros globales")
    year_range = st.slider("Rango de años", min(years), max(years), (min(years), max(years)))
    selected_sources = st.multiselect("Fuentes", sources, default=sources)
    if not selected_sources:
        st.warning("Elegí al menos una fuente.")
        selected_sources = sources
    st.caption("Los filtros afectan Inicio y Análisis histórico.")
    st.divider()
    st.caption("Datos nacionales · Sin información provincial exportada")

filtered = annual.loc[
    annual["Año"].between(*year_range) & annual["Fuente"].isin(selected_sources)
].copy()


if section == "Inicio":
    st.markdown(
        """<div class="hero"><div class="eyebrow">ARGENTINA · OBSERVATORIO RENOVABLE</div>
        <h1>ENERGÍA<br>2030</h1><p>Del dato histórico a escenarios transparentes: explorá la evolución,
        compará modelos y entendé qué puede cambiar hacia 2030.</p></div>""",
        unsafe_allow_html=True,
    )

    latest_year = int(filtered["Año"].max())
    latest = filtered.loc[filtered["Año"] == latest_year]
    total_latest = latest["Generacion_GWh"].sum()
    first_total = filtered.loc[filtered["Año"] == filtered["Año"].min(), "Generacion_GWh"].sum()
    growth = (total_latest / first_total - 1) * 100 if first_total else np.nan
    leader = latest.loc[latest["Generacion_GWh"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Generación {latest_year}", fmt_gwh(total_latest))
    c2.metric("Crecimiento del período", f"{growth:+.1f}%")
    c3.metric("Fuente líder", f"{SOURCE_ICONS.get(leader['Fuente'], '⚡')} {leader['Fuente']}")
    c4.metric("Fuentes visibles", str(len(selected_sources)))

    left, right = st.columns([1.65, 1])
    with left:
        totals = filtered.groupby("Año", as_index=False)["Generacion_GWh"].sum()
        fig = px.area(totals, x="Año", y="Generacion_GWh", markers=True, title="Evolución de la generación seleccionada")
        fig.update_traces(line_color=COLORS["turquoise"], fillcolor="rgba(35,213,195,.20)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        explain("La curva suma únicamente las fuentes y años elegidos. Un área creciente indica más energía generada, no necesariamente más capacidad instalada.")
    with right:
        fig = px.pie(
            latest,
            values="Generacion_GWh",
            names="Fuente",
            hole=.62,
            title=f"Participación por fuente · {latest_year}",
            color="Fuente",
            color_discrete_map=SOURCE_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value:,.0f} GWh<extra></extra>")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        explain("La participación compara el aporte relativo de cada fuente dentro de la selección actual.")

    st.subheader("Evolución por fuente")
    fig = px.line(
        filtered, x="Año", y="Generacion_GWh", color="Fuente", markers=True,
        color_discrete_map=SOURCE_COLORS,
    )
    st.plotly_chart(chart_layout(fig, 470), use_container_width=True)
    explain("Separar las series evita que el crecimiento de una tecnología oculte la estabilidad o caída de otra.")



elif section == "Storytelling":
    st.title("Storytelling energético")
    st.caption("De los registros a una lectura ejecutiva: qué cambió, dónde se concentra la generación y qué debemos mirar hacia 2030.")

    complete_years = sorted(annual["Año"].dropna().astype(int).unique())
    # Excluimos el último año si parece parcial; los datos mensuales llegan solo hasta agosto de 2026.
    end_year = max([y for y in complete_years if y < max(complete_years)], default=max(complete_years))
    start_year = min(complete_years)
    if end_year - start_year >= 5:
        start_year = max(start_year, end_year - 7)
    story = annual.loc[annual["Año"].between(start_year, end_year)].copy()
    totals_by_year = story.groupby("Año", as_index=False)["Generacion_GWh"].sum()
    source_totals = story.groupby("Fuente", as_index=False)["Generacion_GWh"].sum().sort_values("Generacion_GWh", ascending=False)
    first_year, last_year = int(totals_by_year["Año"].min()), int(totals_by_year["Año"].max())
    first_value = float(totals_by_year.loc[totals_by_year["Año"] == first_year, "Generacion_GWh"].sum())
    last_value = float(totals_by_year.loc[totals_by_year["Año"] == last_year, "Generacion_GWh"].sum())
    total_change = ((last_value / first_value) - 1) * 100 if first_value else np.nan
    leader = source_totals.iloc[0] if not source_totals.empty else None
    source_first = story.loc[story["Año"] == first_year].set_index("Fuente")["Generacion_GWh"]
    source_last = story.loc[story["Año"] == last_year].set_index("Fuente")["Generacion_GWh"]
    common = source_first.index.intersection(source_last.index)
    source_change = ((source_last.loc[common] / source_first.loc[common].replace(0, np.nan) - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna()
    fastest = source_change.idxmax() if not source_change.empty else None

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Cambio total · {first_year}–{last_year}", f"{total_change:+.1f}%" if pd.notna(total_change) else "N/D")
    k2.metric("Mayor aporte acumulado", str(leader["Fuente"]) if leader is not None else "N/D")
    k3.metric("Mayor crecimiento relativo", str(fastest) if fastest else "N/D")

    st.markdown("### La historia en tres hallazgos")
    if pd.notna(total_change):
        direction = "creció" if total_change >= 0 else "disminuyó"
        st.markdown(
            f'<div class="hero"><div class="eyebrow">HALLAZGO 01 · TENDENCIA</div>'
            f'<h2>La generación renovable {direction} {abs(total_change):.1f}%</h2>'
            f'<p>Entre {first_year} y {last_year}, el total pasó de {fmt_gwh(first_value)} a {fmt_gwh(last_value)}. '
            f'La comparación utiliza los años históricos seleccionados para evitar tratar 2026, que es parcial, como año completo.</p></div>',
            unsafe_allow_html=True,
        )
    if leader is not None:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">HALLAZGO 02 · COMPOSICIÓN</div>'
            f'<h2>{SOURCE_ICONS.get(leader["Fuente"], "⚡")} {leader["Fuente"]} lidera el aporte acumulado</h2>'
            f'<p>En el período {first_year}–{last_year}, esta fuente sumó {fmt_gwh(float(leader["Generacion_GWh"]))}. '
            f'El liderazgo por volumen no significa necesariamente que sea la fuente que más crece porcentualmente.</p></div>',
            unsafe_allow_html=True,
        )
    if fastest:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">HALLAZGO 03 · CAMBIO RELATIVO</div>'
            f'<h2>{SOURCE_ICONS.get(fastest, "⚡")} {fastest} muestra el mayor crecimiento relativo</h2>'
            f'<p>La comparación se realiza entre {first_year} y {last_year} cuando existe un valor positivo en ambos extremos. '
            f'Los porcentajes altos pueden partir de una base pequeña, por lo que también conviene mirar los GWh absolutos.</p></div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(totals_by_year, x="Año", y="Generacion_GWh", markers=True, title="Generación renovable total")
        fig.update_traces(line_color=COLORS["turquoise"], line_width=3, marker_size=8)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    with col2:
        mix = story.groupby("Fuente", as_index=False)["Generacion_GWh"].sum().sort_values("Generacion_GWh", ascending=True)
        fig = px.bar(mix, x="Generacion_GWh", y="Fuente", orientation="h", color="Fuente",
                     color_discrete_map=SOURCE_COLORS, title=f"Aporte acumulado por fuente · {first_year}–{last_year}")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
    explain("El storytelling traduce métricas en una lectura breve. Las conclusiones se recalculan desde el CSV y no son afirmaciones sobre capacidad instalada ni causalidad.")


elif section == "Análisis histórico":
    st.title("Análisis histórico")
    st.caption("Exploración anual y mensual de las cinco fuentes con cobertura reciente.")

    pivot = filtered.pivot_table(index="Año", columns="Fuente", values="Generacion_GWh", aggfunc="sum")
    fig = px.imshow(
        pivot.T,
        aspect="auto",
        color_continuous_scale=[[0, "#0B2338"], [.5, "#167D7F"], [1, "#63E6A6"]],
        labels=dict(color="GWh"),
        title="Intensidad anual por fuente",
    )
    st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
    explain("El mapa de calor facilita detectar cambios de escala: los tonos claros representan más GWh dentro del rango visible.")

    col_a, col_b = st.columns(2)
    with col_a:
        first = filtered.sort_values("Año").groupby("Fuente").first()["Generacion_GWh"]
        last = filtered.sort_values("Año").groupby("Fuente").last()["Generacion_GWh"]
        change = ((last / first - 1) * 100).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        change_df = change.rename("Variacion_%").reset_index()
        fig = px.bar(
            change_df, x="Variacion_%", y="Fuente", orientation="h", color="Variacion_%",
            color_continuous_scale=[COLORS["coral"], COLORS["amber"], COLORS["green"]],
            title=f"Variación entre {year_range[0]} y {year_range[1]}",
        )
        st.plotly_chart(chart_layout(fig, 410), use_container_width=True)
        explain("La variación usa el primer y último año del filtro; por eso puede cambiar al mover el rango.")
    with col_b:
        monthly_filtered = monthly.loc[
            monthly["Fuente"].isin(selected_sources)
            & monthly["MES"].dt.year.between(*year_range)
        ].dropna(subset=["Generacion_GWh"])
        season = monthly_filtered.assign(Mes=monthly_filtered["MES"].dt.month).groupby("Mes", as_index=False)["Generacion_GWh"].mean()
        fig = px.line(season, x="Mes", y="Generacion_GWh", markers=True, title="Promedio estacional mensual")
        fig.update_traces(line_color=COLORS["turquoise"], marker_size=8)
        st.plotly_chart(chart_layout(fig, 410), use_container_width=True)
        explain("Promediar cada mes del calendario ayuda a ver estacionalidad, aunque reúne años y fuentes de escalas distintas.")

    with st.expander("Ver tabla histórica filtrada"):
        st.dataframe(filtered.sort_values(["Año", "Fuente"]), use_container_width=True, hide_index=True)


elif section == "Machine Learning":
    st.title("Machine Learning")
    st.caption("Comparador reproducible y laboratorio con separación temporal — el futuro nunca entra al entrenamiento.")

    metrics_df = data["models"].sort_values("RMSE_GWh")
    best = metrics_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor RMSE", best["Modelo"], f"{best['RMSE_GWh']:.1f} GWh", delta_color="off")
    c2.metric("MAE", f"{best['MAE_GWh']:.1f} GWh")
    c3.metric("R²", f"{best['R2']:.3f}")

    fig = px.bar(
        metrics_df.melt(id_vars="Modelo", value_vars=["MAE_GWh", "RMSE_GWh"], var_name="Métrica", value_name="GWh"),
        x="GWh", y="Modelo", color="Métrica", barmode="group", orientation="h",
        color_discrete_sequence=[COLORS["turquoise"], COLORS["blue"]], title="Error de los modelos en 2025",
    )
    st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
    explain("MAE y RMSE más bajos son mejores. RMSE penaliza con más fuerza los errores grandes; R² más alto indica mayor capacidad explicativa.")

    st.subheader("Laboratorio interactivo por etapas")
    st.markdown("**1 · Elegí el problema**")
    a, b, c = st.columns(3)
    lab_source = a.selectbox("Fuente a predecir", sources, index=sources.index("EOLICO") if "EOLICO" in sources else 0)
    lab_model = b.selectbox("Método", metrics_df["Modelo"].tolist())
    max_test = min(24, max(6, len(model_frame(monthly, lab_source)) // 3))
    test_months = c.slider("Meses de prueba", 6, max_test, min(12, max_test))

    frame, train, result, lab_metrics = run_lab(monthly, lab_source, lab_model, test_months)
    st.markdown("**2 · Separación temporal**")
    st.info(
        f"Entrenamiento: {train.index.min():%b %Y} → {train.index.max():%b %Y} ({len(train)} meses) · "
        f"Prueba: {result.index.min():%b %Y} → {result.index.max():%b %Y} ({len(result)} meses)."
    )
    st.markdown("**3 · Resultado fuera de muestra**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{lab_metrics['MAE']:.1f} GWh")
    m2.metric("RMSE", f"{lab_metrics['RMSE']:.1f} GWh")
    m3.metric("R²", f"{lab_metrics['R²']:.3f}")
    plot_result = result.reset_index().melt(id_vars="MES", var_name="Serie", value_name="GWh")
    fig = px.line(
        plot_result, x="MES", y="GWh", color="Serie", markers=True,
        color_discrete_map={"Real_GWh": COLORS["green"], "Prediccion_GWh": COLORS["turquoise"]},
        title=f"Real vs predicción · {lab_source}",
    )
    st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
    explain("La prueba usa los últimos meses disponibles y no se mezcla con el entrenamiento. Cambiá fuente, método o ventana para comprobar la estabilidad del resultado.")
    with st.expander("¿Qué hace cada etapa?"):
        st.markdown(
            """1. **Preparación:** ordena la serie y crea rezagos de 1, 2 y 3 meses.  
2. **Entrenamiento:** ajusta el modelo solo con el pasado. Ridge suma variables seno/coseno para representar el ciclo anual.  
3. **Evaluación:** compara predicciones con meses que el modelo no vio. La persistencia usa el mes anterior y funciona como referencia mínima."""
        )


elif section == "Proyecciones 2030":
    st.title("Proyecciones 2030")
    st.caption("Tres escenarios exploratorios con bandas de incertidumbre derivadas del backtest histórico.")

    scenario = st.segmented_control(
        "Escenario destacado",
        ["Conservador", "Intermedio", "Alto"],
        default="Intermedio",
        selection_mode="single",
    ) or "Intermedio"
    scen = data["scenarios"]
    totals = scen.groupby(["Año", "Escenario"], as_index=False)["Generacion_GWh"].sum()
    fig = px.line(
        totals, x="Año", y="Generacion_GWh", color="Escenario", markers=True,
        color_discrete_map={"Conservador": COLORS["blue"], "Intermedio": COLORS["turquoise"], "Alto": COLORS["green"]},
        title="Generación renovable total por escenario",
    )
    st.plotly_chart(chart_layout(fig, 450), use_container_width=True)
    explain("Los escenarios no son promesas: muestran trayectorias bajo tasas de crecimiento distintas y deben leerse junto al error histórico.")

    selected = totals.loc[totals["Escenario"] == scenario].sort_values("Año").copy()
    errors = data["backtest_scenarios"].loc[data["backtest_scenarios"]["Escenario"] == scenario]
    bands = []
    for _, row in selected.iterrows():
        horizon = min(int(row["Año"] - selected["Año"].min() + 1), 4)
        pool = errors.loc[errors["Horizonte"] == horizon, "Error_total_pct"].dropna()
        error_p80 = float(pool.quantile(.80)) if len(pool) else float(errors["Error_total_pct"].quantile(.80))
        bands.append(error_p80)
    selected["Error_P80_pct"] = bands
    selected["Limite_inferior"] = selected["Generacion_GWh"] * (1 - selected["Error_P80_pct"] / 100).clip(lower=0)
    selected["Limite_superior"] = selected["Generacion_GWh"] * (1 + selected["Error_P80_pct"] / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected["Año"], y=selected["Limite_superior"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=selected["Año"], y=selected["Limite_inferior"], fill="tonexty", line=dict(width=0),
        fillcolor="rgba(35,213,195,.17)", name="Banda empírica P80", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=selected["Año"], y=selected["Generacion_GWh"], mode="lines+markers", name=scenario,
        line=dict(color=COLORS["turquoise"], width=3), marker=dict(size=9),
    ))
    fig.update_layout(title=f"{scenario}: trayectoria y límites de incertidumbre")
    st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
    explain("La banda aplica el percentil 80 del error absoluto observado en backtests del mismo escenario y horizonte. Para 2030 se reutiliza el horizonte 4, el máximo validado. No es un intervalo de confianza probabilístico.")

    by_source = scen.loc[scen["Escenario"] == scenario]
    fig = px.area(
        by_source, x="Año", y="Generacion_GWh", color="Fuente", color_discrete_map=SOURCE_COLORS,
        title=f"Composición del escenario {scenario}",
    )
    st.plotly_chart(chart_layout(fig, 450), use_container_width=True)
    explain("El área apilada muestra cuánto aporta cada fuente al total proyectado y si la matriz se concentra o diversifica.")

    table_2030 = by_source.loc[by_source["Año"] == 2030].sort_values("Generacion_GWh", ascending=False).copy()
    table_2030["Participacion_%"] = table_2030["Generacion_GWh"] / table_2030["Generacion_GWh"].sum() * 100
    st.subheader(f"Foto de 2030 · {scenario}")
    st.dataframe(table_2030[["Fuente", "Generacion_GWh", "Participacion_%"]], hide_index=True, use_container_width=True)


elif section == "Mapa provincial":
    st.title("Mapa energético provincial")
    st.caption("Generación renovable registrada por provincia y fuente predominante en los datos disponibles.")

    provincial = data["provincial"].copy()
    prov_years = sorted(provincial["Año"].dropna().astype(int).unique())
    map_year = st.select_slider("Año del mapa", options=prov_years, value=max(prov_years))
    year_data = provincial.loc[provincial["Año"] == map_year].copy()
    if year_data.empty:
        st.warning("No hay registros provinciales para ese año.")
    else:
        totals_prov = year_data.groupby("Provincia", as_index=False)["Generacion_GWh"].sum()
        pred = year_data.sort_values("Generacion_GWh", ascending=False).drop_duplicates("Provincia")
        pred = pred[["Provincia", "Fuente", "Generacion_GWh"]].rename(
            columns={"Fuente": "Fuente_predominante", "Generacion_GWh": "GWh_fuente_predominante"}
        )
        prov_summary = totals_prov.merge(pred, on="Provincia", how="left")
        prov_summary["Icono"] = prov_summary["Fuente_predominante"].map(SOURCE_ICONS).fillna("⚡")
        prov_summary = prov_summary.sort_values("Generacion_GWh", ascending=False)

        def normalize_name(value):
            value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
            return " ".join(value.upper().replace("CIUDAD AUTONOMA DE BUENOS AIRES", "CIUDAD AUTONOMA DE BUENOS AIRES").split())

        # Centroides aproximados para colocar los iconos sobre el mapa; los polígonos vienen de Georef Argentina.
        centroids = {
            "BUENOS AIRES": (-36.7, -60.6), "CATAMARCA": (-27.3, -66.0), "CHACO": (-26.5, -60.8),
            "CHUBUT": (-43.3, -68.5), "CORDOBA": (-32.1, -63.8), "CORRIENTES": (-28.8, -58.0),
            "ENTRE RIOS": (-32.0, -59.0), "FORMOSA": (-24.9, -59.5), "JUJUY": (-23.3, -65.8),
            "LA PAMPA": (-37.2, -66.0), "LA RIOJA": (-29.4, -67.0), "MENDOZA": (-34.5, -68.6),
            "MISIONES": (-26.9, -54.5), "NEUQUEN": (-38.9, -70.0), "RIO NEGRO": (-40.5, -67.5),
            "SALTA": (-25.1, -65.4), "SAN JUAN": (-30.9, -68.9), "SAN LUIS": (-33.8, -66.0),
            "SANTA CRUZ": (-48.8, -69.3), "SANTA FE": (-30.7, -60.9),
            "SANTIAGO DEL ESTERO": (-27.8, -63.3), "TIERRA DEL FUEGO": (-54.3, -67.8),
            "TUCUMAN": (-26.8, -65.2), "CIUDAD AUTONOMA DE BUENOS AIRES": (-34.6, -58.4),
        }
        try:
            # Geometrías oficiales de provincias desde la API pública de Georef Argentina.
            geojson = load_province_geojson()
            features = geojson.get("features", [])
            # Se cruzan nombres normalizados para tolerar tildes y diferencias de mayúsculas.
            feature_name_by_norm = {}
            for feature in features:
                props = feature.get("properties", {})
                official_name = props.get("nombre") or props.get("nombre_completo")
                if official_name:
                    feature_name_by_norm[normalize_name(official_name)] = official_name
            plot_data = []
            summary_by_norm = {normalize_name(row["Provincia"]): row for _, row in prov_summary.iterrows()}
            for norm_name, official_name in feature_name_by_norm.items():
                item = summary_by_norm.get(norm_name)
                plot_data.append({
                    "Provincia_geo": official_name,
                    "Generacion_GWh": float(item["Generacion_GWh"]) if item is not None else None,
                    "Provincia": item["Provincia"] if item is not None else official_name,
                    "Fuente_predominante": item["Fuente_predominante"] if item is not None else "Sin datos",
                })
            geo_df = pd.DataFrame(plot_data)
            fig = go.Figure()
            fig.add_trace(go.Choropleth(
                geojson=geojson,
                locations=geo_df["Provincia_geo"],
                z=geo_df["Generacion_GWh"],
                featureidkey="properties.nombre",
                colorscale=[[0, "#173657"], [.5, "#167DB6"], [1, "#39C8FF"]],
                marker_line_color="#8FA7C5",
                marker_line_width=.7,
                colorbar_title="GWh",
                customdata=geo_df[["Provincia", "Fuente_predominante"]],
                hovertemplate="<b>%{customdata[0]}</b><br>Generación: %{z:,.1f} GWh<br>Fuente líder: %{customdata[1]}<extra></extra>",
                name="Generación",
            ))
            icon_rows = []
            for _, row in prov_summary.iterrows():
                key = normalize_name(row["Provincia"])
                if key in centroids:
                    lat, lon = centroids[key]
                    icon_rows.append({
                        "Provincia": row["Provincia"], "Lat": lat, "Lon": lon,
                        "Icono": row["Icono"], "Fuente": row["Fuente_predominante"],
                        "Generacion_GWh": row["Generacion_GWh"],
                    })
            icon_df = pd.DataFrame(icon_rows)
            if not icon_df.empty:
                fig.add_trace(go.Scattergeo(
                    lat=icon_df["Lat"], lon=icon_df["Lon"], text=icon_df["Icono"],
                    mode="text", textfont=dict(size=18), customdata=icon_df[["Provincia", "Fuente", "Generacion_GWh"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>Predominante: %{customdata[1]}<br>Total: %{customdata[2]:,.1f} GWh<extra></extra>",
                    showlegend=False, name="Fuente predominante",
                ))
            fig.update_geos(
                scope="south america", center=dict(lat=-38.5, lon=-64.5),
                projection_type="mercator", showland=True, landcolor="#0A1B31",
                showocean=True, oceancolor="#061329", showcountries=True,
                countrycolor="#526B8B", showcoastlines=True, coastlinecolor="#526B8B",
                lataxis_range=[-56, -20], lonaxis_range=[-76, -52],
            )
            fig.update_layout(title=f"Argentina · generación renovable por provincia · {map_year}")
            st.plotly_chart(chart_layout(fig, 620), use_container_width=True)
            st.caption("Los polígonos se consultan a Georef Argentina. Los íconos se colocan en centroides aproximados para lectura visual; no representan la ubicación exacta de centrales.")
        except Exception as exc:
            st.warning("No se pudieron cargar los polígonos oficiales en línea. Mostramos una vista alternativa por centroides aproximados.")
            fallback_rows = []
            for _, row in prov_summary.iterrows():
                key = normalize_name(row["Provincia"])
                if key in centroids:
                    lat, lon = centroids[key]
                    fallback_rows.append({
                        "Provincia": row["Provincia"], "Lat": lat, "Lon": lon,
                        "Icono": row["Icono"], "Fuente": row["Fuente_predominante"],
                        "Generacion_GWh": row["Generacion_GWh"],
                    })
            fallback_df = pd.DataFrame(fallback_rows)
            if not fallback_df.empty:
                fallback_fig = go.Figure(go.Scattergeo(
                    lat=fallback_df["Lat"], lon=fallback_df["Lon"],
                    text=fallback_df["Icono"], mode="markers+text",
                    textposition="top center", textfont=dict(size=16, color="#F2F7FF"),
                    marker=dict(size=10, color=COLORS["turquoise"], line=dict(width=1, color="#FFFFFF")),
                    customdata=fallback_df[["Provincia", "Fuente", "Generacion_GWh"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>Predominante: %{customdata[1]}<br>Total: %{customdata[2]:,.1f} GWh<extra></extra>",
                ))
                fallback_fig.update_geos(
                    scope="south america", center=dict(lat=-38.5, lon=-64.5),
                    projection_type="mercator", showland=True, landcolor="#102744",
                    showocean=True, oceancolor="#061329", showcountries=True,
                    countrycolor="#526B8B", lataxis_range=[-56, -20], lonaxis_range=[-76, -52],
                )
                fallback_fig.update_layout(title=f"Vista alternativa · {map_year}")
                st.plotly_chart(chart_layout(fallback_fig, 560), use_container_width=True)
            st.caption(f"Detalle técnico: {type(exc).__name__}. El mapa detallado requiere conexión a Georef Argentina.")

        left, right = st.columns([1, 1.3])
        with left:
            province_options = prov_summary["Provincia"].tolist()
            selected_province = st.selectbox("Explorar provincia", province_options)
            one = year_data.loc[year_data["Provincia"] == selected_province].sort_values("Generacion_GWh", ascending=False)
            total_selected = float(one["Generacion_GWh"].sum())
            leader_selected = one.iloc[0]
            st.metric("Generación registrada", fmt_gwh(total_selected))
            st.metric("Fuente predominante", f'{SOURCE_ICONS.get(leader_selected["Fuente"], "⚡")} {leader_selected["Fuente"]}')
            fig = px.pie(one, values="Generacion_GWh", names="Fuente", hole=.55,
                         color="Fuente", color_discrete_map=SOURCE_COLORS, title=f"Matriz renovable · {selected_province}")
            st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
        with right:
            rank = prov_summary.head(10).sort_values("Generacion_GWh")
            rank["Provincia con fuente"] = rank["Icono"] + "  " + rank["Provincia"]
            fig = px.bar(rank, x="Generacion_GWh", y="Provincia con fuente", orientation="h",
                         color="Fuente_predominante", color_discrete_map=SOURCE_COLORS,
                         title="Top 10 provincias por generación renovable")
            st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
            st.dataframe(
                prov_summary[["Provincia", "Fuente_predominante", "Generacion_GWh"]].rename(
                    columns={"Fuente_predominante": "Fuente predominante", "Generacion_GWh": "Generación (GWh)"}
                ),
                hide_index=True, use_container_width=True,
            )
        explain("La fuente predominante se calcula como la tecnología con más GWh registrados en la provincia y año elegidos. Las provincias sin registros no se etiquetan con una fuente inventada.")

else:  # Descargas
    st.title("Descargas")
    st.caption("Datos originales exportados, selección filtrada y reporte ejecutivo.")

    st.subheader("Selección actual")
    filtered_csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Descargar histórico filtrado", filtered_csv, "historico_filtrado.csv", "text/csv", use_container_width=True)

    st.subheader("Archivos del proyecto")
    file_names = [
        "historico_anual_fuentes.csv", "historico_mensual_fuentes.csv", "comparacion_modelos_ml.csv",
        "escenarios_energia_2030.csv", "backtest_escenarios_48m.csv", "backtest_fuentes_48m.csv", "historico_provincial_fuentes.csv",
    ]
    cols = st.columns(2)
    for i, name in enumerate(file_names):
        with cols[i % 2]:
            st.download_button(
                f"CSV · {name}",
                (DATA_DIR / name).read_bytes(),
                name,
                "text/csv",
                key=f"download_{name}",
                use_container_width=True,
            )

    st.subheader("Reporte")
    st.download_button(
        "📄 Descargar reporte ejecutivo (HTML)",
        build_report(data),
        "reporte_energia_2030.html",
        "text/html",
        use_container_width=True,
    )
    explain("El HTML es autocontenido: puede abrirse en cualquier navegador y guardarse como PDF desde la opción Imprimir.")


st.divider()
st.caption("ENERGÍA 2030 · Fuente: archivos exportados por el notebook original · Las proyecciones son escenarios exploratorios.")
