# ENERGÍA 2030 — Aplicación Streamlit

Dashboard de análisis histórico, storytelling, comparación de modelos, escenarios 2030, mapa provincial y descargas.

## Ejecutar en Windows

1. Descomprimí la carpeta en una ubicación fácil de encontrar.
2. Abrí la carpeta `energia2030_app` en VS Code.
3. En **Terminal → Nueva terminal**, ejecutá:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Abrí la dirección local que muestre Streamlit, normalmente `http://localhost:8501`.

Si PowerShell bloquea la activación del entorno, podés instalar y ejecutar con `python -m pip install -r requirements.txt` y luego `python -m streamlit run app.py` sin activar `.venv`.

## Estructura

- `app.py`: aplicación.
- `.streamlit/config.toml`: tema y configuración visual.
- `data/`: CSV históricos, métricas, escenarios, backtests y datos provinciales.
- `data/historico_provincial_fuentes.csv`: agregación por año, provincia y fuente creada desde la hoja `Base de Datos` del XLSM original.

## Mapa provincial

La generación y la fuente predominante se calculan a partir de los registros de la hoja `Base de Datos`, filtrando EOLICO, SOLAR, HIDRO <=50MW, BIOMASA y BIOGAS. Los límites de provincias se consultan desde la API pública de Georef Argentina; se necesita conexión a Internet para ver los polígonos. Si no hay conexión, la aplicación intenta mostrar una vista alternativa con centroides aproximados. Las provincias sin registros no reciben una fuente predominante inventada.

## Storytelling

La sección de storytelling genera hallazgos a partir de los CSV históricos. Para evitar tratar un año incompleto como año entero, la comparación excluye el año máximo disponible.
