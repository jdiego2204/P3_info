# config.py
from pathlib import Path

# Carpeta raíz del proyecto (donde está este archivo)
ROOT = Path(__file__).resolve().parent

# Rutas principales (carpetas del proyecto)
DATA_DIR    = ROOT / "data"        # aquí están PPMI / Sarcoma / T2 / series
RESULTS_DIR = ROOT / "resultados"  # imágenes (vistas, zoom, segmentación, morfología)
DFS_DIR     = ROOT / "dataframes"  # CSV con info del estudio
NIFTI_DIR   = ROOT / "nifti"       # archivos .nii exportados

# Crear carpetas de salida si no existen (evita errores)
for d in (RESULTS_DIR, DFS_DIR, NIFTI_DIR):
    d.mkdir(parents=True, exist_ok=True)
