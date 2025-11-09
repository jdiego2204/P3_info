# clases.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import nibabel as nib

from config import RESULTS_DIR, DFS_DIR, NIFTI_DIR

# Utilidades  
def _safe_get(ds, name, default=""):
    try:
        return getattr(ds, name)
    except Exception:
        return default

def _to_uint8(img):
    # Normaliza a [0,255] y convierte a uint8 
    img = img.astype(np.float32)
    minv, maxv = float(np.min(img)), float(np.max(img))
    if maxv == minv:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - minv) / (maxv - minv) * 255.0
    return out.astype(np.uint8)

def _parse_time(hhmmss: str):
    # Recibe 'HHMMSS.FFFFFF' o 'HHMMSS' y devuelve segundos
    if not hhmmss:
        return None
    s = str(hhmmss).split(".")[0]
    s = s.zfill(6)
    hh, mm, ss = int(s[0:2]), int(s[2:4]), int(s[4:6])
    return hh*3600 + mm*60 + ss

# Clase 1: EstudioImaginologico 
class EstudioImaginologico:
    """
    Guarda metadatos del estudio y la matriz 3D reconstruida.
    Implementa: zoom, segmentación, morfología y conversión a NIfTI.
    """
    def __init__(self, study_date, study_time, modality, description,
                 series_time, pixel_spacing, slice_thickness, volumen3d, nombre="estudio"):
        self.study_date = study_date
        self.study_time = study_time
        self.modality = modality
        self.description = description
        self.series_time = series_time
        self.pixel_spacing = pixel_spacing  # [row, col] mm
        self.slice_thickness = slice_thickness  # mm
        self.volumen3d = volumen3d          # numpy 3D
        self.nombre = nombre

        # Duración (serie - estudio) en segundos si es posible
        t1 = _parse_time(study_time)
        t2 = _parse_time(series_time)
        self.duracion_s = (t2 - t1) if (t1 is not None and t2 is not None) else None

    #  Punto 3b: ZOOM con OpenCV 
    def zoom(self, plano="axial", idx=None, x=30, y=30, w=50, h=50, nombre_salida="zoom"):
        """
        - plano: 'axial' (transversal), 'coronal', 'sagital'
        - idx: índice del corte. Si None, usa la mitad.
        - (x,y,w,h): rectángulo en píxeles dentro del corte.
        Guarda PNG con original y recorte redimensionado.
        """
        corte = self._get_slice(plano, idx)         # 2D
        img8 = _to_uint8(corte)                     # normaliza
        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        # Dibujar rectángulo (verde)
        pt1, pt2 = (x, y), (x + w, y + h)
        cv2.rectangle(bgr, pt1, pt2, (0, 255, 0), 2)

        # Texto con medidas en mm
        px, py = (self.pixel_spacing or [1, 1])
        w_mm = w * float(px)
        h_mm = h * float(py)
        texto = f"{w_mm:.1f}mm x {h_mm:.1f}mm"
        cv2.putText(bgr, texto, (x, max(10, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Recorte y resize (2x como ejemplo)
        recorte = img8[y:y+h, x:x+w]
        if recorte.size == 0:
            raise ValueError("El rectángulo de zoom está fuera del corte.")
        rec_resized = cv2.resize(recorte, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

        # Mostrar y guardar (dos subplots)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cmap='gray')
        axs[0].set_title("Original con ROI")
        axs[0].axis('off')
        axs[1].imshow(rec_resized, cmap='gray')
        axs[1].set_title("Recorte redimensionado")
        axs[1].axis('off')

        out = RESULTS_DIR / f"{nombre_salida}_{self.nombre}_{plano}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close(fig)
        return out

    # Punto 3c: Segmentación 
    def segmentar(self, plano="axial", idx=None, metodo="binary", umbral=127, maxval=255, nombre_salida="segmentacion"):
        """
        metodo: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv'
        """
        corte = self._get_slice(plano, idx)
        img8 = _to_uint8(corte)

        tipos = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
        }
        t = tipos.get(metodo, cv2.THRESH_BINARY)
        _, seg = cv2.threshold(img8, int(umbral), int(maxval), t)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img8, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
        axs[1].imshow(seg, cmap='gray');  axs[1].set_title(f"Segmentación: {metodo}"); axs[1].axis('off')

        out = RESULTS_DIR / f"{nombre_salida}_{self.nombre}_{plano}_{metodo}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close(fig)
        return out

    # Punto 3d: Morfología 
    def morfologia(self, plano="axial", idx=None, op="open", k=3, nombre_salida="morfologia"):
        """
        op: 'erode','dilate','open','close','grad','tophat','blackhat'
        k : tamaño de kernel cuadrado
        """
        corte = self._get_slice(plano, idx)
        img8 = _to_uint8(corte)
        kernel = np.ones((int(k), int(k)), np.uint8)

        if op == "erode":
            out_img = cv2.erode(img8, kernel, iterations=1)
        elif op == "dilate":
            out_img = cv2.dilate(img8, kernel, iterations=1)
        else:
            ops = {
                "open": cv2.MORPH_OPEN,
                "close": cv2.MORPH_CLOSE,
                "grad": cv2.MORPH_GRADIENT,
                "tophat": cv2.MORPH_TOPHAT,
                "blackhat": cv2.MORPH_BLACKHAT,
            }
            out_img = cv2.morphologyEx(img8, ops.get(op, cv2.MORPH_OPEN), kernel)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img8, cmap='gray'); axs[0].set_title("Original"); axs[0].axis('off')
        axs[1].imshow(out_img, cmap='gray'); axs[1].set_title(f"Morfología: {op} (k={k})"); axs[1].axis('off')

        out = RESULTS_DIR / f"{nombre_salida}_{self.nombre}_{plano}_{op}_k{k}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close(fig)
        return out

    # Punto 3e: Convertir a NIfTI 
    def convertir_a_nifti(self, nombre_archivo=None):
        if nombre_archivo is None:
            nombre_archivo = f"{self.nombre}.nii"
        affine = np.eye(4)  # sencillo: identidad
        img = nib.Nifti1Image(self.volumen3d.astype(np.float32), affine)
        out = NIFTI_DIR / nombre_archivo
        nib.save(img, str(out))
        return out

    # Helper: obtener un corte 2D 
    def _get_slice(self, plano="axial", idx=None):
        vol = self.volumen3d
        if vol.ndim != 3:
            raise ValueError("El volumen 3D no es válido.")
        sx, sy, sz = vol.shape  # ojo al orden

        if plano == "coronal":
            if idx is None: idx = sy // 2
            idx = int(np.clip(idx, 0, sy - 1))
            return vol[:, idx, :]
        elif plano == "sagital":
            if idx is None: idx = sx // 2
            idx = int(np.clip(idx, 0, sx - 1))
            return vol[idx, :, :]
        else:  # axial (transversal)
            if idx is None: idx = sz // 2
            idx = int(np.clip(idx, 0, sz - 1))
            return vol[:, :, idx]

# Clase 2: GestorDICOM
class GestorDICOM:
    """
    Se encarga de:
    - Cargar carpeta DICOM y reconstruir 3D.
    - Mostrar 3 cortes.
    - Extraer dataelements y crear EstudioImaginologico.
    - Guardar CSV con la info básica del estudio.
    - Almacena los objetos creados en memoria.
    """
    def __init__(self):
        self.estudios = []  # almacén simple

    #  Punto 2: reconstrucción 3D 
    def cargar_carpeta(self, carpeta: Path, nombre="estudio"):
        carpeta = Path(carpeta)
        files = sorted(carpeta.rglob("*.dcm"))
        if not files:
            raise FileNotFoundError("No se encontraron archivos .dcm en la carpeta.")

        # Leer todos los DICOMS
        dsets = [pydicom.dcmread(str(f), force=True) for f in files]

        # Ordenar por ImagePositionPatient (z) o por InstanceNumber como respaldo
        def _orden(ds):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                return int(_safe_get(ds, "InstanceNumber", 0))
        dsets.sort(key=_orden)

        # Construir volumen 3D (stack)
        slices = [ds.pixel_array.astype(np.float32) for ds in dsets]
        vol = np.stack(slices, axis=-1)  # (rows, cols, num_slices)

        # Metadatos principales (del primer DICOM)
        ds0 = dsets[0]
        pxsp = _safe_get(ds0, "PixelSpacing", [1.0, 1.0])
        sltk = float(_safe_get(ds0, "SliceThickness", 1.0))

        study_date = _safe_get(ds0, "StudyDate", "")
        study_time = _safe_get(ds0, "StudyTime", "")
        modality = _safe_get(ds0, "Modality", "")
        descr = _safe_get(ds0, "StudyDescription", "")
        series_time = _safe_get(ds0, "SeriesTime", "")

        est = EstudioImaginologico(study_date, study_time, modality, descr,
                                   series_time, pxsp, sltk, vol, nombre=nombre)
        self.estudios.append(est)
        return est

    # Punto 2: mostrar 3 vistas 
    def mostrar_tres_cortes(self, estudio: EstudioImaginologico, nombre_salida="vistas3"):
        vol = estudio.volumen3d
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))

        axial   = estudio._get_slice("axial")
        coronal = estudio._get_slice("coronal")
        sagital = estudio._get_slice("sagital")

        axs[0].imshow(_to_uint8(coronal), cmap='gray'); axs[0].set_title("Coronal"); axs[0].axis('off')
        axs[1].imshow(_to_uint8(sagital), cmap='gray'); axs[1].set_title("Sagital"); axs[1].axis('off')
        axs[2].imshow(_to_uint8(axial),   cmap='gray'); axs[2].set_title("Transversal"); axs[2].axis('off')

        out = RESULTS_DIR / f"{nombre_salida}_{estudio.nombre}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close(fig)
        return out

    #  Punto 3: CSV con dataelements 
    def guardar_info_estudio_csv(self, estudio: EstudioImaginologico):
        info = {
            "StudyDate": estudio.study_date,
            "StudyTime": estudio.study_time,
            "StudyModality": estudio.modality,
            "StudyDescription": estudio.description,
            "SeriesTime": estudio.series_time,
            "Duracion_s": estudio.duracion_s,
            "Shape": str(estudio.volumen3d.shape),
            "PixelSpacing": str(estudio.pixel_spacing),
            "SliceThickness": estudio.slice_thickness,
            "Nombre": estudio.nombre
        }
        df = pd.DataFrame([info])
        out = DFS_DIR / f"info_{estudio.nombre}.csv"
        df.to_csv(out, index=False)
        return out
