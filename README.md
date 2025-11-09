#  Programa de Bioingeniería – Parcial 3  
### Asignatura: Informática II – Semestre 2025-2  
**Tema:** Extracción de datos y procesamiento de imágenes médicas DICOM
## Juan Diego Quintero Vanegas - Paula Alejandra Hios  


## Descripción general

Este proyecto implementa un algoritmo en **Python** orientado a objetos (POO) para la **gestión, procesamiento y análisis de imágenes médicas DICOM**, cumpliendo con los requerimientos del **Parcial 3** de la asignatura *Informática II*.

El programa permite:
- Cargar carpetas con archivos DICOM (`PPMI`, `Sarcoma`, `T2`, `series`).
- Reconstruir un volumen 3D a partir de cortes tomográficos.
- Visualizar los tres planos principales: **coronal, sagital y transversal**.
- Extraer los metadatos de los estudios y guardarlos en un **DataFrame (CSV)**.
- Aplicar operaciones de **zoom, segmentación y transformaciones morfológicas** usando OpenCV.
- Convertir el volumen reconstruido a formato **NIfTI (.nii)**.
- Gestionar múltiples estudios mediante un **menú interactivo en consola**.


## Estructura del proyecto

P3_JDiego_Paula/
│
├── clases.py # Clases principales: GestorDICOM y EstudioImaginologico
├── main.py # Menú principal del programa
├── config.py # Configuración de rutas (data, resultados, csv, nifti)
├── requirements.txt # Librerías requeridas
├── README.md # Este documento
│
├── resultados/ # Imágenes generadas (vistas, zoom, segmentación, morfología)
├── dataframes/ # CSV con información de los estudios
├── nifti/ # Archivos NIfTI exportados (.nii)
└── data/ # Carpeta local para datos DICOM (vacía en la entrega)
├── series/ # (Opcional) conjunto pequeño de prueba
└── README.txt # Explica dónde ubicar PPMI, Sarcoma y T2


##  Instalación y ejecución

### 1️ Requisitos previos
Tener instalado **Python 3.10+** y `pip`.

### 2️ Instalar dependencias
En una terminal ubicada dentro del proyecto:
pip install -r requirements.txt

## Colocar los datos DICOM
Ubicar dentro de la carpeta data/ las carpetas:
PPMI/
Sarcoma/
T2/
series/

## Ejecutar el programa
python main.py

##  Menú principal

| Opción | Descripción |
|:-------|:-------------|
| 1 | Cargar carpeta DICOM y crear un objeto de estudio |
| 2 | Mostrar los 3 cortes (coronal, sagital y transversal) |
| 3 | Guardar CSV con la información del estudio |
| 4 | Aplicar Zoom (recorte, resize y cuadro con dimensiones en mm) |
| 5 | Aplicar Segmentación (tipo binario, truncado, tozero, etc.) |
| 6 | Aplicar Transformación morfológica (open, close, dilate, etc.) |
| 7 | Convertir el estudio a formato NIfTI (.nii) |
| 8 | Listar estudios cargados |
| 0 | Salir del programa |
