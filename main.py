# main.py
from pathlib import Path
from clases import GestorDICOM
from config import DATA_DIR

def pedir_carpeta():
    print("\nCarpetas disponibles dentro de /data:")
    for p in DATA_DIR.iterdir():
        if p.is_dir():
            print(" -", p.name)
    nombre = input("Nombre de la carpeta a cargar (ej: PPMI, Sarcoma, T2 o series): ").strip()
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        print("Esa carpeta no existe.")
        return None
    return ruta

def pedir_plano():
    p = input("Plano [axial/transversal | coronal | sagital]: ").strip().lower()
    return "axial" if p in ("", "axial", "transversal") else p

def menu():
    print("\n--- MENÚ PARCIAL 3 ---")
    print("1) Cargar carpeta DICOM y crear Estudio")
    print("2) Mostrar 3 cortes (coronal, sagital, transversal)")
    print("3) Guardar CSV con info del estudio")
    print("4) ZOOM (recortar + resize + cuadro)")
    print("5) Segmentación (threshold)")
    print("6) Morfología (kernel)")
    print("7) Convertir a NIFTI")
    print("8) Listar estudios en memoria")
    print("0) Salir")

if __name__ == "__main__":
    gestor = GestorDICOM()
    ultimo = None  # referencia al estudio más reciente

    while True:
        menu()
        op = input("Opción: ").strip()

        if op == "1":
            carpeta = pedir_carpeta()
            if carpeta:
                nombre = input("Nombre corto para el estudio (ej: ppmi_01): ").strip() or "estudio"
                try:
                    ultimo = gestor.cargar_carpeta(Path(carpeta), nombre=nombre)
                    print(f"✓ Estudio creado: {ultimo.nombre}, shape={ultimo.volumen3d.shape}")
                except Exception as e:
                    print("Error cargando carpeta:", e)

        elif op == "2":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                ruta = gestor.mostrar_tres_cortes(ultimo)
                print("Imagen guardada en:", ruta)

        elif op == "3":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                ruta = gestor.guardar_info_estudio_csv(ultimo)
                print("CSV guardado en:", ruta)

        elif op == "4":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                plano = pedir_plano()
                try:
                    idx = input("Índice de corte (Enter=medio): ").strip()
                    idx = None if idx == "" else int(idx)
                    x = int(input("x (px): "))
                    y = int(input("y (px): "))
                    w = int(input("ancho (px): "))
                    h = int(input("alto (px): "))
                    out = ultimo.zoom(plano=plano, idx=idx, x=x, y=y, w=w, h=h,
                                      nombre_salida=f"zoom_{ultimo.nombre}")
                    print("Zoom guardado en:", out)
                except Exception as e:
                    print("Error en zoom:", e)

        elif op == "5":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                plano = pedir_plano()
                metodo = input("Método [binary/binary_inv/trunc/tozero/tozero_inv]: ").strip() or "binary"
                try:
                    out = ultimo.segmentar(plano=plano, metodo=metodo,
                                           nombre_salida=f"seg_{ultimo.nombre}")
                    print("Segmentación guardada en:", out)
                except Exception as e:
                    print("Error en segmentación:", e)

        elif op == "6":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                plano = pedir_plano()
                opm = input("Operación [erode/dilate/open/close/grad/tophat/blackhat]: ").strip() or "open"
                k = int(input("Tamaño de kernel (ej: 3): "))
                try:
                    out = ultimo.morfologia(plano=plano, op=opm, k=k,
                                            nombre_salida=f"morf_{ultimo.nombre}")
                    print("Morfología guardada en:", out)
                except Exception as e:
                    print("Error en morfología:", e)

        elif op == "7":
            if not ultimo:
                print("Primero carga un estudio (opción 1).")
            else:
                try:
                    out = ultimo.convertir_a_nifti(nombre_archivo=f"{ultimo.nombre}.nii")
                    print("NIfTI guardado en:", out)
                except Exception as e:
                    print("Error en NIfTI:", e)

        elif op == "8":
            if not gestor.estudios:
                print("No hay estudios cargados.")
            else:
                for i, e in enumerate(gestor.estudios, 1):
                    print(f"{i}) {e.nombre}  shape={e.volumen3d.shape}  duracion_s={e.duracion_s}")

        elif op == "0":
            print("¡Listo! Saliendo.")
            break
        else:
            print("Opción inválida.")
