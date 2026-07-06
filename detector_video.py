import cv2
import time
import json
import numpy as np
import requests
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as ShapelyBox

def tiene_soporte_gui_opencv():
    try:
        return "GUI:                            NONE" not in cv2.getBuildInformation()
    except Exception:
        return False

def iniciar_sistema_core():
    try:
        with open("./Proyecto/zonasubb.json", "r") as f:
            zonas_guardadas = json.load(f)
        print(f"Zonas cargadas correctamente: {list(zonas_guardadas.keys())}")
    except FileNotFoundError:
        print("No se encontro el archivo .json. Ejecuta el mapeador primero.")
        return

    try:
        modelo = YOLO('./Proyecto/best.pt') 
        print("Modelo YOLO cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return

    ruta_video = "./Proyecto/video_parkingubb.mp4"
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    frames_a_saltar = 2 
    UMBRAL_COBERTURA = 0.15
    TIEMPO_ESPERA = 3.0
    soporte_gui = tiene_soporte_gui_opencv()
    
    if not soporte_gui:
        print("OpenCV fue compilado sin soporte de GUI. Se omite la ventana de visualizacion.")

    estado_oficial = {nombre: "DISPONIBLE" for nombre in zonas_guardadas.keys()}
    temporizadores = {nombre: None for nombre in zonas_guardadas.keys()}
    ultimo_estado_enviado = None

    while cap.isOpened():
        tiempo_inicio = time.time()
        
        exito, frame = cap.read()
        if not exito:
            print("Video finalizado.")
            break

        for _ in range(frames_a_saltar):
            cap.grab()

        frame = cv2.resize(frame, (1024, 768))

        deteccion_frame_actual = {nombre: False for nombre in zonas_guardadas.keys()}
        resultados = modelo.predict(source=frame, conf=0.4, imgsz=640, verbose=False)
        frame_anotado = frame.copy()
        
        cajas = resultados[0].boxes
        
        for nombre_plaza, puntos_plaza in zonas_guardadas.items():
            poligono_plaza = ShapelyPolygon(puntos_plaza)
            area_plaza = poligono_plaza.area
            
            for caja in cajas:
                x1, y1, x2, y2 = caja.xyxy[0].cpu().numpy()

                # --- OPCIÓN 1: TÉCNICA DE LA HUELLA (Recorte Inferior) ---
                ancho = x2 - x1
                alto = y2 - y1
                hx1 = x1 + (ancho * 0.15)
                hx2 = x2 - (ancho * 0.15)
                hy1 = y2 - (alto * 0.30)
                hy2 = y2
                
                caja_auto = ShapelyBox(hx1, hy1, hx2, hy2)
                cv2.rectangle(frame_anotado, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 255, 0), 2)

                # --- OPCIÓN 2: CAJA COMPLETA (Sugerencia del profesor) ---
                # Para usar la caja completa, comenta las 9 líneas de arriba y descomenta estas dos:
                #caja_auto = ShapelyBox(x1, y1, x2, y2)
                #cv2.rectangle(frame_anotado, (int(x1), int(y1)), (int(x2), int(y2)), (255, 150, 0), 2)
                
                if poligono_plaza.intersects(caja_auto):
                    area_choque = poligono_plaza.intersection(caja_auto).area
                    porcentaje_cobertura = area_choque / area_plaza
                    
                    if porcentaje_cobertura > UMBRAL_COBERTURA:
                        deteccion_frame_actual[nombre_plaza] = True
                        break 

        tiempo_actual = time.time()
        
        for nombre in zonas_guardadas.keys():
            estado_actual = estado_oficial[nombre]
            hay_auto_ahora = deteccion_frame_actual[nombre]

            if estado_actual == "DISPONIBLE" and hay_auto_ahora:
                if temporizadores[nombre] is None:
                    temporizadores[nombre] = tiempo_actual 
                elif (tiempo_actual - temporizadores[nombre]) >= TIEMPO_ESPERA:
                    estado_oficial[nombre] = "OCUPADO" 
                    temporizadores[nombre] = None
            
            elif estado_actual == "OCUPADO" and not hay_auto_ahora:
                if temporizadores[nombre] is None:
                    temporizadores[nombre] = tiempo_actual 
                elif (tiempo_actual - temporizadores[nombre]) >= TIEMPO_ESPERA:
                    estado_oficial[nombre] = "DISPONIBLE" 
                    temporizadores[nombre] = None
            
            else:
                temporizadores[nombre] = None

        for nombre_plaza, puntos_plaza in zonas_guardadas.items():
            contorno = np.array(puntos_plaza, dtype=np.int32).reshape((-1, 1, 2))
            
            if estado_oficial[nombre_plaza] == "OCUPADO":
                color_linea = (0, 0, 255) 
            else:
                color_linea = (0, 255, 0) 
            
            cv2.polylines(frame_anotado, [contorno], isClosed=True, color=color_linea, thickness=3)
            
            cx_texto = int(sum([p[0] for p in puntos_plaza]) / 4)
            cy_texto = int(sum([p[1] for p in puntos_plaza]) / 4)
            cv2.putText(frame_anotado, nombre_plaza, (cx_texto - 15, cy_texto), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_linea, 2)
            
            if temporizadores[nombre_plaza] is not None:
                progreso = min(1.0, (tiempo_actual - temporizadores[nombre_plaza]) / TIEMPO_ESPERA)
                cv2.putText(frame_anotado, f"... {int(progreso*100)}%", (cx_texto - 20, cy_texto + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        tiempo_fin = time.time()
        fps = 1.0 / (tiempo_fin - tiempo_inicio)
        cv2.putText(frame_anotado, f"FPS: {int(fps)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- ENVÍO DE DATOS OPTIMIZADO POR EVENTO ---
        if ultimo_estado_enviado is None or estado_oficial != ultimo_estado_enviado:
            url_backend = "http://localhost:8000/actualizar_estado"
            paquete_json = {
                "estado_plazas": estado_oficial
            }
            
            try:
                respuesta = requests.post(url_backend, json=paquete_json, timeout=0.1)
                if respuesta.status_code == 200:
                    ultimo_estado_enviado = estado_oficial.copy()
                    print(f"Cambio detectado. Transmisión exitosa: {estado_oficial}\n")
            except requests.exceptions.RequestException:
                pass

        if soporte_gui:
            cv2.imshow("SmartParking UBB - Motor de Deteccion", frame_anotado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if soporte_gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_sistema_core()