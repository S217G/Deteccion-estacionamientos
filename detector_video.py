import cv2
import time
from ultralytics import YOLO

def probar_con_video_optimizado():
    print("---Sistema SmartParking UBB ---")
    
    # Cargar el modelo ONNX
    try:
        
        modelo = YOLO('best.onnx') 
        print("Modelo ONNX cargado correctamente.")
    except Exception as e:
        print("Error al cargar. ¿Ejecutaste el comando de exportación a ONNX?")
        return

    ruta_video = "video_noche.mp4" 
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print(f"No se pudo abrir el video.")
        return

    # Variable para saltar frames
    frames_a_saltar = 2 

    while cap.isOpened():
        tiempo_inicio = time.time()
        
        exito, frame = cap.read()
        if not exito:
            break

        # Leer y descartar frames para acelerar la reproducción del video
        for _ in range(frames_a_saltar):
            cap.grab() # .grab() avanza el video sin gastar memoria decodificando la imagen


        frame = cv2.resize(frame, (800, 600))


        resultados = modelo.predict(source=frame, conf=0.5, imgsz=640, verbose=False)

        frame_anotado = resultados[0].plot()

        # Calcular FPS reales
        tiempo_fin = time.time()
        fps = 1.0 / (tiempo_fin - tiempo_inicio)
        
        # Mostrar datos en pantalla
        cv2.putText(frame_anotado, f"FPS (Inferencia): {int(fps)}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame_anotado, f"Formato: ONNX | Res: 640", (15, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Prueba Optimizada - SmartParking", frame_anotado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    probar_con_video_optimizado()