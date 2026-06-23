import cv2
import mediapipe as mp
import json
import os
from math_utils import get_hand_angles

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

JSON_PATH = '../data/gestures.json'

def load_gestures():
    if not os.path.exists(JSON_PATH):
        return {}

    try:
        with open(JSON_PATH, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("gestures.json está vacío o corrupto. Creando uno nuevo.")
        return {}
    
def save_gesture(gestures_dict):
    # Asegurar que el directorio data/ exista
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w') as file:
        json.dump(gestures_dict, file, indent=4)
    print("Diccionario de señas actualizado.")

def run_calibrator():
    cap = cv2.VideoCapture(0)
    gestures = load_gestures()
    
    print("\n--- MODO CALIBRACIÓN INICIADO ---")
    print("1. Haz una seña frente a la cámara.")
    print("2. Presiona la tecla 'C' para capturarla.")
    print("3. Presiona 'Q' para salir.\n")

    # Nuevas variables de estado para grabar movimiento antes del while
    recording_dynamic = False
    frames_recorded = 0
    max_frames = 30 # ~1 segundo de video a 30fps
    dynamic_base_angles = []
    dynamic_path_x = []
    dynamic_path_y = []

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        current_angles = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_angles = get_hand_angles(hand_landmarks.landmark)

                # Lógica si estamos grabando una seña con movimiento
                if recording_dynamic:
                    # El punto 8 es la punta del dedo índice
                    wrist = hand_landmarks.landmark[0]
                    dynamic_path_x.append(round(wrist.x, 3))
                    dynamic_path_y.append(round(wrist.y, 3))
                    frames_recorded += 1

                    # Mostrar progreso en pantalla
                    cv2.putText(image, f"Grabando Movimiento: {frames_recorded}/{max_frames}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Finalizar la grabación automáticamente
                    if frames_recorded >= max_frames:
                        recording_dynamic = False
                        print("\nGrabación finalizada.")
                        letter = input("Ingresa la letra dinámica (ej. Z) o Enter para cancelar: ").upper()
                        
                        if letter:
                            gestures[letter] = {
                                "tipo": "dinamica",
                                "angulos_base": dynamic_base_angles,
                                "trayectoria_x": dynamic_path_x,
                                "trayectoria_y": dynamic_path_y
                            }
                            save_gesture(gestures)

                else:
                    cv2.putText(image, "Mano Detectada - 'C' Estatica | 'M' Movimiento", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Calibrador LESSA', image)
        key = cv2.waitKey(1) & 0xFF

        # Captura ESTÁTICA
        if key == ord('c') and not recording_dynamic:
            if current_angles:
                letter = input("\nIngresa la letra estática (ej. A): ").upper()
                if letter:
                    gestures[letter] = {
                        "tipo": "estatica",
                        "angulos": current_angles
                    }
                    save_gesture(gestures)

        # Iniciar captura DINÁMICA
        elif key == ord('m') and not recording_dynamic:
            if current_angles:
                print("\n Grabando movimiento...")
                recording_dynamic = True
                frames_recorded = 0
                dynamic_base_angles = current_angles
                dynamic_path_x = []
                dynamic_path_y = []

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibrator()