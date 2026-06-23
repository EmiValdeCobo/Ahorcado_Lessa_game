import cv2
import threading
import queue
import mediapipe as mp
import numpy as np
import sys
import os

# Agregar el directorio principal al path para poder importar ui_native
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui_native.hangman_renderer import HangmanRenderer
from validator import GestureValidator
from math_utils import get_hand_angles, get_hand_orientation

class LessaGameEngine:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=2)
        self.running = True
        self.state = "MENU"
        
        # --- Componentes UI ---
        self.hangman = HangmanRenderer(start_x=40, start_y=380)
        
        # --- Variables del Ahorcado ---
        self.word_to_guess = "HOLA"
        self.guessed_letters = set()
        self.errors = 0
        self.max_errors = 6
        
        # --- Variables del Menú Espacial ---
        self.hovered_button = None
        self.hover_frames = 0
        self.selection_threshold = 30 # ~1 segundo para hacer clic
        self.buttons = [
            [150, 100, 490, 180, "Modulo: Abecedario", "ABECEDARIO"],
            [150, 220, 490, 300, "Juego: Ahorcado", "AHORCADO"],
            [150, 340, 490, 420, "Salir del Programa", "SALIR"]
        ]
        # Memoria para señas dinámicas
        self.dynamic_lock_frames = 0
        self.last_dynamic_letter = None

    def vision_worker(self):
        """HILO SECUNDARIO: Procesa cámara, matemáticas y rotación global."""
        cap = cv2.VideoCapture(0)
        validator = GestureValidator(json_path='../data/gestures.json')
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        recording_dynamic = False
        dynamic_frames = 0
        dynamic_target = None
        path_x, path_y = [], []
        missing_frames = 0

        while self.running:
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            detected_letter = None
            landmarks_list = None

            if results.multi_hand_landmarks:
                missing_frames = 0 # Reiniciar contador porque vimos la mano
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_list = hand_landmarks
                
                # Extraer ángulos y orientación
                current_angles = get_hand_angles(hand_landmarks.landmark)
                # OJO: Asegúrate de tener get_hand_orientation en tu math_utils.py como configuramos para la U/H
                current_orientation = get_hand_orientation(hand_landmarks.landmark) 
                
                # 1. Evaluar letra estática
                clean_letter, distance = validator.recognize_static(current_angles, current_orientation)
                detected_letter = clean_letter
                
                # 2. SIEMPRE evaluar si es el inicio de una dinámica, incluso si ya detectó una estática
                possible_starts = validator.check_dynamic_start(current_angles)
                
                # Solo iniciamos una nueva grabación si no hay una letra dinámica congelada
                if possible_starts and not recording_dynamic and self.dynamic_lock_frames == 0:
                    recording_dynamic = True
                    dynamic_frames = 0
                    dynamic_target = possible_starts[0]
                    path_x, path_y = [], []
                    
                # 3. Lógica de trazado
                if recording_dynamic:
                    index_tip = hand_landmarks.landmark[8]
                    path_x.append(round(index_tip.x, 3))
                    path_y.append(round(index_tip.y, 3))
                    dynamic_frames += 1
                    
                    if clean_letter:
                        detected_letter = f"{clean_letter} (Trazando {dynamic_target}...)"
                    else:
                        detected_letter = f"Trazando {dynamic_target}..."

                    if dynamic_frames >= 60:
                        if validator.validate_trajectory(dynamic_target, path_x, path_y):
                            self.last_dynamic_letter = dynamic_target
                            self.dynamic_lock_frames = 30 
                        recording_dynamic = False
                        
                if self.dynamic_lock_frames > 0:
                    detected_letter = self.last_dynamic_letter
                    self.dynamic_lock_frames -= 1
                    recording_dynamic = False
            else:
                # NUEVA LÓGICA: Si no hay mano, aumentar el contador de pérdida
                missing_frames += 1
                
                # Mantener el texto en pantalla para que no parpadee
                if recording_dynamic:
                    detected_letter = f"Trazando {dynamic_target}..."
                elif self.dynamic_lock_frames > 0:
                    detected_letter = self.last_dynamic_letter
                    self.dynamic_lock_frames -= 1
                
                # Si la mano desaparece por más de 15 frames (~0.5 segundos), abortar
                if missing_frames > 15:
                    recording_dynamic = False
                    dynamic_frames = 0
                    path_x, path_y = [], []

            # Enviar datos a la cola de la interfaz
            if not self.data_queue.empty():
                try: self.data_queue.get_nowait()
                except queue.Empty: pass
                    
            self.data_queue.put({"frame": frame, "landmarks": landmarks_list, "letter": detected_letter})
        cap.release()

    # ==========================================
    # MÓDULOS MODULARIZADOS (Capa de Presentación)
    # ==========================================

    def _procesar_menu(self, frame, index_x, index_y):
        """Dibuja y maneja la lógica del menú principal"""
        cv2.putText(frame, "SELECCIONA UNA OPCION", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        button_hovered_this_frame = None

        for x1, y1, x2, y2, text, target in self.buttons:
            if x1 < index_x < x2 and y1 < index_y < y2:
                button_hovered_this_frame = target
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, text, (x1 + 20, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                progress_width = int(((x2 - x1) / self.selection_threshold) * self.hover_frames)
                cv2.rectangle(frame, (x1, y2 - 10), (x1 + progress_width, y2), (255, 0, 255), cv2.FILLED)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.putText(frame, text, (x1 + 20, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if button_hovered_this_frame:
            if self.hovered_button == button_hovered_this_frame:
                self.hover_frames += 1
                if self.hover_frames >= self.selection_threshold:
                    if button_hovered_this_frame == "SALIR":
                        self.running = False
                    else:
                        self.state = button_hovered_this_frame
                        # Reiniciar variables del ahorcado al entrar
                        if self.state == "AHORCADO":
                            self.errors = 0
                            self.guessed_letters.clear()
                    self.hover_frames = 0
            else:
                self.hovered_button = button_hovered_this_frame
                self.hover_frames = 0
        else:
            self.hovered_button, self.hover_frames = None, 0

    def _procesar_ahorcado(self, frame, current_letter, key):
        """Dibuja y maneja la lógica del juego del Ahorcado"""
        display_word = " ".join([char if char in self.guessed_letters else "_" for char in self.word_to_guess])
        cv2.putText(frame, f"Palabra: {display_word}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        status_text = f"Detectando: {current_letter}" if current_letter else "Detectando: ..."
        cv2.putText(frame, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Errores: {self.errors}/{self.max_errors}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Presiona 'M' para volver al Menu", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self.hangman.draw(frame, self.errors)

        if key == ord(' '):
            if current_letter and len(current_letter) == 1: # Ignorar textos como "Trazando..."
                if current_letter in self.word_to_guess and current_letter not in self.guessed_letters:
                    print(f"¡Correcto! {current_letter}")
                    self.guessed_letters.add(current_letter)
                elif current_letter not in self.guessed_letters:
                    print(f"¡Incorrecto! {current_letter}")
                    self.errors += 1
                    self.guessed_letters.add(current_letter)
                    
                if set(self.word_to_guess).issubset(self.guessed_letters):
                    print("¡GANASTE!")
                    self.state = "MENU"
                elif self.errors >= self.max_errors:
                    print("PERDISTE.")
                    self.state = "MENU"

        if key == ord('m'):
            self.state = "MENU"

    def _procesar_abecedario(self, frame, current_letter, key):
        """Dibuja y maneja la lógica del modo educativo"""
        cv2.putText(frame, "MODULO ABECEDARIO - EN CONSTRUCCION", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Presiona 'M' para volver al Menu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if key == ord('m'):
            self.state = "MENU"

    # ==========================================
    # BUCLE PRINCIPAL (El "Director de Tráfico")
    # ==========================================

    def run_game(self):
        vision_thread = threading.Thread(target=self.vision_worker, daemon=True)
        vision_thread.start()
        
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        while self.running:
            try:
                data = self.data_queue.get(timeout=0.1)
                frame = data["frame"]
                landmarks = data["landmarks"]
                current_letter = data["letter"]
                
                # 1. Dibujar esqueleto de la mano siempre
                if landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    
                # 2. Obtener posición del índice para el cursor (Menú)
                index_x, index_y = 0, 0
                if landmarks:
                    h, w, _ = frame.shape
                    index_x, index_y = int(landmarks.landmark[8].x * w), int(landmarks.landmark[8].y * h)
                    cv2.circle(frame, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)

                # 3. Leer teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): self.running = False

                # 4. Enrutar al módulo correspondiente según el estado actual
                if self.state == "MENU":
                    self._procesar_menu(frame, index_x, index_y)
                elif self.state == "AHORCADO":
                    self._procesar_ahorcado(frame, current_letter, key)
                elif self.state == "ABECEDARIO":
                    self._procesar_abecedario(frame, current_letter, key)

                # 5. Renderizar
                cv2.imshow('LESSA Game Engine', frame)

            except queue.Empty:
                pass

        cv2.destroyAllWindows()
        vision_thread.join()

if __name__ == "__main__":
    game = LessaGameEngine()
    game.run_game()