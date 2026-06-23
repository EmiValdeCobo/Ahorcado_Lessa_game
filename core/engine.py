import cv2
import threading
import queue
import mediapipe as mp
import numpy as np
import sys
import os
import random
import json

# Agregar el directorio principal al path para poder importar ui_native
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui_native.hangman_renderer import HangmanRenderer
from validator import GestureValidator
from math_utils import get_hand_angles, get_hand_orientation, get_finger_spread

class LessaGameEngine:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=2)
        self.running = True
        self.state = "MENU"
        self.hangman = HangmanRenderer(start_x=40, start_y=380)
        
        self.word_bank = self._cargar_palabras()
        self.word_to_guess = ""
        self.guessed_letters = set()
        self.errors = 0
        self.max_errors = 6
        self.score = 0
        self.player_name = ""
        
        # --- Variables del Menú Espacial (ACTUALIZADAS) ---
        self.hovered_button = None
        self.hover_frames = 0
        self.selection_threshold = 30
        self.buttons = [
            [100, 80, 540, 150, "Modulo: Abecedario (Playground)", "ABECEDARIO"],
            [100, 180, 540, 250, "Juego: Ahorcado", "AHORCADO"],
            [100, 280, 540, 350, "Clasificacion (Ranking)", "RANKING"],
            [100, 380, 540, 450, "Salir del Programa", "SALIR"]
        ]
        
        self.dynamic_lock_frames = 0
        self.last_dynamic_letter = None
        
    def _cargar_palabras(self):
        """Carga el banco de palabras desde el JSON. Si no existe, lo crea."""
        ruta_words = '../data/words.json'
        if os.path.exists(ruta_words):
            with open(ruta_words, 'r', encoding='utf-8') as file:
                datos = json.load(file)
                return datos.get("palabras", ["HOLA"])
        else:
            # Crear archivo por defecto si no existe
            palabras_default = {"palabras": ["LESSA", "HOLA", "MUNDO"]}
            os.makedirs(os.path.dirname(ruta_words), exist_ok=True)
            with open(ruta_words, 'w', encoding='utf-8') as file:
                json.dump(palabras_default, file, indent=4)
            return palabras_default["palabras"]
        
    def _iniciar_partida(self, mantener_puntaje=False):
        """Selecciona una palabra aleatoria y reinicia las variables del juego."""
        self.word_to_guess = random.choice(self.word_bank)
        self.guessed_letters.clear()
        self.errors = 0
        
        if not mantener_puntaje:
            self.score = 0 
            
        self.state = "AHORCADO"

    def _guardar_puntaje(self):
        """Guarda el nombre y la puntuación en un archivo JSON."""
        ruta_ranking = '../data/ranking.json'
        ranking = []
        if os.path.exists(ruta_ranking):
            with open(ruta_ranking, 'r') as file:
                ranking = json.load(file)
                
        ranking.append({"nombre": self.player_name, "puntos": self.score})
        # Ordenar de mayor a menor puntuación
        ranking = sorted(ranking, key=lambda x: x["puntos"], reverse=True)[:5] # Guardar solo el Top 5
        
        os.makedirs(os.path.dirname(ruta_ranking), exist_ok=True)
        with open(ruta_ranking, 'w') as file:
            json.dump(ranking, file, indent=4)
        
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
                
                # --- NUEVA LÓGICA: INTERCEPTOR U vs V ---
                if clean_letter in ['U', 'V', 'U1', 'V1', 'U2', 'V2']:
                    spread_ratio = get_finger_spread(hand_landmarks.landmark)
                    # Si el ratio es mayor a 0.35, los dedos están muy separados (Es V)
                    if spread_ratio > 0.35:
                        clean_letter = 'V'
                    else:
                        clean_letter = 'U'
                # ----------------------------------------
                detected_letter = clean_letter
                
                # 2. SIEMPRE evaluar si es el inicio de una dinámica, incluso si ya detectó una estática
                possible_starts = validator.check_dynamic_start(current_angles, current_orientation)      
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

                    if dynamic_frames >= 60: # 60 frames = 2 segundos (Cámbialo a 90 si quieres exactamente 3 segundos)
                        if validator.validate_trajectory(dynamic_target, path_x, path_y):
                            # Si el trazo fue correcto, gana la letra dinámica (Ej. Ñ)
                            self.last_dynamic_letter = dynamic_target
                            self.dynamic_lock_frames = 30 
                        else:
                            # Si pasó el tiempo, no hubo movimiento, 
                            # y la cámara está viendo una letra estática (Ej. A o N), GANA LA ESTÁTICA.
                            if clean_letter:
                                self.last_dynamic_letter = clean_letter
                                # Congelamos la letra ganadora por 45 frames (1.5 segundos) 
                                # para que desaparezca el "Trazando..." y puedas presionar la barra espaciadora en paz.
                                self.dynamic_lock_frames = 45 
                                
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
                    elif button_hovered_this_frame == "AHORCADO":
                        self._iniciar_partida() 
                    else:
                        self.state = button_hovered_this_frame
                    self.hover_frames = 0
            else:
                self.hovered_button = button_hovered_this_frame
                self.hover_frames = 0
        else:
            self.hovered_button, self.hover_frames = None, 0

    def _procesar_ahorcado(self, frame, current_letter, key):
        """Dibuja y maneja la lógica del juego del Ahorcado con visuales mejorados"""
        
        # --- 1. Dibujar la palabra a adivinar (Más grande y centrada) ---
        display_word = " ".join([char if char in self.guessed_letters else "_" for char in self.word_to_guess])
        display_word_safe = display_word.replace('Ñ', 'N~')
        
        # Sombra negra para dar relieve al texto
        cv2.putText(frame, display_word_safe, (152, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 5)
        # Texto principal en amarillo brillante
        cv2.putText(frame, display_word_safe, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 5)
        
        # --- 2. Panel de Información lateral ---
        status_text = f"Detectando: {current_letter}" if current_letter else "Haz una seña..."
        status_text_safe = status_text.replace('Ñ', 'N~')
        
        # Letra actual detectada
        cv2.putText(frame, status_text_safe, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Errores en rojo
        cv2.putText(frame, f"Errores: {self.errors}/{self.max_errors}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Puntos actuales
        cv2.putText(frame, f"Puntos: {self.score}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instrucción para salir
        cv2.putText(frame, "Presiona 'M' para volver al Menu", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Dibujar el muñeco
        self.hangman.draw(frame, self.errors)

        if key == ord(' '):
            if current_letter and len(current_letter) == 1:
                if current_letter in self.word_to_guess and current_letter not in self.guessed_letters:
                    self.guessed_letters.add(current_letter)
                    self.score += 10 # +10 puntos por letra correcta
                elif current_letter not in self.guessed_letters:
                    self.errors += 1
                    self.guessed_letters.add(current_letter)
                    self.score = max(0, self.score - 5) # -5 puntos por error
                    
                if set(self.word_to_guess).issubset(self.guessed_letters):
                    # Bono por ganar basado en vidas restantes
                    self.score += (self.max_errors - self.errors) * 20
                    self.state = "VICTORIA"
                elif self.errors >= self.max_errors:
                    self.state = "DERROTA"

        if key == ord('m'):
            self.state = "MENU"

    def _procesar_abecedario(self, frame, current_letter, key):
        """Modo libre (Playground). Interfaz visual corregida."""
        cv2.putText(frame, "PLAYGROUND LESSA - Practica libre", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        if current_letter:
            # 1. Limpiar caracteres incompatibles con OpenCV
            safe_letter = current_letter.replace('Ñ', 'N~')
            
            # 2. Lógica visual responsiva
            if "Trazando" in safe_letter:
                # Si está evaluando un movimiento, dibujar el texto mediano
                cv2.putText(frame, safe_letter, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            else:
                # Si es una letra final, dibujarla masiva y centrada
                cv2.putText(frame, safe_letter, (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 12)
        else:
            cv2.putText(frame, "?", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (100, 100, 100), 12)

        cv2.putText(frame, "Presiona 'M' para volver al Menu", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if key == ord('m'):
            self.state = "MENU"
            
    def _procesar_victoria(self, frame, key):
        cv2.putText(frame, "¡GANASTE!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        cv2.putText(frame, f"Puntuacion Final: {self.score}", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona 'ESPACIO' para volver a jugar", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Presiona 'R' para registrar tu nombre", (80, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Presiona 'M' para salir al Menu", (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if key == ord(' '): 
            self._iniciar_partida(mantener_puntaje=True) # <-- AQUÍ SE MANTIENE TU RACHA
        elif key == ord('r'): 
            self.player_name = ""
            self.state = "REGISTRO"
        elif key == ord('m'): 
            self.state = "MENU"

    def _procesar_derrota(self, frame, key):
        cv2.putText(frame, "¡PERDISTE!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, f"La palabra era: {self.word_to_guess}", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona 'ESPACIO' para volver a jugar", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Presiona 'M' para salir al Menu", (80, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if key == ord(' '): 
            self._iniciar_partida(mantener_puntaje=False) 
        elif key == ord('m'): 
            self.state = "MENU"

    def _procesar_registro(self, frame, current_letter, key):
        cv2.putText(frame, "DELETREA TU NOMBRE EN LESSA", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Nombre actual: {self.player_name}_", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        status = f"Detectando: {current_letter}" if current_letter else "Haz una seña..."
        cv2.putText(frame, status, (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, "[ESPACIO] Agregar Letra | [B] Borrar ultima | [ENTER] Guardar", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if key == ord(' '):
            if current_letter and len(current_letter) == 1:
                self.player_name += current_letter
        elif key == ord('b'):
            self.player_name = self.player_name[:-1]
        elif key == 13: # Código ASCII para ENTER
            if self.player_name.strip() != "":
                self._guardar_puntaje()
                self.state = "RANKING"

    def _procesar_ranking(self, frame, key):
        cv2.putText(frame, "TOP 5 - MEJORES PUNTAJES", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        ruta_ranking = '../data/ranking.json'
        if os.path.exists(ruta_ranking):
            with open(ruta_ranking, 'r') as file:
                ranking = json.load(file)
            
            y_offset = 120
            for i, p in enumerate(ranking):
                texto = f"{i+1}. {p['nombre']} - {p['puntos']} pts"
                cv2.putText(frame, texto, (150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                y_offset += 50
        else:
            cv2.putText(frame, "Aun no hay puntajes registrados.", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.putText(frame, "Presiona 'M' para volver al Menu", (120, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if key == ord('m'): self.state = "MENU"

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
                elif self.state == "VICTORIA":
                    self._procesar_victoria(frame, key)
                elif self.state == "DERROTA":
                    self._procesar_derrota(frame, key)
                elif self.state == "REGISTRO":
                    self._procesar_registro(frame, current_letter, key)
                elif self.state == "RANKING":
                    self._procesar_ranking(frame, key)

                # 5. Renderizar
                cv2.imshow('LESSA Game Engine', frame)

            except queue.Empty:
                pass

        cv2.destroyAllWindows()
        vision_thread.join()

if __name__ == "__main__":
    game = LessaGameEngine()
    game.run_game()