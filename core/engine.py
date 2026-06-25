import cv2
import threading
import queue
import mediapipe as mp
import numpy as np
import sys
import os
import random
import json
import time
import logging

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='[LESSA] %(levelname)s %(asctime)s – %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('lessa')

# ─── Resolución de rutas robusta para dev y PyInstaller ──────────────────────

def _get_data_path(filename: str) -> str:
    """
    Resuelve la ruta al directorio /data/ tanto en desarrollo como
    en el ejecutable compilado (PyInstaller one-folder con COLLECT).

    En dev:        <proyecto>/data/<filename>
    Compilado:     <dist/LESSA_Game>/data/<filename>
    """
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base, 'data', filename)


# ─── Importaciones del proyecto ───────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui_native.hangman_renderer import HangmanRenderer
from validator import GestureValidator
from math_utils import get_hand_angles, get_hand_orientation, get_finger_spread


# ─── Paleta de colores (BGR) ──────────────────────────────────────────────────
C_WHITE  = (255, 255, 255)
C_GRAY   = (180, 180, 180)
C_DARK   = ( 20,  20,  20)
C_YELLOW = (  0, 220, 255)   # Dorado / Amarillo
C_GREEN  = ( 60, 210,  60)   # Verde éxito
C_RED    = ( 60,  60, 220)   # Rojo error
C_CYAN   = (220, 220,   0)   # Cyan info
C_ORANGE = (  0, 160, 255)   # Naranja
C_PURPLE = (200,  80, 180)   # Magenta suave

# Color de accent por botón del menú
BTN_ACCENTS = [C_GREEN, C_YELLOW, C_CYAN, C_RED]
BTN_LABELS  = [
    "[PLAY]  Playground  -  Practica el abecedario LESSA",
    "[GAME]  Ahorcado    -  Juega y acumula puntos",
    "[TOP5]  Ranking     -  Los mejores jugadores",
    "[EXIT]  Salir       -  Cerrar el programa",
]

# Fuente única usada en toda la app (evita el string literal repetido)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
class LessaGameEngine:
# =============================================================================

    def __init__(self):
        self.data_queue  = queue.Queue(maxsize=2)
        self.running     = True
        self.state       = "MENU"
        self.hangman     = HangmanRenderer(start_x=40, start_y=380)

        # Estado del juego
        self.word_bank        = self._cargar_palabras()
        self.word_to_guess    = ""
        self.guessed_letters  = set()
        self.errors           = 0
        self.max_errors       = 6
        self.score            = 0
        self.player_name      = ""

        # Menú
        self.hovered_button      = None
        self.hover_frames        = 0
        self.selection_threshold = 30
        self.buttons = [
            [ 70, 115, 570, 185, "Playground",  "ABECEDARIO"],
            [ 70, 205, 570, 275, "Ahorcado",    "AHORCADO"  ],
            [ 70, 295, 570, 365, "Ranking",     "RANKING"   ],
            [ 70, 385, 570, 455, "Salir",       "SALIR"     ],
        ]

        # Control dinámico
        self.dynamic_lock_frames = 0
        self.last_dynamic_letter = None

        # Cache del ranking (evita leer el JSON en cada frame)
        self._ranking_cache = None

        # FPS counter
        self._fps_clock   = time.time()
        self._fps_count   = 0
        self._fps_display = 0

    # ─── Helpers de datos ─────────────────────────────────────────────────────

    def _cargar_palabras(self) -> list:
        ruta = _get_data_path('words.json')
        try:
            if os.path.exists(ruta):
                with open(ruta, 'r', encoding='utf-8') as f:
                    return json.load(f).get("palabras", ["LESSA"])
            default = {"palabras": ["LESSA", "HOLA", "MUNDO"]}
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(default, f, indent=4)
            return default["palabras"]
        except (OSError, json.JSONDecodeError) as e:
            log.error("Error cargando words.json: %s", e)
            return ["LESSA", "HOLA", "MUNDO"]

    def _iniciar_partida(self, mantener_puntaje: bool = False):
        self.word_to_guess   = random.choice(self.word_bank)
        self.guessed_letters.clear()
        self.errors = 0
        if not mantener_puntaje:
            self.score = 0
        self.state = "AHORCADO"

    def _guardar_puntaje(self):
        ruta = _get_data_path('ranking.json')
        try:
            ranking = []
            if os.path.exists(ruta):
                with open(ruta, 'r', encoding='utf-8') as f:
                    ranking = json.load(f)
            ranking.append({"nombre": self.player_name, "puntos": self.score})
            ranking = sorted(ranking, key=lambda x: x["puntos"], reverse=True)[:5]
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(ranking, f, indent=4)
            self._ranking_cache = ranking   # actualizar cache en memoria
        except (OSError, json.JSONDecodeError, KeyError) as e:
            log.error("Error guardando puntaje: %s", e)

    def _cargar_ranking(self) -> list:
        """Lee el ranking desde cache; solo toca el disco la primera vez."""
        if self._ranking_cache is None:
            ruta = _get_data_path('ranking.json')
            try:
                self._ranking_cache = (
                    json.load(open(ruta, encoding='utf-8'))
                    if os.path.exists(ruta) else []
                )
            except (OSError, json.JSONDecodeError) as e:
                log.error("Error cargando ranking.json: %s", e)
                self._ranking_cache = []
        return self._ranking_cache

    # ─── Helpers visuales (OpenCV) ────────────────────────────────────────────

    @staticmethod
    def _panel(frame, x1: int, y1: int, x2: int, y2: int,
               color: tuple = C_DARK, alpha: float = 0.6):
        """Rectángulo semi-transparente superpuesto sobre el frame."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _text_shadow(frame, text: str, pos: tuple, font, scale: float,
                     color: tuple, thickness: int):
        """Texto con sombra negra desplazada para mejorar legibilidad en cámara."""
        sx, sy = pos[0] + 2, pos[1] + 2
        cv2.putText(frame, text, (sx, sy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos,      font, scale, color,     thickness,     cv2.LINE_AA)

    @staticmethod
    def _progress_bar(frame, x1: int, y1: int, x2: int, y2: int,
                      progress: float, color: tuple, bg: tuple = (50, 50, 50)):
        """Barra de progreso horizontal. progress: 0.0 → 1.0."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, cv2.FILLED)
        fill_w = int((x2 - x1) * min(max(progress, 0.0), 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), color, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

    # ─── NUEVO: utilidades de ajuste de texto responsivo ──────────────────────

    @staticmethod
    def _fit_scale(text: str, font, base_scale: float,
                   thickness: int, max_width: int, margin: int = 2) -> float:
        """
        Calcula la escala necesaria para que 'text' quepa en 'max_width' px.
        Nunca supera base_scale; retorna base_scale si ya cabe.
        'margin' añade un colchón de seguridad para absorber el redondeo
        de punto flotante de OpenCV al renderizar con la escala calculada.
        """
        if not text or max_width <= 0:
            return base_scale
        (tw, _), _ = cv2.getTextSize(text, font, base_scale, thickness)
        if tw <= 0:
            return base_scale
        effective_max = max(1, max_width - margin)
        return base_scale * (effective_max / tw) if tw > effective_max else base_scale

    @staticmethod
    def _centered_x(text: str, font, scale: float,
                    thickness: int, x1: int, x2: int) -> int:
        """
        Coordenada X para centrar texto horizontalmente entre x1 y x2.
        Si el texto es más ancho que el espacio, lo ancla a x1.
        """
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        return max(x1, x1 + (x2 - x1 - tw) // 2)

    # ─── HUD Header ──────────────────────────────────────────────────────────

    def _draw_hud_header(self, frame, title: str):
        """Barra de estado fija en la parte superior de la pantalla."""
        h, w = frame.shape[:2]
        self._panel(frame, 0, 0, w, 65, color=(10, 10, 40), alpha=0.88)

        # [FIX] Reservar 90px a la derecha para el FPS counter antes de escalar
        max_title_w = w - 18 - 90
        scale = self._fit_scale(title, _FONT, 0.9, 2, max_title_w)
        self._text_shadow(frame, title, (18, 44), _FONT, scale, C_YELLOW, 2)

        cv2.putText(frame, f"FPS:{self._fps_display}", (w - 85, 44),
                    _FONT, 0.58, (80, 180, 80), 1, cv2.LINE_AA)

    def _update_fps(self):
        self._fps_count += 1
        now = time.time()
        if now - self._fps_clock >= 1.0:
            self._fps_display = self._fps_count
            self._fps_count   = 0
            self._fps_clock   = now

    # ─── Hilo de visión ───────────────────────────────────────────────────────

    def vision_worker(self):
        """HILO SECUNDARIO: captura, MediaPipe, matemáticas y cola de datos."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            validator = GestureValidator(json_path=_get_data_path('gestures.json'))
        except FileNotFoundError as e:
            log.error("GestureValidator: %s", e)
            validator = None

        mp_hands  = mp.solutions.hands
        hands     = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
        )

        recording_dynamic = False
        dynamic_frames    = 0
        dynamic_target    = None
        path_x, path_y   = [], []
        missing_frames    = 0

        while self.running:
            try:
                success, frame = cap.read()
                if not success:
                    continue

                frame     = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = hands.process(frame_rgb)

                detected_letter = None
                landmarks_list  = None

                if results.multi_hand_landmarks:
                    missing_frames  = 0
                    hand_landmarks  = results.multi_hand_landmarks[0]
                    landmarks_list  = hand_landmarks

                    current_angles      = get_hand_angles(hand_landmarks.landmark)
                    current_orientation = get_hand_orientation(hand_landmarks.landmark)

                    # 1. Reconocimiento estático
                    if validator:
                        clean_letter, _ = validator.recognize_static(
                            current_angles, current_orientation
                        )
                    else:
                        clean_letter = None

                    # 2. Interceptor U vs V (separación de dedos)
                    if clean_letter and clean_letter.rstrip('0123456789') in ('U', 'V'):
                        spread       = get_finger_spread(hand_landmarks.landmark)
                        clean_letter = 'V' if spread > 0.35 else 'U'

                    detected_letter = clean_letter

                    # 3. Detección de inicio dinámico
                    if validator:
                        possible_starts = validator.check_dynamic_start(
                            current_angles, current_orientation
                        )
                    else:
                        possible_starts = []

                    if possible_starts and not recording_dynamic and self.dynamic_lock_frames == 0:
                        recording_dynamic = True
                        dynamic_frames    = 0
                        dynamic_target    = possible_starts[0]
                        path_x, path_y    = [], []

                    # 4. Grabación de trayectoria
                    if recording_dynamic:
                        tip = hand_landmarks.landmark[8]
                        path_x.append(round(tip.x, 3))
                        path_y.append(round(tip.y, 3))
                        dynamic_frames += 1

                        detected_letter = (
                            f"{clean_letter} (Trazando {dynamic_target}...)"
                            if clean_letter else f"Trazando {dynamic_target}..."
                        )

                        if dynamic_frames >= 60:
                            if validator and validator.validate_trajectory(
                                dynamic_target, path_x, path_y
                            ):
                                self.last_dynamic_letter = dynamic_target
                                self.dynamic_lock_frames = 30
                            elif clean_letter:
                                # Fallback: la estática gana si no hubo trazo válido
                                self.last_dynamic_letter = clean_letter
                                self.dynamic_lock_frames = 45
                            recording_dynamic = False

                    # 5. Bloqueo dinámico activo
                    if self.dynamic_lock_frames > 0:
                        detected_letter          = self.last_dynamic_letter
                        self.dynamic_lock_frames -= 1
                        recording_dynamic        = False

                else:
                    # Mano no visible
                    missing_frames += 1
                    if recording_dynamic:
                        detected_letter = f"Trazando {dynamic_target}..."
                    elif self.dynamic_lock_frames > 0:
                        detected_letter          = self.last_dynamic_letter
                        self.dynamic_lock_frames -= 1

                    if missing_frames > 15:
                        recording_dynamic = False
                        dynamic_frames    = 0
                        path_x, path_y    = [], []

                # Enviar a la cola del hilo principal (descartar frame viejo si llena)
                if not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.data_queue.put({
                    "frame":     frame,
                    "landmarks": landmarks_list,
                    "letter":    detected_letter,
                })

            except Exception as e:
                log.error("vision_worker error: %s", e)

        cap.release()

    # ─── Módulos de presentación ──────────────────────────────────────────────

    def _procesar_menu(self, frame, index_x: int, index_y: int):
        h, w = frame.shape[:2]

        # Overlay general
        self._panel(frame, 0, 0, w, h, color=(5, 5, 20), alpha=0.45)

        # Header
        self._draw_hud_header(frame, "LESSA  GAME  ENGINE")
        cv2.putText(frame, "Apunta con tu dedo indice y mantenlo sobre una opcion",
                    (55, 88), _FONT, 0.55, C_GRAY, 1, cv2.LINE_AA)

        button_hovered_this_frame = None

        for i, (x1, y1, x2, y2, _, target) in enumerate(self.buttons):
            is_hovered = x1 < index_x < x2 and y1 < index_y < y2
            accent     = BTN_ACCENTS[i]
            label      = BTN_LABELS[i]

            # [FIX] Calcular escala dinámica para que la etiqueta quepa dentro del botón.
            # Área útil: desde x1+8 (barra de color) + 14 (margen) hasta x2-10.
            btn_available_w = (x2 - x1) - 40   # 8px barra + 22px margen izq + 10px margen der
            scale = self._fit_scale(label, _FONT, 0.70, 2, btn_available_w)

            # [FIX] Centrar verticalmente la etiqueta dentro del botón con th real.
            btn_h          = y2 - y1
            (_, th), _     = cv2.getTextSize(label, _FONT, scale, 2)
            text_y         = y1 + (btn_h + th) // 2

            if is_hovered:
                button_hovered_this_frame = target
                # Fondo con tinte del color accent
                self._panel(frame, x1, y1, x2, y2, color=accent, alpha=0.14)
                cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 2, cv2.LINE_AA)
                # Barra vertical accent izquierda
                cv2.rectangle(frame, (x1, y1), (x1 + 8, y2), accent, cv2.FILLED)
                # [FIX] Texto centrado verticalmente con escala ajustada
                self._text_shadow(frame, label, (x1 + 22, text_y),
                                  _FONT, scale, accent, 2)
                # Barra de progreso (hover timer)
                progress = self.hover_frames / self.selection_threshold
                self._progress_bar(frame, x1, y2 - 7, x2, y2, progress, accent)
            else:
                self._panel(frame, x1, y1, x2, y2, color=(25, 25, 25), alpha=0.65)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x1 + 8, y2), (70, 70, 70), cv2.FILLED)
                # [FIX] Misma escala dinámica en estado normal
                cv2.putText(frame, label, (x1 + 22, text_y),
                            _FONT, scale, C_GRAY, 1, cv2.LINE_AA)

        # Lógica de selección con hover timer
        if button_hovered_this_frame:
            if self.hovered_button == button_hovered_this_frame:
                self.hover_frames += 1
                if self.hover_frames >= self.selection_threshold:
                    self.hover_frames = 0
                    if button_hovered_this_frame == "SALIR":
                        self.running = False
                    elif button_hovered_this_frame == "AHORCADO":
                        self._iniciar_partida()
                    else:
                        self.state = button_hovered_this_frame
            else:
                self.hovered_button = button_hovered_this_frame
                self.hover_frames   = 0
        else:
            self.hovered_button = None
            self.hover_frames   = 0

    def _procesar_ahorcado(self, frame, current_letter, key):
        h, w = frame.shape[:2]

        # Overlay base
        self._panel(frame, 0, 0, w, h, color=(5, 5, 20), alpha=0.35)

        # ── Header con puntos ──────────────────────────────────────────────
        self._panel(frame, 0, 0, w, 65, color=(10, 10, 40), alpha=0.88)
        self._text_shadow(frame, "AHORCADO  LESSA", (18, 44),
                          _FONT, 0.9, C_YELLOW, 2)
        cv2.putText(frame, f"Puntos: {self.score}", (w - 195, 44),
                    _FONT, 0.78, C_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS:{self._fps_display}", (w - 85, 22),
                    _FONT, 0.55, (80, 180, 80), 1, cv2.LINE_AA)

        # ── Panel de la palabra ────────────────────────────────────────────
        display_word = " ".join(
            [c if c in self.guessed_letters else "_" for c in self.word_to_guess]
        )
        display_safe = display_word.replace('Ñ', 'N~')

        panel_x1, panel_y1 = 25, 72
        panel_x2, panel_y2 = w - 25, 130
        self._panel(frame, panel_x1, panel_y1, panel_x2, panel_y2,
                    color=(0, 25, 0), alpha=0.75)

        # [FIX] Escalar dinámicamente el display_word para que quepa en el panel.
        # Panel inner width: panel_x2 - panel_x1 - 20px de margen total.
        word_max_w     = (panel_x2 - panel_x1) - 20
        word_base_sc   = 1.6
        word_thickness = 4
        word_scale     = self._fit_scale(display_safe, _FONT, word_base_sc,
                                         word_thickness, word_max_w)

        # [FIX] Centrar horizontalmente y verticalmente dentro del panel.
        (tw_word, th_word), _ = cv2.getTextSize(display_safe, _FONT, word_scale, word_thickness)
        word_x = panel_x1 + (panel_x2 - panel_x1 - tw_word) // 2
        word_y = panel_y1 + (panel_y2 - panel_y1 + th_word) // 2
        word_x = max(panel_x1 + 10, word_x)    # nunca fuera del borde izquierdo

        self._text_shadow(frame, display_safe, (word_x, word_y),
                          _FONT, word_scale, C_YELLOW, word_thickness)

        # ── Panel lateral de información ───────────────────────────────────
        px1, px2 = w - 225, w - 10
        self._panel(frame, px1, 138, px2, 390, color=(20, 10, 10), alpha=0.78)
        cv2.rectangle(frame, (px1, 138), (px2, 390), (60, 40, 40), 1, cv2.LINE_AA)

        # Vidas (iconos O / X)
        cv2.putText(frame, "Vidas:", (px1 + 10, 168),
                    _FONT, 0.6, C_GRAY, 1, cv2.LINE_AA)
        for i in range(self.max_errors):
            icon  = "X" if i < self.errors else "O"
            color = C_RED if i < self.errors else C_GREEN
            cv2.putText(frame, icon, (px1 + 10 + i * 30, 200),
                        _FONT, 0.65, color, 2, cv2.LINE_AA)

        # Barra de errores
        self._progress_bar(frame, px1 + 10, 208, px2 - 10, 221,
                           self.errors / self.max_errors, C_RED, bg=(0, 50, 0))

        # Letra detectada
        cv2.putText(frame, "Detectando:", (px1 + 10, 252),
                    _FONT, 0.58, C_GRAY, 1, cv2.LINE_AA)
        if current_letter:
            safe = current_letter.replace('Ñ', 'N~')
            col  = C_ORANGE if "Trazando" in safe else C_GREEN

            # [FIX] Escala dinámica para la letra en el panel lateral.
            # Ancho disponible: desde px1+10 hasta px2-10.
            avail_w      = (px2 - 10) - (px1 + 10)
            letter_scale = self._fit_scale(safe, _FONT, 0.68, 2, avail_w)
            cv2.putText(frame, safe, (px1 + 10, 282),
                        _FONT, letter_scale, col, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Haz una sena...", (px1 + 10, 282),
                        _FONT, 0.52, (80, 80, 80), 1, cv2.LINE_AA)

        # Letras ya usadas
        cv2.putText(frame, "Usadas:", (px1 + 10, 320),
                    _FONT, 0.58, C_GRAY, 1, cv2.LINE_AA)
        used_str = " ".join(sorted(self.guessed_letters))
        cv2.putText(frame, used_str[:22], (px1 + 10, 348),
                    _FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)
        if len(used_str) > 22:
            cv2.putText(frame, used_str[22:44], (px1 + 10, 372),
                        _FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)

        # ── Footer ────────────────────────────────────────────────────────
        self._panel(frame, 0, h - 38, w, h, color=(10, 10, 40), alpha=0.85)
        cv2.putText(frame, "[ESPACIO] Confirmar letra    [M] Volver al menu",
                    (18, h - 13), _FONT, 0.58, C_GRAY, 1, cv2.LINE_AA)

        # Dibujar el muñeco del ahorcado
        self.hangman.draw(frame, self.errors)

        # Lógica de juego
        if key == ord(' ') and current_letter and len(current_letter) == 1:
            if current_letter in self.word_to_guess and current_letter not in self.guessed_letters:
                self.guessed_letters.add(current_letter)
                self.score += 10
            elif current_letter not in self.guessed_letters:
                self.errors += 1
                self.guessed_letters.add(current_letter)
                self.score = max(0, self.score - 5)

            if set(self.word_to_guess).issubset(self.guessed_letters):
                self.score += (self.max_errors - self.errors) * 20
                self.state = "VICTORIA"
            elif self.errors >= self.max_errors:
                self.state = "DERROTA"

        if key == ord('m'):
            self.state = "MENU"

    def _procesar_abecedario(self, frame, current_letter, key):
        h, w = frame.shape[:2]

        self._panel(frame, 0, 0, w, h, color=(5, 5, 20), alpha=0.38)
        self._draw_hud_header(frame, "PLAYGROUND  LESSA  -  Practica libre")

        # Panel oscuro detrás de la letra detectada
        box_x1, box_y1 = w // 2 - 110, 145
        box_x2, box_y2 = w // 2 + 110, 345
        self._panel(frame, box_x1, box_y1, box_x2, box_y2,
                    color=(0, 25, 0), alpha=0.75)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2),
                      C_GREEN, 1, cv2.LINE_AA)

        if current_letter:
            safe = current_letter.replace('Ñ', 'N~')

            if "Trazando" in safe:
                # [FIX] Calcular escala para que el texto "Trazando..." quepa
                # en el ancho útil de la pantalla (margen 40px a cada lado).
                # Luego centrarlo horizontalmente debajo del panel de la letra.
                traz_max_w = w - 80
                traz_scale = self._fit_scale(safe, _FONT, 1.1, 3, traz_max_w)

                (tw_tr, _), _ = cv2.getTextSize(safe, _FONT, traz_scale, 3)
                traz_x = (w - tw_tr) // 2           # centrado en pantalla
                traz_x = max(40, traz_x)             # nunca fuera del margen

                # [FIX] Posición Y fija a 285, dentro del área media de la pantalla
                cv2.putText(frame, safe, (traz_x, 285),
                            _FONT, traz_scale, C_ORANGE, 3, cv2.LINE_AA)
            else:
                # Letra grande centrada en el panel
                self._text_shadow(frame, safe, (w // 2 - 65, 315),
                                  _FONT, 4.5, C_GREEN, 9)
        else:
            cv2.putText(frame, "?", (w // 2 - 30, 315),
                        _FONT, 4.5, (55, 55, 55), 9, cv2.LINE_AA)

        cv2.putText(frame, "Haz una sena con tu mano", (w // 2 - 155, 372),
                    _FONT, 0.62, C_GRAY, 1, cv2.LINE_AA)

        self._panel(frame, 0, h - 38, w, h, color=(10, 10, 40), alpha=0.85)
        cv2.putText(frame, "[M] Volver al menu",
                    (18, h - 13), _FONT, 0.58, C_GRAY, 1, cv2.LINE_AA)

        if key == ord('m'):
            self.state = "MENU"

    def _procesar_victoria(self, frame, key):
        h, w = frame.shape[:2]
        self._panel(frame, 0, 0, w, h,  color=(0, 30, 0), alpha=0.72)
        self._panel(frame, 70, 95, w - 70, 430, color=(0, 45, 0), alpha=0.82)
        cv2.rectangle(frame, (70, 95), (w - 70, 430), C_GREEN, 2, cv2.LINE_AA)

        self._text_shadow(frame, "GANASTE!", (w // 2 - 155, 185),
                          _FONT, 2.4, C_GREEN, 5)
        cv2.putText(frame, f"Puntuacion: {self.score} pts", (w // 2 - 125, 255),
                    _FONT, 1.0, C_YELLOW, 2, cv2.LINE_AA)

        # [FIX] Escalar la palabra si es muy larga para que no salga del panel
        palabra_txt   = f"Palabra: {self.word_to_guess}"
        palabra_max_w = (w - 70) - (w // 2 - 110) - 10
        palabra_scale = self._fit_scale(palabra_txt, _FONT, 0.8, 1, palabra_max_w)
        cv2.putText(frame, palabra_txt, (w // 2 - 110, 300),
                    _FONT, palabra_scale, C_WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, "[ESPACIO]  Siguiente palabra  (mantener racha)", (100, 345),
                    _FONT, 0.65, C_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, "[R]        Registrar tu nombre en el ranking", (100, 378),
                    _FONT, 0.65, C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "[M]        Volver al menu principal", (100, 411),
                    _FONT, 0.65, C_GRAY, 1, cv2.LINE_AA)

        if key == ord(' '):
            self._iniciar_partida(mantener_puntaje=True)
        elif key == ord('r'):
            self.player_name = ""
            self.state = "REGISTRO"
        elif key == ord('m'):
            self.state = "MENU"

    def _procesar_derrota(self, frame, key):
        h, w = frame.shape[:2]
        self._panel(frame, 0, 0, w, h,  color=(25, 0, 0), alpha=0.72)
        self._panel(frame, 70, 95, w - 70, 420, color=(40, 0, 0), alpha=0.82)
        cv2.rectangle(frame, (70, 95), (w - 70, 420), C_RED, 2, cv2.LINE_AA)

        self._text_shadow(frame, "PERDISTE!", (w // 2 - 155, 180),
                          _FONT, 2.4, C_RED, 5)

        # [FIX] Escalar dinámicamente si la palabra es larga
        era_txt   = f"La palabra era: {self.word_to_guess}"
        era_max_w = (w - 70) - (w // 2 - 150) - 10
        era_scale = self._fit_scale(era_txt, _FONT, 0.85, 2, era_max_w)
        cv2.putText(frame, era_txt, (w // 2 - 150, 248),
                    _FONT, era_scale, C_WHITE, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Puntuacion final: {self.score} pts", (w // 2 - 130, 295),
                    _FONT, 0.78, C_GRAY, 1, cv2.LINE_AA)

        cv2.putText(frame, "[ESPACIO]  Volver a intentarlo", (120, 345),
                    _FONT, 0.65, C_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, "[M]        Volver al menu principal", (120, 380),
                    _FONT, 0.65, C_GRAY, 1, cv2.LINE_AA)

        if key == ord(' '):
            self._iniciar_partida(mantener_puntaje=False)
        elif key == ord('m'):
            self.state = "MENU"

    def _procesar_registro(self, frame, current_letter, key):
        h, w = frame.shape[:2]
        self._panel(frame, 0, 0, w, h, color=(5, 5, 20), alpha=0.52)
        self._draw_hud_header(frame, "REGISTRO  -  Deletrea tu nombre en LESSA")

        # Panel del nombre actual
        name_x1, name_y1 = 35,   95
        name_x2, name_y2 = w-35, 165
        self._panel(frame, name_x1, name_y1, name_x2, name_y2,
                    color=(0, 20, 30), alpha=0.82)
        cv2.rectangle(frame, (name_x1, name_y1), (name_x2, name_y2),
                      C_CYAN, 1, cv2.LINE_AA)

        nombre_disp = (self.player_name + "_")[:20]
        # [FIX] Ajustar escala si el nombre crece más allá del panel
        name_avail_w = (name_x2 - 10) - (name_x1 + 20)
        name_scale   = self._fit_scale(nombre_disp, _FONT, 1.4, 3, name_avail_w)
        self._text_shadow(frame, nombre_disp, (name_x1 + 20, name_y1 + 53),
                          _FONT, name_scale, C_CYAN, 3)

        # Panel de detección
        det_x1, det_y1 = 35,   180
        det_x2, det_y2 = w-35, 260
        self._panel(frame, det_x1, det_y1, det_x2, det_y2,
                    color=(0, 20, 0), alpha=0.75)
        cv2.putText(frame, "Detectando:", (det_x1 + 20, det_y1 + 32),
                    _FONT, 0.6, C_GRAY, 1, cv2.LINE_AA)

        status     = (current_letter.replace('Ñ', 'N~') if current_letter else "Haz una sena...")
        status_col = C_GREEN if (current_letter and len(current_letter) == 1) else C_ORANGE

        # [FIX] Escalar dinámicamente el estado de detección
        det_avail_w  = (det_x2 - 10) - (det_x1 + 20)
        status_scale = self._fit_scale(status, _FONT, 0.82, 2, det_avail_w)
        cv2.putText(frame, status, (det_x1 + 20, det_y1 + 68),
                    _FONT, status_scale, status_col, 2, cv2.LINE_AA)

        # Footer con controles
        self._panel(frame, 0, h - 72, w, h, color=(10, 10, 40), alpha=0.88)
        cv2.putText(frame, "[ESPACIO] Agregar la letra detectada", (18, h - 44),
                    _FONT, 0.6, C_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, "[B] Borrar ultima letra    [ENTER] Guardar y ver ranking",
                    (18, h - 14), _FONT, 0.6, C_GRAY, 1, cv2.LINE_AA)

        if key == ord(' ') and current_letter and len(current_letter) == 1:
            self.player_name += current_letter
        elif key == ord('b'):
            self.player_name = self.player_name[:-1]
        elif key == 13 and self.player_name.strip():    # ENTER
            self._guardar_puntaje()
            self.state = "RANKING"

    def _procesar_ranking(self, frame, key):
        h, w = frame.shape[:2]
        self._panel(frame, 0, 0, w, h, color=(10, 10, 40), alpha=0.78)
        self._draw_hud_header(frame, "RANKING  -  Top 5 Mejores Puntajes")

        self._panel(frame, 55, 75, w - 55, 450, color=(5, 5, 30), alpha=0.82)
        cv2.rectangle(frame, (55, 75), (w - 55, 450), C_CYAN, 1, cv2.LINE_AA)

        ranking       = self._cargar_ranking()
        medal_colors  = [C_YELLOW, C_GRAY, C_ORANGE, C_WHITE, C_WHITE]
        medal_labels  = ["1.", "2.", "3.", "4.", "5."]

        if ranking:
            y = 130
            for i, entry in enumerate(ranking[:5]):
                try:
                    col    = medal_colors[i]
                    nombre = str(entry.get('nombre', '???'))
                    puntos = int(entry.get('puntos', 0))

                    cv2.putText(frame, medal_labels[i], (80, y),
                                _FONT, 0.88, col, 2, cv2.LINE_AA)

                    # [FIX] Escalar nombre si es muy largo para no chocar con los puntos
                    nombre_max_w = (w - 155) - 120 - 10   # hasta donde comienzan los pts
                    nombre_scale = self._fit_scale(nombre, _FONT, 0.88, 2, nombre_max_w)
                    self._text_shadow(frame, nombre, (120, y),
                                      _FONT, nombre_scale, col, 2)

                    pts = f"{puntos} pts"
                    cv2.putText(frame, pts, (w - 155, y),
                                _FONT, 0.82, col, 2, cv2.LINE_AA)

                    # Separador
                    cv2.line(frame, (80, y + 10), (w - 70, y + 10),
                             (45, 45, 65), 1, cv2.LINE_AA)
                    y += 58
                except (KeyError, TypeError, ValueError) as e:
                    log.warning("Entrada de ranking malformada [%d]: %s", i, e)
        else:
            cv2.putText(frame, "Aun no hay puntajes registrados.", (100, 270),
                        _FONT, 0.72, C_GRAY, 1, cv2.LINE_AA)

        self._panel(frame, 0, h - 38, w, h, color=(10, 10, 40), alpha=0.88)
        cv2.putText(frame, "[M] Volver al menu", (18, h - 13),
                    _FONT, 0.58, C_GRAY, 1, cv2.LINE_AA)

        if key == ord('m'):
            self.state = "MENU"

    # ─── Bucle principal ──────────────────────────────────────────────────────

    def run_game(self):
        vision_thread = threading.Thread(target=self.vision_worker, daemon=True)
        vision_thread.start()

        mp_drawing = mp.solutions.drawing_utils
        mp_hands   = mp.solutions.hands

        # Estilo personalizado para los landmarks de la mano
        lm_spec   = mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3)
        conn_spec = mp_drawing.DrawingSpec(color=(160, 160, 0), thickness=2)

        while self.running:
            try:
                data           = self.data_queue.get(timeout=0.1)
                frame          = data["frame"]
                landmarks      = data["landmarks"]
                current_letter = data["letter"]

                # 1. Esqueleto de la mano con estilo personalizado
                if landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS,
                        lm_spec, conn_spec,
                    )

                # 2. Cursor del dedo índice con halo
                index_x, index_y = 0, 0
                if landmarks:
                    h, w, _ = frame.shape
                    index_x = int(landmarks.landmark[8].x * w)
                    index_y = int(landmarks.landmark[8].y * h)
                    cv2.circle(frame, (index_x, index_y), 18, C_PURPLE, 2, cv2.LINE_AA)
                    cv2.circle(frame, (index_x, index_y),  6, C_PURPLE, cv2.FILLED, cv2.LINE_AA)

                # 3. Lectura de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False

                # 4. Enrutar al módulo de estado correspondiente
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

                # 5. Actualizar FPS y renderizar
                self._update_fps()
                cv2.imshow('LESSA Game Engine', frame)

            except queue.Empty:
                pass
            except Exception as e:
                log.error("run_game loop error [state=%s]: %s", self.state, e)

        cv2.destroyAllWindows()
        vision_thread.join()


if __name__ == "__main__":
    game = LessaGameEngine()
    game.run_game()