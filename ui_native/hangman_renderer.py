import cv2


class HangmanRenderer:
    # ─── Paleta de colores (BGR) ─────────────────────────────────────────────
    COL_SCAFFOLD = (160, 130,  80)   # Azul-acero para la estructura
    COL_ROPE     = ( 80,  80, 200)   # Rojo oscuro para la cuerda
    COL_HEAD     = (110, 190, 230)   # Naranja suave (piel)
    COL_BODY     = ( 80, 210,  80)   # Verde para torso y extremidades
    COL_FACE     = ( 20,  20,  20)   # Negro para los rasgos
    COL_BORDER   = (220, 220, 220)   # Blanco/gris para el borde de la cabeza

    def __init__(self, start_x: int = 50, start_y: int = 350):
        self.origin_x = start_x
        self.origin_y = start_y
        self.t = 4  # grosor base

    # ─── Helper: línea antialiased con caps redondeados ─────────────────────

    def _line(self, frame, p1: tuple, p2: tuple, color: tuple, t: int = None):
        """Dibuja una línea antialiased con terminaciones redondeadas."""
        t = t or self.t
        cv2.line(frame, p1, p2, color, t, cv2.LINE_AA)
        cv2.circle(frame, p1, t // 2, color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, p2, t // 2, color, cv2.FILLED, cv2.LINE_AA)

    # ─── Dibujo principal ────────────────────────────────────────────────────

    def draw(self, frame, errors: int):
        """
        Dibuja el ahorcado según la cantidad de errores (0–6).
        Actúa como capa superpuesta; retorna el frame modificado.
        """
        x, y = self.origin_x, self.origin_y

        # ── 1. Base y Poste Principal ────────────────────────────────────────
        if errors >= 1:
            self._line(frame, (x,       y),       (x + 120, y),      self.COL_SCAFFOLD)
            self._line(frame, (x + 60,  y),       (x + 60,  y - 250), self.COL_SCAFFOLD)

        # ── 2. Viga Superior y Cuerda ────────────────────────────────────────
        if errors >= 2:
            self._line(frame, (x + 60,  y - 250), (x + 180, y - 250), self.COL_SCAFFOLD)
            self._line(frame, (x + 180, y - 250), (x + 180, y - 210), self.COL_ROPE, self.t - 1)

        # ── 3. Cabeza ────────────────────────────────────────────────────────
        if errors >= 3:
            cx, cy = x + 180, y - 180  # Centro de la cabeza

            # Relleno + borde
            cv2.circle(frame, (cx, cy), 30, self.COL_HEAD,   cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 30, self.COL_BORDER, 2,          cv2.LINE_AA)

            # Ojos
            cv2.circle(frame, (cx - 10, cy - 5), 4, self.COL_FACE, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx + 10, cy - 5), 4, self.COL_FACE, cv2.FILLED, cv2.LINE_AA)

            # Boca dinámica: sonriente antes de 5 errores, triste a partir de 5
            # cv2.ellipse: ángulo 0→180 = arco inferior (∪ = sonrisa)
            #              ángulo 180→360 = arco superior (∩ = tristeza)
            mouth_center = (cx, cy + 15)
            if errors < 5:
                cv2.ellipse(frame, mouth_center, (10, 6), 0,   0, 180, self.COL_FACE, 2, cv2.LINE_AA)
            else:
                cv2.ellipse(frame, mouth_center, (10, 6), 0, 180, 360, self.COL_FACE, 2, cv2.LINE_AA)

        # ── 4. Torso ─────────────────────────────────────────────────────────
        if errors >= 4:
            self._line(frame, (x + 180, y - 150), (x + 180, y - 50), self.COL_BODY)

        # ── 5. Brazos ────────────────────────────────────────────────────────
        if errors >= 5:
            self._line(frame, (x + 180, y - 130), (x + 140, y - 80), self.COL_BODY)
            self._line(frame, (x + 180, y - 130), (x + 220, y - 80), self.COL_BODY)

        # ── 6. Piernas ───────────────────────────────────────────────────────
        if errors >= 6:
            self._line(frame, (x + 180, y - 50), (x + 140, y + 10), self.COL_BODY)
            self._line(frame, (x + 180, y - 50), (x + 220, y + 10), self.COL_BODY)

        return frame