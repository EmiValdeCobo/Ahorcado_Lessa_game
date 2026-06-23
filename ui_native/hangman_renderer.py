import cv2

class HangmanRenderer:
    def __init__(self, start_x=50, start_y=350):
        # Punto de origen (esquina inferior izquierda de la base del poste)
        self.origin_x = start_x
        self.origin_y = start_y
        self.color = (255, 255, 255)  # Blanco en formato BGR
        self.thickness = 4

    def draw(self, frame, errors):
        """
        Dibuja el estado del ahorcado en el frame dependiendo de la cantidad de errores.
        Actúa como una capa superpuesta.
        """
        x, y = self.origin_x, self.origin_y

        # 1. Base y Poste Principal
        if errors >= 1:
            cv2.line(frame, (x, y), (x + 120, y), self.color, self.thickness)             # Base
            cv2.line(frame, (x + 60, y), (x + 60, y - 250), self.color, self.thickness)   # Poste vertical

        # 2. Viga Superior y Cuerda
        if errors >= 2:
            cv2.line(frame, (x + 60, y - 250), (x + 180, y - 250), self.color, self.thickness) # Viga
            cv2.line(frame, (x + 180, y - 250), (x + 180, y - 210), self.color, self.thickness) # Cuerda

        # 3. Cabeza
        if errors >= 3:
            cv2.circle(frame, (x + 180, y - 180), 30, self.color, self.thickness)

        # 4. Torso
        if errors >= 4:
            cv2.line(frame, (x + 180, y - 150), (x + 180, y - 50), self.color, self.thickness)

        # 5. Brazos
        if errors >= 5:
            cv2.line(frame, (x + 180, y - 130), (x + 140, y - 80), self.color, self.thickness) # Brazo izquierdo
            cv2.line(frame, (x + 180, y - 130), (x + 220, y - 80), self.color, self.thickness) # Brazo derecho

        # 6. Piernas
        if errors >= 6:
            cv2.line(frame, (x + 180, y - 50), (x + 140, y + 10), self.color, self.thickness) # Pierna izquierda
            cv2.line(frame, (x + 180, y - 50), (x + 220, y + 10), self.color, self.thickness) # Pierna derecha

        return frame