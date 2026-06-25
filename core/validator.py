import json
import numpy as np
import os


class GestureValidator:
    def __init__(self, json_path='../data/gestures.json'):
        self.gestures         = self._load_gestures(json_path)
        # Umbral máximo de "distancia euclidiana" permitida.
        self.threshold_static = 50.0

    # ─── Carga de datos ─────────────────────────────────────────────────────

    def _load_gestures(self, path: str) -> dict:
        """
        Carga gestures.json y precomputa los arrays numpy para evitar
        la conversión repetida en recognize_static y check_dynamic_start.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró '{path}'. Asegúrate de correr el calibrador primero."
            )
        with open(path, 'r') as f:
            raw = json.load(f)

        # Precomputar arrays numpy una sola vez al iniciar
        for data in raw.values():
            tipo = data.get("tipo")
            if tipo == "estatica":
                data["_np"] = np.array(data["angulos"], dtype=np.float32)
            elif tipo == "dinamica":
                data["_np"] = np.array(data["angulos_base"], dtype=np.float32)

        return raw

    # ─── Reconocimiento estático ─────────────────────────────────────────────

    def recognize_static(self, current_angles: list, current_orientation: float):
        """
        Encuentra la seña estática más cercana usando distancia euclidiana
        con penalización por diferencia de orientación.

        Retorna: (letra, distancia) o (None, distancia_mínima)
        """
        best_match   = None
        min_distance = float('inf')
        current      = np.array(current_angles, dtype=np.float32)

        for key, data in self.gestures.items():
            if data.get("tipo") != "estatica":
                continue

            # Usar array precomputado (no np.array() en cada frame)
            distance = float(np.linalg.norm(data["_np"] - current))

            # Penalización por orientación
            saved_ori = data.get("orientacion")
            if saved_ori is not None:
                ori_diff = abs(current_orientation - saved_ori)
                if ori_diff > 180:
                    ori_diff = 360 - ori_diff
                if ori_diff > 45:
                    distance += 1000

            if distance < min_distance:
                min_distance = distance
                best_match   = key
                if distance == 0:
                    break  # Coincidencia perfecta → salida anticipada

        if min_distance <= self.threshold_static and best_match:
            return best_match.rstrip('0123456789'), min_distance

        return None, min_distance

    # ─── Detección de inicio dinámico ────────────────────────────────────────

    def check_dynamic_start(self, current_angles: list, current_orientation: float) -> list:
        """
        Verifica si la posición actual coincide con la posición base
        de alguna seña dinámica (ej. J, Ñ, Z).
        """
        possible_starts = []
        current         = np.array(current_angles, dtype=np.float32)

        for key, data in self.gestures.items():
            if data.get("tipo") != "dinamica":
                continue

            distance = float(np.linalg.norm(data["_np"] - current))

            saved_ori = data.get("orientacion")
            if saved_ori is not None:
                ori_diff = abs(current_orientation - saved_ori)
                if ori_diff > 180:
                    ori_diff = 360 - ori_diff
                if ori_diff > 45:
                    distance += 1000

            if distance <= self.threshold_static:
                possible_starts.append(key.rstrip('0123456789'))

        return list(set(possible_starts))

    # ─── Validación de trayectoria ───────────────────────────────────────────

    def validate_trajectory(self, letter: str, path_x: list, path_y: list) -> bool:
        """
        Valida el movimiento grabado contra la trayectoria esperada.
        Requiere recorrido neto > 15% de la pantalla.
        Retorna True en el primer match encontrado (early exit).
        """
        if len(path_x) < 15:
            return False

        user_dx        = path_x[-1] - path_x[0]
        user_dy        = path_y[-1] - path_y[0]
        total_distance = (user_dx ** 2 + user_dy ** 2) ** 0.5

        if total_distance < 0.15:
            return False

        for key, data in self.gestures.items():
            # Saltar si no es la letra buscada o no es dinámica
            if key.rstrip('0123456789') != letter or data.get("tipo") != "dinamica":
                continue

            exp_x = data["trayectoria_x"]
            exp_y = data["trayectoria_y"]

            if len(exp_x) < 15:
                continue

            exp_dx = exp_x[-1] - exp_x[0]
            exp_dy = exp_y[-1] - exp_y[0]

            # Coincidencia en X
            if abs(exp_dx) > 0.05:
                match_x = (user_dx * exp_dx) > 0 and abs(user_dx) > 0.10
            else:
                match_x = abs(user_dx) < 0.10

            # Coincidencia en Y
            if abs(exp_dy) > 0.05:
                match_y = (user_dy * exp_dy) > 0 and abs(user_dy) > 0.10
            else:
                match_y = abs(user_dy) < 0.10

            if match_x and match_y:
                return True  # Early exit: primer match encontrado

        return False