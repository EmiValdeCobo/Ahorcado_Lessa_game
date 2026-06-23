import json
import numpy as np
import os

class GestureValidator:
    def __init__(self, json_path='../data/gestures.json'):
        self.gestures = self._load_gestures(json_path)
        # Umbral máximo de "distancia" permitida. 
        # Si la distancia es mayor a 50, significa que la mano no está haciendo ninguna seña válida.
        self.threshold_static = 50.0 

    def _load_gestures(self, path):
        if os.path.exists(path):
            with open(path, 'r') as file:
                return json.load(file)
        raise FileNotFoundError(f"No se encontró el archivo {path}. Asegúrate de correr el calibrador primero.")

    # Cambia la definición para recibir la orientación
    def recognize_static(self, current_angles, current_orientation):
        """Encuentra la seña estática más parecida usando Distancia Euclidiana y Orientación."""
        best_match = None
        min_distance = float('inf')

        for key, data in self.gestures.items():
            if data.get("tipo") == "estatica":
                saved_angles = np.array(data["angulos"])
                current = np.array(current_angles)
                
                distance = np.linalg.norm(saved_angles - current)

                saved_ori = data.get("orientacion")
                if saved_ori is not None: # Solo aplica si la letra fue recalibrada
                    # Calcular diferencia absoluta de rotación
                    ori_diff = abs(current_orientation - saved_ori)
                    if ori_diff > 180: 
                        ori_diff = 360 - ori_diff # Compensar el salto de 180 a -180
                    
                    # Si la mano está girada más de 45°, descartar la letra
                    if ori_diff > 45:
                        distance += 1000 
                # ------------------------------------

                if distance < min_distance:
                    min_distance = distance
                    best_match = key

        if min_distance <= self.threshold_static and best_match:
            clean_letter = best_match.rstrip('0123456789')
            return clean_letter, min_distance
        
        return None, min_distance

    def check_dynamic_start(self, current_angles):
        """Verifica si la mano está en la posición base con mayor tolerancia."""
        possible_starts = []
        # Le damos +15 de tolerancia porque al empezar a mover la mano, los ángulos se deforman un poco
        start_threshold = self.threshold_static + 15.0 
        
        for key, data in self.gestures.items():
            if data.get("tipo") == "dinamica":
                saved_base = np.array(data["angulos_base"])
                current = np.array(current_angles)
                distance = np.linalg.norm(saved_base - current)
                
                if distance <= start_threshold:
                    possible_starts.append(key.rstrip('0123456789'))
                    
        return list(set(possible_starts))

    def validate_trajectory(self, letter, path_x, path_y):
        """Valida el movimiento basándose en el Desplazamiento Neto (Punto final vs Punto inicial)."""
        if len(path_x) < 15: 
            return False 

        # Cuánto se movió el usuario en total
        user_dx = path_x[-1] - path_x[0]
        user_dy = path_y[-1] - path_y[0]

        for key, data in self.gestures.items():
            clean_key = key.rstrip('0123456789')
            if clean_key == letter and data.get("tipo") == "dinamica":
                exp_x = data["trayectoria_x"]
                exp_y = data["trayectoria_y"]
                
                if len(exp_x) < 15: continue
                
                # Cuánto se movió la mano en la grabación original
                exp_dx = exp_x[-1] - exp_x[0]
                exp_dy = exp_y[-1] - exp_y[0]

                # Validar el eje X (Derecha/Izquierda)
                match_x = False
                if abs(exp_dx) > 0.05: # Si en la grabación hubo un movimiento claro en X
                    # Multiplicar signos para asegurar que van en la misma dirección
                    match_x = (user_dx * exp_dx) > 0 and abs(user_dx) > 0.02
                else: # Si era un movimiento vertical, permitimos un pequeño error en X
                    match_x = abs(user_dx) < 0.08 
                    
                # Validar el eje Y (Arriba/Abajo)
                match_y = False
                if abs(exp_dy) > 0.05: 
                    match_y = (user_dy * exp_dy) > 0 and abs(user_dy) > 0.02
                else:
                    match_y = abs(user_dy) < 0.08

                # Si el movimiento general coincide, la seña es válida
                if match_x and match_y:
                    return True
                    
        return False