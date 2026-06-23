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

    def recognize_static(self, current_angles):
        """
        Encuentra la seña estática más parecida usando Distancia Euclidiana.
        """
        best_match = None
        min_distance = float('inf')

        for key, data in self.gestures.items():
            if data.get("tipo") == "estatica":
                saved_angles = np.array(data["angulos"])
                current = np.array(current_angles)
                
                # Cálculo de la distancia euclidiana entre los dos arrays de 10 ángulos
                distance = np.linalg.norm(saved_angles - current)

                if distance < min_distance:
                    min_distance = distance
                    best_match = key

        if min_distance <= self.threshold_static and best_match:
            # Limpia los números al final (ej. 'M1' se convierte en 'M', 'N2' en 'N')
            clean_letter = best_match.rstrip('0123456789')
            return clean_letter, min_distance
        
        return None, min_distance

    def check_dynamic_start(self, current_angles):
        """
        Verifica si la mano está en la posición base de una seña dinámica (ej. inicio de la Z o Ñ).
        """
        possible_starts = []
        
        for key, data in self.gestures.items():
            if data.get("tipo") == "dinamica":
                saved_base = np.array(data["angulos_base"])
                current = np.array(current_angles)
                distance = np.linalg.norm(saved_base - current)
                
                if distance <= self.threshold_static:
                    possible_starts.append(key.rstrip('0123456789'))
                    
        return list(set(possible_starts)) # Retorna letras únicas

    def validate_trajectory(self, letter, path_x, path_y):
        """
        Valida la dirección general de una trayectoria comparando el desplazamiento neto.
        """
        if len(path_x) < 5: 
            return False # Ignorar movimientos demasiado cortos
        
        # Calcular el vector de desplazamiento del usuario (Punto final - Punto inicial)
        user_dx = path_x[-1] - path_x[0]
        user_dy = path_y[-1] - path_y[0]
        
        for key, data in self.gestures.items():
            clean_key = key.rstrip('0123456789')
            if clean_key == letter and data.get("tipo") == "dinamica":
                # Calcular el vector de desplazamiento guardado en el JSON
                expected_dx = data["trayectoria_x"][-1] - data["trayectoria_x"][0]
                expected_dy = data["trayectoria_y"][-1] - data["trayectoria_y"][0]
                
                # Tolerancia mínima para no contar temblores de la mano como movimiento
                noise_threshold = 0.05 
                
                # Verificar si el movimiento tiene la misma dirección general multiplicando los signos
                match_x = (user_dx * expected_dx) > 0 or abs(user_dx) < noise_threshold
                match_y = (user_dy * expected_dy) > 0 or abs(user_dy) < noise_threshold
                
                if match_x and match_y:
                    return True
                    
        return False