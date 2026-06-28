import json
import numpy as np
import os
import logging

log = logging.getLogger('lessa')


class GestureValidator:
    """
    ─────────────────────────────────────────────────────────────────────────
    • _path_features(): nueva función que extrae la "forma" de una trayectoria
      mediante longitud de arco, inversiones de dirección (x_rev / y_rev) y
      el ratio de oscilación. Estos descriptores se precomputan para cada
      plantilla dinámica al cargar gestures.json.

    • validate_trajectory() — Rama 1 (gestos oscilantes, ej. Ñ):
      El umbral anterior `total_distance < 0.15` rechazaba Ñ porque la ola
      vuelve casi al punto de inicio (net_dist ≈ 0.05). Ahora se usa la
      longitud de arco como criterio alternativo, y la comparación se hace
      sobre el conteo de inversiones (x_rev ≈ 4, y_rev ≈ 4 para Ñ).

    • validate_trajectory() — Rama 2 (gestos lineales con cambios, ej. Z):
      Se añade comparación de x_rev / y_rev con tolerancia ±2 para
      discriminar Z (x_rev=2) de J (x_rev=0) aunque tengan desplazamientos
      netos similares.
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, json_path: str = '../data/gestures.json'):
        self.gestures         = self._load_gestures(json_path)
        self.threshold_static = 50.0

    # ─── Carga de datos ──────────────────────────────────────────────────────

    def _load_gestures(self, path: str) -> dict:
        """
        Carga gestures.json y precomputa:
        · arrays numpy de ángulos (para reconocimiento estático y base dinámica)
        · descriptores de forma de trayectoria (para validate_trajectory)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró '{path}'. "
                f"Asegúrate de correr el calibrador primero."
            )
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for key, data in raw.items():
            tipo = data.get("tipo")
            if tipo == "estatica":
                data["_np"] = np.array(data["angulos"], dtype=np.float32)

            elif tipo == "dinamica":
                data["_np"] = np.array(data["angulos_base"], dtype=np.float32)

                # [NEW] Precomputar descriptores de forma de la trayectoria
                tx = data.get("trayectoria_x", [])
                ty = data.get("trayectoria_y", [])
                if len(tx) >= 15:
                    data["_feat"] = self._path_features(tx, ty)
                    log.debug(
                        "Plantilla '%s': x_rev=%d y_rev=%d arc=%.3f osc=%.1f",
                        key,
                        data["_feat"]["x_rev"],
                        data["_feat"]["y_rev"],
                        data["_feat"]["arc_len"],
                        data["_feat"]["osc_ratio"],
                    )
                else:
                    data["_feat"] = None

        return raw

    # ─── Análisis de forma de trayectoria ────────────────────────────────────

    @staticmethod
    def _smooth(path: list, window: int = 5) -> list:
        """
        Media móvil simple para reducir el ruido de la cámara
        antes de contar inversiones de dirección.
        """
        out = []
        half = window // 2
        for i in range(len(path)):
            s = max(0, i - half)
            e = min(len(path), i + half + 1)
            out.append(sum(path[s:e]) / (e - s))
        return out

    @staticmethod
    def _count_reversals(path: list, min_delta: float = 0.007) -> int:
        """
        Cuenta inversiones de dirección en un camino 1D suavizado.

        min_delta (normalizado 0-1): mínimo desplazamiento entre muestras
        para considerarlo movimiento real (filtra jitter de cámara).
        A 640px de ancho, 0.007 ≈ 4.5 px → ignora temblores menores.
        """
        last_dir = 0
        rev      = 0
        for i in range(1, len(path)):
            d = path[i] - path[i - 1]
            if abs(d) > min_delta:
                cur = 1 if d > 0 else -1
                if last_dir != 0 and cur != last_dir:
                    rev += 1
                last_dir = cur
        return rev

    def _path_features(self, path_x: list, path_y: list) -> dict:
        """
        Extrae descriptores compactos de una trayectoria 2D.

        Retorna dict con:
        ─────────────────────────────────────────────────────────────────
        x_rev / y_rev   : inversiones de dirección en cada eje.
                          J→ x:0 y:0 | Ñ→ x:4+ ó y:4+ | Z→ x:2 y:0
        arc_len         : longitud total del arco (normalizada 0-1).
                          Clave para Ñ que tiene arc>>net.
        net_dx / net_dy : desplazamiento neto inicio→fin.
        net_dist        : módulo del desplazamiento neto.
        osc_ratio       : arc_len / net_dist.
                          Ñ ≈ 11  |  J ≈ 1.1  |  Z ≈ 2.4
        is_oscillating  : True si osc_ratio > 3.5 Y arc_len > 0.12.
                          Identifica gestos de ola (Ñ).
        ─────────────────────────────────────────────────────────────────
        """
        # Suavizar para contar inversiones sin ruido
        sx = self._smooth(path_x)
        sy = self._smooth(path_y)

        x_rev = self._count_reversals(sx)
        y_rev = self._count_reversals(sy)

        arc_len = sum(
            ((path_x[i + 1] - path_x[i]) ** 2
             + (path_y[i + 1] - path_y[i]) ** 2) ** 0.5
            for i in range(len(path_x) - 1)
        )

        net_dx   = path_x[-1] - path_x[0]
        net_dy   = path_y[-1] - path_y[0]
        net_dist = (net_dx ** 2 + net_dy ** 2) ** 0.5

        # Ratio de oscilación: cuántas veces más largo es el arco que el despl. neto
        osc_ratio = arc_len / max(net_dist, 1e-6)

        return {
            "x_rev":          x_rev,
            "y_rev":          y_rev,
            "arc_len":        arc_len,
            "net_dx":         net_dx,
            "net_dy":         net_dy,
            "net_dist":       net_dist,
            "osc_ratio":      osc_ratio,
            # Gesto oscilante: arco mucho mayor que desplazamiento neto (ej. Ñ)
            "is_oscillating": osc_ratio > 3.5 and arc_len > 0.12,
        }

    # ─── Reconocimiento estático ──────────────────────────────────────────────

    def recognize_static(self, current_angles: list, current_orientation: float):
        """
        Encuentra la seña estática más cercana usando distancia euclidiana
        con penalización por diferencia de orientación.

        Retorna: (letra, distancia) ó (None, distancia_mínima)
        """
        best_match   = None
        min_distance = float('inf')
        current      = np.array(current_angles, dtype=np.float32)

        for key, data in self.gestures.items():
            if data.get("tipo") != "estatica":
                continue

            distance = float(np.linalg.norm(data["_np"] - current))

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
                    break

        if min_distance <= self.threshold_static and best_match:
            return best_match.rstrip('0123456789'), min_distance

        return None, min_distance

    # ─── Detección de inicio dinámico ────────────────────────────────────────

    def check_dynamic_start(self, current_angles: list,
                            current_orientation: float) -> list:
        """
        Verifica si la posición actual coincide con la posición base
        de alguna seña dinámica (J, Ñ, Z).
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

    # ─── Validación de trayectoria ────────────────────────────────────────────

    def validate_trajectory(self, letter: str,
                            path_x: list, path_y: list) -> bool:
        """
        Valida el movimiento grabado contra la trayectoria esperada.

        Mejoras sobre la versión anterior:
        ────────────────────────────────────────────────────────────────
        ANTES:  if total_distance < 0.15: return False
                → Ñ SIEMPRE fallaba (su desplazamiento neto ≈ 0.05).
                → Z y J se distinguían solo por dirección neta (insuficiente).

        AHORA:
        1. Umbral de movimiento mínimo: se acepta si net_dist > 0.12
           ó arc_len > 0.18 → Ñ pasa gracias a su largo arco.
        2. Rama A — gestos oscilantes (Ñ):
           Compara conteo de inversiones (x_rev + y_rev) con tolerancia ±2.
           No usa dirección neta (es casi cero e inestable).
        3. Rama B — gestos direccionales (J, Z):
           Mantiene la comparación de dirección neta + agrega comparación
           de x_rev y y_rev (tolerancia ±2) para discriminar J de Z.
        ────────────────────────────────────────────────────────────────
        """
        if len(path_x) < 15:
            return False

        user = self._path_features(path_x, path_y)

        # ── Umbral mínimo de movimiento ───────────────────────────────────────
        # [FIX] Usar arc_len como alternativa a net_dist para no rechazar Ñ.
        # Ñ: net_dist ≈ 0.05 (falla umbral anterior), pero arc_len ≈ 0.60.
        if user["net_dist"] < 0.12 and user["arc_len"] < 0.18:
            return False

        for key, data in self.gestures.items():
            if (key.rstrip('0123456789') != letter
                    or data.get("tipo") != "dinamica"):
                continue

            # Usar descriptores precomputados; calcular on-the-fly si faltan
            tmpl = data.get("_feat")
            if tmpl is None:
                exp_x = data.get("trayectoria_x", [])
                exp_y = data.get("trayectoria_y", [])
                if len(exp_x) < 15:
                    continue
                tmpl = self._path_features(exp_x, exp_y)

            # ── RAMA A: Gesto oscilante (Ñ) ──────────────────────────────────
            # [FIX] La dirección neta de Ñ es casi 0 e inestable → no usarla.
            # En su lugar, verificar que ambos (plantilla y usuario) oscilen
            # y que el número de inversiones sea similar.
            if tmpl["is_oscillating"]:
                if not user["is_oscillating"]:
                    # La plantilla oscila pero el usuario no → no coincide
                    continue

                tmpl_total_rev = tmpl["x_rev"] + tmpl["y_rev"]
                user_total_rev = user["x_rev"] + user["y_rev"]

                # Tolerancia amplia (±3) porque la velocidad de la ola varía
                if abs(user_total_rev - tmpl_total_rev) <= 3:
                    log.debug(
                        "Ñ validada: user_rev=%d tmpl_rev=%d arc=%.3f",
                        user_total_rev, tmpl_total_rev, user["arc_len"],
                    )
                    return True

                continue  # Plantilla oscilante sin coincidencia suficiente

            # ── RAMA B: Gesto direccional (J, Z y futuros) ───────────────────
            # Paso B-1: Verificar dirección neta.
            # [FIX] Umbral relativo al template (60% del despl. calibrado).
            # Evita rechazar J cuyo componente X es pequeño pero real.
            exp_dx = tmpl["net_dx"]
            exp_dy = tmpl["net_dy"]

            if abs(exp_dx) > 0.05:
                min_x   = max(0.05, abs(exp_dx) * 0.6)
                match_x = (user["net_dx"] * exp_dx) > 0 and abs(user["net_dx"]) >= min_x
            else:
                match_x = abs(user["net_dx"]) < 0.12

            if abs(exp_dy) > 0.05:
                min_y   = max(0.05, abs(exp_dy) * 0.6)
                match_y = (user["net_dy"] * exp_dy) > 0 and abs(user["net_dy"]) >= min_y
            else:
                match_y = abs(user["net_dy"]) < 0.12

            if not (match_x and match_y):
                continue

            # Paso B-2: [FIX] Comparar conteo de inversiones (discrimina J vs Z).
            # J: x_rev ≈ 0-1, y_rev ≈ 0   (curva suave hacia abajo)
            # Z: x_rev ≈ 2,   y_rev ≈ 0-1 (zigzag con 2 cambios en X)
            # Tolerancia ±2 para absorber variación natural entre intentos.
            x_rev_ok = abs(user["x_rev"] - tmpl["x_rev"]) <= 2
            y_rev_ok = abs(user["y_rev"] - tmpl["y_rev"]) <= 2

            if x_rev_ok and y_rev_ok:
                log.debug(
                    "%s validada: net(%.2f,%.2f) x_rev=%d/%d y_rev=%d/%d",
                    letter,
                    user["net_dx"], user["net_dy"],
                    user["x_rev"], tmpl["x_rev"],
                    user["y_rev"], tmpl["y_rev"],
                )
                return True

        return False