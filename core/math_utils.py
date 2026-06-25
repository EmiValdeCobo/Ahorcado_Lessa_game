import numpy as np

_P1 = np.array([1,  2,  5,  6,  9, 10, 13, 14, 17, 18], dtype=np.int32)
_P2 = np.array([2,  3,  6,  7, 10, 11, 14, 15, 18, 19], dtype=np.int32)
_P3 = np.array([3,  4,  7,  8, 11, 12, 15, 16, 19, 20], dtype=np.int32)


def get_hand_angles(landmarks) -> list:
    """
    Calcula los 10 ángulos de articulación de los dedos en una sola operación
    batch, en lugar de 10 llamadas secuenciales.

    Retorna: lista de 10 floats redondeados a 2 decimales.
    """
    # Extraer los 21 landmarks como un array (21, 3) en un solo paso
    lm = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)

    # Vectores BA y BC para los 10 ángulos simultáneamente  →  (10, 3)
    ba = lm[_P1] - lm[_P2]
    bc = lm[_P3] - lm[_P2]

    # Producto punto vectorizado: einsum 'ij,ij->i' = suma(ba * bc) por fila
    dot   = np.einsum('ij,ij->i', ba, bc)                              # (10,)
    norms = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)   # (10,)

    # Evitar división por cero
    norms = np.where(norms == 0, 1e-9, norms)

    angles = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
    return np.round(angles, 2).tolist()


def get_hand_orientation(landmarks) -> float:
    """
    Ángulo global de la mano: vector muñeca (0) → nudillo medio (9).
    Retorna grados [-180, 180].
    """
    wrist = landmarks[0]
    mid   = landmarks[9]
    angle = np.degrees(np.arctan2(mid.y - wrist.y, mid.x - wrist.x))
    return round(float(angle), 2)


def get_finger_spread(landmarks) -> float:
    """
    Distancia normalizada entre la punta del índice (8) y la del medio (12).
    Normalizada con el tamaño de la palma para invarianza de escala.
    Útil para distinguir U (dedos juntos) vs V (dedos separados).
    """
    lm       = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    spread   = np.linalg.norm(lm[8]  - lm[12])
    palm_ref = np.linalg.norm(lm[9]  - lm[0])
    return float(spread / palm_ref) if palm_ref > 1e-9 else 0.0


# ─── Mantenida para compatibilidad con calibrator.py ────────────────────────
def calculate_angle(a, b, c) -> float:
    """Calcula el ángulo 3D en el vértice 'b' (compatibilidad)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))))