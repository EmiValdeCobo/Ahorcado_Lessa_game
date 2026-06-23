import numpy as np

def calculate_angle(a, b, c):
    """
    Calcula el ángulo 3D en el vértice 'b' formado por los puntos a, b y c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Definir los vectores
    vector_ba = a - b
    vector_bc = c - b
    
    # Calcular el producto punto y las magnitudes
    dot_product = np.dot(vector_ba, vector_bc)
    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)
    
    # Evitar división por cero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    cosine_angle = dot_product / (norm_ba * norm_bc)
    # Recortar el valor para evitar errores de punto flotante en arccos
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def get_hand_angles(landmarks):
    """
    Recibe los 21 landmarks de MediaPipe y devuelve un array con los 10 ángulos
    críticos de las articulaciones de los dedos.
    """
    # Definición de vértices: (punto_base, vértice, punto_punta)
    # Estas combinaciones cubren la flexión de cada dedo
    angle_definitions = [
        (1, 2, 3), (2, 3, 4),       # Pulgar
        (5, 6, 7), (6, 7, 8),       # Índice
        (9, 10, 11), (10, 11, 12),  # Medio
        (13, 14, 15), (14, 15, 16), # Anular
        (17, 18, 19), (18, 19, 20)  # Meñique
    ]
    
    angles = []
    for (p1, p2, p3) in angle_definitions:
        # Extraer coordenadas [x, y, z] de cada landmark
        coord_1 = [landmarks[p1].x, landmarks[p1].y, landmarks[p1].z]
        coord_2 = [landmarks[p2].x, landmarks[p2].y, landmarks[p2].z]
        coord_3 = [landmarks[p3].x, landmarks[p3].y, landmarks[p3].z]
        
        angle = calculate_angle(coord_1, coord_2, coord_3)
        angles.append(round(angle, 2))
        
    return angles

def get_hand_orientation(landmarks):
    """
    Calcula la inclinación global de la mano en la pantalla (en grados).
    Usa el vector desde la muñeca (0) hasta el nudillo del dedo medio (9).
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    
    # Delta Y y Delta X
    dy = middle_mcp.y - wrist.y
    dx = middle_mcp.x - wrist.x
    
    # arctan2 maneja correctamente los cuadrantes
    angle = np.degrees(np.arctan2(dy, dx))
    
    return round(angle, 2)

def get_finger_spread(landmarks):
    """
    Calcula la distancia normalizada entre la punta del índice y el dedo medio.
    Sirve para diferenciar letras como la U (dedos juntos) y la V (dedos separados).
    """
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    
    # Usamos la palma como referencia de escala (distancia de la muñeca a la base del dedo medio)
    palm_base = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    palm_top = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    
    spread_dist = np.linalg.norm(index_tip - middle_tip)
    palm_size = np.linalg.norm(palm_top - palm_base)
    
    # Evitar división por cero
    if palm_size == 0:
        return 0
        
    return spread_dist / palm_size