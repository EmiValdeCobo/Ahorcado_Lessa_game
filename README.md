# LESSA Game Engine

Este proyecto es una aplicación interactiva basada en visión por computadora diseñada para practicar y aprender la Lengua de Señas Salvadoreña (LESSA). Utiliza la cámara web para rastrear los movimientos de la mano en tiempo real y traducirlos a letras mediante cálculos geométricos y validaciones de trayectoria.

El sistema incluye:

* **Menú Espacial:** Navegación utilizando la punta del dedo índice como cursor.
* **Juego de Ahorcado:** Modo supervivencia donde el jugador adivina palabras aleatorias de un banco de datos, acumulando puntos por rachas de victorias.
* **Playground (Abecedario):** Un entorno de práctica libre que muestra en pantalla la letra detectada en tamaño masivo para calibrar y aprender las señas.

## Requisitos

* Python 3.8 o superior.
* Cámara web funcional.

## Instalación y Configuración

1. Clona este repositorio o descarga los archivos.
2. Crea y activa un entorno virtual (recomendado):
`python -m venv venv`
`.\venv\Scripts\activate` (En Windows)
3. Instala las dependencias necesarias:
`pip install opencv-python mediapipe numpy`

## Forma de Ejecutarlo

**Opción 1: Desde el código fuente**
Navega a la carpeta `core` y ejecuta el motor principal:
`cd core`
`python engine.py`

**Opción 2: Ejecutable (Si ya fue compilado)**
Ve a la carpeta `dist/LESSA_Game/` y haz doble clic en `LESSA_Game.exe`. Asegúrate de que la carpeta `data` (que contiene los archivos `gestures.json`, `words.json` y `ranking.json`) se encuentre junto a la carpeta del ejecutable.

---

