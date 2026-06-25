# Documentación de Vibecoding y Proceso de Desarrollo

## 2. Evidencia de vibecoding (prompts)

Durante el desarrollo de este proyecto, se utilizó una metodología de interacción constante con la IA para resolver problemas de lógica, geometría espacial, UI y compilación. A continuación, se documentan los prompts clave utilizados en orden cronológico:

### Fase 1: Desafíos de Geometría y Detección Dinámica

**1. Prompt inicial de trayectorias:**

* **Qué se pidió:** *"Me sigue sin detectar el Z, Ñ y J, entonces me podrias ayudar con esa parte, porfavor, ya sea modificando calibrator para una mejor deteccion o añadiedno mas"*
* **Por qué se ajustó:** El modelo base medía el movimiento en línea recta (A a B). La IA ajustó el validador para dividir la trayectoria en 3 fases analíticas y cambió el "lápiz" virtual de la muñeca a la punta del dedo índice (`landmark 8`).

**2. Prompt de invarianza a la rotación:**

* **Qué se pidió:** *"Quiero mejorar lo que viene siendo la recalibracion por ejemplo la u me la confunde con la H porque la u es enfrente la mano y la h es una pistola acostado..."*
* **Por qué se ajustó:** La IA no distinguía orientación. Se le solicitó resolverlo y generó la función `get_hand_orientation` usando arcotangente para medir el ángulo global de la muñeca respecto a los nudillos.

**3. Prompt de separación de dedos:**

* **Qué se pidió:** *"tengo un problema con la U y V, no me esta identificanod la V, unicamente me identifca la U y es debido a que esas dos son las que esta levantada y la que diferencia es la abertura"*
* **Por qué se ajustó:** Se necesitó crear una regla matemática específica (`get_finger_spread`) para medir la distancia Euclidiana normalizada entre las puntas del dedo índice y medio, ya que los ángulos de flexión eran idénticos.

### Fase 2: Ajustes de Tolerancia y Flujo de Juego

**4. Prompt sobre el "robo de estado":**

* **Qué se pidió:** *"Si teniendo problemas con la J, no la identifica como J y de ahi igualmente con la Ñ, con la z siempre aparece cuanndo comienzo a Z que lo estoy trazando"*
* **Por qué se ajustó:** Se detectó que las señas estáticas (N) interrumpían a las dinámicas (Ñ). La IA modificó el bucle de cámara para evaluar ambas simultáneamente y creó un bloqueo temporal (`dynamic_lock_frames`).

**5. Prompt sobre la oclusión de la cámara:**

* **Qué se pidió:** *"Como giro completamente la mano, ya no detecta la mano o como termina entonces ya no puede completar el identificar, entonces en ese caso que se deberia de hacer?"*
* **Por qué se ajustó:** MediaPipe perdía la mano al girarla. Se ajustó el código añadiendo un "Buffer de persistencia" (`missing_frames`) que permite que la cámara pierda de vista la mano durante 15 frames sin cancelar el trazo.

**6. Prompt para validación por inactividad:**

* **Qué se pidió:** *"hagamos una validacion donde si ya paso 3 segundos donde esta el A (trazando Ñ) o cualquier caso entonces gano la A, entonces colocar como esa"*
* **Por qué se ajustó:** Para evitar bucles infinitos de "Trazando...", se le indicó a la IA que implementara un *fallback*: si pasa el tiempo límite sin movimiento real, el sistema debe darle la victoria a la seña estática de forma definitiva.

### Fase 3: Resolución de Errores (Bugs) y Compilación

**7. Prompt de error de variables:**

* **Qué se pidió:** *"da este error (venv) PS C:\Users...\core> python .\calibrator.py [...] NameError: name 'path_x' is not defined"*
* **Por qué se ajustó:** Fue un error de tipeo en las variables del calibrador. Se proporcionó el *traceback* del error y la IA corrigió las listas inicializadas como `dynamic_path_x`.

**8. Prompt de incompatibilidad de OpenCV:**

* **Qué se pidió:** *"Sigue sin funcionar, tira lo que viene siendo trazando ??, de ahi en vez de colocar lo que viene siendo la letra que reconocee coloca el ??..."*
* **Por qué se ajustó:** Se descubrió que `cv2.putText` no soporta caracteres especiales nativos. La IA ajustó la capa de presentación para aplicar un `.replace('Ñ', 'N~')` exclusivamente para el renderizado visual.

**9. Prompts de PyInstaller:**

* **Qué se pidió:** *"Me dio este error [...] ModuleNotFoundError: No module named 'validator'"* y luego *"Tiro el mismo error, no cambio y si elimine todo lo relacionado antes y lo volvi a ejecutar el comando"*
* **Por qué se ajustó:** El uso de `sys.path.append` en el código confundía al empaquetador. La IA tuvo que iterar el comando de terminal usando `--paths "."` y `--hidden-import` para forzar la inyección de los módulos locales al ejecutable `.exe`.

---

## 3. Iteración y mejora

**Expansión masiva de funcionalidades:**
Durante el proceso, el juego pasó de ser un simple validador de letras a un sistema completo. Se le solicitó a la IA la implementación de las siguientes mejoras de forma iterativa:

1. **Sistema de puntuación y rachas:** Se modificó la lógica para implementar un "Modo Arcade" donde al adivinar una palabra y pulsar "Volver a jugar", se mantiene la puntuación acumulada.
2. **Banco de palabras escalable:** Se iteró para pasar de una lista *quemada* en el código a un sistema dinámico que lee y escribe desde un archivo `words.json`.
3. **Mejoras visuales:** A partir de una captura de pantalla del Playground con letras desproporcionadas, se ajustó el tamaño y el grosor de las fuentes en OpenCV, haciendo la UI responsiva al contenido (letras gigantes vs textos medianos).

---

## 4. Validación del resultado

**a. Pruebas del código:**
El código se testeó repetidamente frente a la cámara web. Se validó la interacción de los menús usando el dedo índice como cursor (basado en colisiones espaciales), el reconocimiento de señas dinámicas complejas (J, Z, Ñ), y la robustez del sistema frente a temblores de mano (falsos positivos). Finalmente, se validó el comportamiento del ejecutable compilado en un entorno Windows.

**b. Identificación de límites y ajustes realizados:**

* Se identificó que las trayectorias de la Z se disparaban solas por el temblor natural de la mano. Se ajustó añadiendo una validación geométrica de la hipotenusa: el movimiento debía recorrer al menos el 15% de la pantalla para ser considerado un trazo válido.
* Se identificó la limitación física de MediaPipe al girar el dorso de la mano. El ajuste fue aplicar una "Regla de 45 grados" en la calibración, optimizando el ángulo para el lente de la cámara en lugar del ojo humano.

---

## 5. Reflexión final

**a. Qué se aprendió usando IA para programar:**
La IA es una herramienta excelente para traducir conceptos físicos (como "la U y la V son iguales pero la V tiene los dedos separados") a operaciones matemáticas de código (distancia Euclidiana en 3D). El rol del programador cambia: ya no se trata de conocer la sintaxis de memoria, sino de estructurar la arquitectura del software y saber cómo pedirle a la IA que maneje el flujo de datos.

**b. Ventajas y límites del vibecoding:**

* **Ventajas:** Acelera exponencialmente la solución a cuellos de botella (como compilar con PyInstaller o manejar hilos de video). Permite crear sistemas robustos en una fracción del tiempo tradicional.
* **Límites:** El "robado de estado" o la "oclusión". La IA no sabe cómo se comporta tu cuerpo frente a la cámara ni qué iluminación tienes; si el programa falla porque rotaste la mano, la IA no puede verlo, necesitas diagnosticar físicamente el error y explicárselo con palabras claras para que lo convierta en código.

**c. Qué partes del código comprende y cuáles necesita reforzar:**
Existe una sólida comprensión sobre la estructura del proyecto: la máquina de estados, el flujo de colas (`Queue`), y el renderizado visual con OpenCV. Las áreas a reforzar incluyen los conceptos avanzados de álgebra lineal (cálculos de vectores espaciales y uso de la función `arctan2` para rotación) presentes en el archivo `math_utils.py`.