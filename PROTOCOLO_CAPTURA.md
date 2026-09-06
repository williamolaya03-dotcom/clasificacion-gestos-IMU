# Captura del dataset de gestos con IMU

## Integrantes
- William David Olaya
- Nelson Leonardo Hernandez

## Procedimiento confirmado por el grupo
Se utilizó una placa Arduino Nano BLE Sense colocada en la mano para registrar
cinco tipos de movimiento: arriba, abajo, izquierda, derecha y círculo.
La IMU proporcionó señales de aceleración y velocidad angular en tres ejes.
Los registros se organizaron por gesto en archivos CSV para su análisis en Python.

## Datos comprobados en los archivos
Hay 200 grabaciones: 40 por cada gesto. Cada tramo útil dura aproximadamente
3 segundos y la frecuencia estimada mediana es de 31,25 Hz. Las columnas son
timestamp_ms, ax, ay, az, gx, gy y gz. La aceleración está expresada en g y la
velocidad angular en grados por segundo, según el formato documentado del proyecto.
Los archivos utilizan participante_01 y sesion_01. Estos identificadores se
conservan: los nombres de los integrantes son la autoría y no permiten asignar
automáticamente cada grabación a una persona.

## Alcance de la descripción
La información disponible no especifica la mano utilizada, la revisión de la
placa, la orientación inicial, el sentido del círculo ni la identidad del
participante_01. Estos detalles pueden incorporarse si el grupo los recuerda;
no se presentan como condiciones controladas que no quedaron registradas.

## Procesamiento
Se conservan las señales originales. Para extraer características y graficar,
el programa elimina muestras inválidas y conserva el tramo continuo más largo
si detecta saltos de tiempo. En dos archivos se recortan 3 muestras en total.
Se calculan ventanas de aproximadamente 1,5 segundos con solape nominal del 50 %.
El redondeo y la ventana alineada al final pueden modificar ligeramente ese solape.
En el dataset actual se obtienen 600 ventanas: 120 por clase.

Se extraen media, varianza muestral, curtosis, asimetría, entropía y energía media
en ocho canales (seis ejes y dos magnitudes), para un total de 48 características.
La energía utilizada es media(x**2); equivale a energía discreta dividida entre
el número de muestras. La entropía es la del histograma de amplitudes (16 bins),
no una medida directa del orden temporal.
