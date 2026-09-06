# Clasificación de cinco gestos con una IMU

Laboratorio de Inteligencia Artificial de Borde.

Repositorio: [williamolaya03-dotcom/clasificacion-gestos-IMU](https://github.com/williamolaya03-dotcom/clasificacion-gestos-IMU).

**Integrantes:** William David Olaya y Nelson Leonardo Hernandez.

Se utilizó una placa **Arduino Nano BLE Sense colocada en la mano** para registrar
los gestos arriba, abajo, izquierda, derecha y círculo. El procedimiento confirmado
y el alcance de la información disponible están en [PROTOCOLO_CAPTURA.md](PROTOCOLO_CAPTURA.md).

## Dataset
| Dato | Valor verificado |

| Duración útil por grabación | aproximadamente 3 s |
| Frecuencia estimada mediana | 31,25 Hz |
| Ventanas | 600 de aproximadamente 1,5 s |
| Características principales | 48: 6 estadísticos en 8 canales |

Las señales se encuentran en `data/raw/<gesto>/*.csv`. Columnas: timestamp_ms,
ax, ay, az, gx, gy, gz. La aceleración se expresa en g y el giro en grados por
segundo. El archivo básico es `data/processed/caracteristicas_gestos.csv`;
el extendido de 184 características se conserva como material adicional.

## Ejecutar en Colab
1. Subir el contenido de esta carpeta a la raíz del repositorio del grupo.
2. Abrir `notebooks/clasificacion_gestos_imu.ipynb` en Google Colab.
3. La URL está configurada en `REPO_URL`; `RAMA` vacío utiliza la rama predeterminada.
4. Ejecutar todas las celdas en una sesión nueva.
5. Descargar el cuaderno con resultados visibles para subirlo a Moodle.

En la raíz deben aparecer `arduino/`, `data/`, `scripts/`, `notebooks/` y este
README. La URL del grupo está configurada. La ejecución incluida se verificó
con la copia local; la descarga completa se debe comprobar después de subir los archivos.

## Qué hace el cuaderno
Regenera el archivo básico desde las señales, muestra ejemplos limpios y divide
70/30 por grabación. Dentro del 70 % compara pares/tríos originales, PCA y LDA
en 2D y 3D, con las variantes lineal y cuadrática del clasificador propio.
Cada pliegue vuelve a seleccionar características y ajustar transformaciones.
La elección usa validación interna agrupada de cinco pliegues. Después se evalúa
una vez la configuración seleccionada con el 30 % de prueba.

Las fronteras son ecuaciones explícitas de primer o segundo grado; en 3D las
regiones se muestran mediante cortes del mismo clasificador. Se verifican las
diez ecuaciones por pares y se conserva el modelo realmente evaluado.

Los resultados actuales se generan en [RESULTADOS_ACTUALES.md](RESULTADOS_ACTUALES.md)
y `results/`. Los de `historial_version_recibida/` corresponden a configuraciones
anteriores y no deben mezclarse con las métricas de esta versión.

## Ejecución local y nuevas capturas
Todos los comandos se ejecutan desde esta carpeta:

```bash
python -m pip install -r requirements.txt
python scripts/extraer_caracteristicas.py --modo ventana --ventana-s 1.5 --solape 0.5 --conjunto basico
python scripts/calidad_datos.py
```

Para generar de nuevo el conjunto extendido con su propio reporte:

```bash
python scripts/extraer_caracteristicas.py --modo ventana --ventana-s 1.5 --solape 0.5 --conjunto completo --salida data/processed/caracteristicas_extendido.csv --reporte data/processed/reporte_extraccion_extendido.json
```



Referencias: [evitar fuga de información](https://scikit-learn.org/stable/common_pitfalls.html),
[validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html),
[bibliotecas de Arduino según revisión](https://support.arduino.cc/hc/en-us/articles/11729186296476-Use-the-new-sensor-libraries-for-Nano-33-BLE-Rev2-and-Nano-BLE-Sense-Rev2).
