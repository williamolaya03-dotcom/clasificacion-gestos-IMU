# Clasificación de gestos mediante IMU (Arduino Nano 33 BLE Sense)

Dataset y sistema de clasificación de **5 gestos** (`abajo`, `arriba`, `circulo`,
`derecha`, `izquierda`) a partir de las señales del acelerómetro y el giroscopio
de un Arduino Nano 33 BLE Sense.

---

## 1. Estado del dataset

| Concepto | Valor |
|---|---|
| Gestos | 5 |
| Grabaciones | 200 (40 por gesto — clases perfectamente balanceadas) |
| Duración por grabación | ≈ 2.98 s |
| Muestras por grabación | 92 – 96 (mediana 93) |
| **Frecuencia de muestreo real** | **31.25 Hz** (mediana de `1000 / dt`) |
| Ancho de banda útil (Nyquist) | 15.6 Hz |
| Enventanado | 1.5 s con 50 % de solape → 47 muestras/ventana, 3 ventanas/grabación |
| Filas del archivo de características | 600 ventanas (120 por gesto) |
| Características por ventana | 48 = 6 estadísticos × 8 canales |
| Participantes | 1 |

> **Sobre la frecuencia de muestreo:** la documentación anterior decía «~50 Hz».
> Es incorrecto. El intervalo entre muestras en los archivos ya grabados es de
> 32 ms de mediana, es decir **31.25 Hz**. Los datos existentes son válidos; lo
> que había que corregir era la cifra declarada, porque de ella dependen todas
> las características en el dominio de la frecuencia.

---

## 2. Estructura del repositorio

```
.
├── arduino/captura_imu/captura_imu.ino   Firmware de adquisición
├── scripts/
│   ├── capturar_serial.py                Captura por puerto serie -> data/raw/
│   ├── extraer_caracteristicas.py        data/raw/ -> data/processed/
│   ├── calidad_datos.py                  Auditoría del dataset
│   ├── dataset_utils.py                  Descargar / cargar / visualizar
│   └── clasificador_geometrico.py        Clasificador 2D/3D + búsqueda de proyección
├── notebooks/clasificacion_gestos_imu.ipynb
├── data/
│   ├── raw/<gesto>/participante_XX_sesion_YY_rep_ZZZ.csv
│   └── processed/
│       ├── caracteristicas_gestos.csv    El dataset de características
│       └── reporte_extraccion.json       Trazabilidad de la extracción
├── results/{figures,metrics,models}
├── INFORME_CORRECCIONES.md               Qué estaba mal y por qué
└── requirements.txt
```

**Todos los scripts se ejecutan desde la raíz del repositorio**, no desde
`scripts/`.

---

## 3. Uso rápido

```bash
pip install -r requirements.txt

# 1) Generar el archivo de características (ventanas de 1.5 s, 6 estadísticos)
python scripts/extraer_caracteristicas.py \
    --modo ventana --ventana-s 1.5 --solape 0.5 --conjunto basico

# 2) Auditar la calidad del resultado
python scripts/calidad_datos.py

# 3) Generar todas las figuras de exploración
python scripts/dataset_utils.py --local .
```

Descargar el dataset desde GitHub y visualizarlo, en tres líneas:

```python
from scripts.dataset_utils import descargar_dataset, cargar_caracteristicas, graficar_pca

raiz = descargar_dataset("https://github.com/USUARIO/REPO")
df   = cargar_caracteristicas(raiz)
graficar_pca(df)
```

---

## 4. Formato de los datos

### 4.1 Señales crudas — `data/raw/<gesto>/*.csv`

| Columna | Unidad | Descripción |
|---|---|---|
| `timestamp_ms` | ms | Milisegundos desde el arranque de la placa |
| `ax`, `ay`, `az` | g | Aceleración (1 g = 9.81 m/s²); incluye la gravedad |
| `gx`, `gy`, `gz` | °/s | Velocidad angular |

### 4.2 Características — `data/processed/caracteristicas_gestos.csv`

**Metadatos (11 columnas):** `gesture`, `participant_id`, `session_id`,
`repetition_id`, `recording_id`, `window_id`, `segment_id`, `source_file`,
`n_muestras`, `fs_hz`, `duracion_s`.

`recording_id` tiene la forma `<gesto>_p01_s01_r001` y es **único por
grabación**: es la columna de agrupamiento para partir los datos sin fuga.

**Características (48):** 8 canales × las 6 características que pide el enunciado
(media, varianza, curtosis, simetría, entropía, energía). El conjunto ampliado de
184 características está a un flag de distancia (`--conjunto completo`).

**Canales:** `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `acc_mag`, `gyro_mag`, donde

$$\|a\| = \sqrt{a_x^2 + a_y^2 + a_z^2}, \qquad
  \|\omega\| = \sqrt{g_x^2 + g_y^2 + g_z^2}$$

**Las 6 del enunciado** (conjunto por defecto), para un segmento de $N$ muestras
$x[0],\dots,x[N-1]$:

| Sufijo | Definición |
|---|---|
| `_mean` | $\mu = \dfrac{1}{N}\displaystyle\sum_{i=0}^{N-1} x[i]$ |
| `_variance` | $s^2 = \dfrac{1}{N-1}\displaystyle\sum_{i=0}^{N-1}(x[i]-\mu)^2$ |
| `_kurtosis` | $g_2 = \dfrac{m_4}{m_2^{2}} - 3$ (Fisher), corregida por sesgo |
| `_skewness` | $g_1 = \dfrac{m_3}{m_2^{3/2}}$, corregida por sesgo, con $m_k = \dfrac{1}{N}\sum_i (x[i]-\mu)^k$ |
| `_entropy` | $H = -\displaystyle\sum_{b=1}^{B} p_b \log_2 p_b$, con $p_b = n_b/N$ y $B = 16$ |
| `_energy` | $E = \dfrac{1}{N}\displaystyle\sum_{i=0}^{N-1} x[i]^2$ |

**Adicionales del conjunto completo** (`--conjunto completo`, 184 columnas):

| Sufijo | Definición |
|---|---|
| `_std` | $s = \sqrt{s^2}$ |
| `_rms` | $\text{RMS} = \sqrt{E}$ |
| `_min`, `_max`, `_range` | $\min x$, $\max x$, $\max x - \min x$ |
| `_median`, `_iqr` | mediana y $Q_3 - Q_1$ |
| `_mad` | $\operatorname{mediana}\big(|x[i] - \operatorname{mediana}(x)|\big)$ |
| `_zcr` | $\dfrac{1}{N-1}\displaystyle\sum_{i=1}^{N-1}\mathbb{1}\{\operatorname{sign}(y[i]) \ne \operatorname{sign}(y[i-1])\}$, con $y = x - \mu$ |
| `_mean_abs_diff` | $\dfrac{1}{N-1}\displaystyle\sum_{i=1}^{N-1}\big|x[i]-x[i-1]\big|$ |
| `_t_argmax`, `_t_argmin` | posición del máximo/mínimo, normalizada a $[0,1]$ |
| `_f_dom` | $f[\,\arg\max_{k\ge 1} P[k]\,]$ |
| `_f_centroid` | $\displaystyle\sum_{k\ge1} f[k]\,p[k]$ |
| `_f_spread` | $\sqrt{\displaystyle\sum_{k\ge1}(f[k]-f_\text{centroid})^2\,p[k]}$ |
| `_spec_entropy` | $-\displaystyle\sum_{k\ge1} p[k]\log_2 p[k] \;\big/\; \log_2 K$ |

Para las características espectrales, con ventana de Hann $w[i]$:

$$X[k] = \sum_{i=0}^{N-1} w[i]\,(x[i]-\mu)\,e^{-j2\pi k i/N},
\qquad P[k] = |X[k]|^2,
\qquad f[k] = \frac{k\,f_s}{N},
\qquad p[k] = \frac{P[k]}{\sum_{m\ge1} P[m]}$$

**Entre ejes:** `corr_ax_ay`, `corr_ax_az`, `corr_ay_az`, `corr_gx_gy`,
`corr_gx_gz`, `corr_gy_gz` (Pearson), y

$$\text{SMA}_\text{acc} = \frac{1}{N}\sum_{i=0}^{N-1}\big(|a_x[i]|+|a_y[i]|+|a_z[i]|\big)$$

`_t_argmax` y `_t_argmin` son las únicas características **sensibles al orden
temporal** de las muestras. Las 6 del enunciado son todas invariantes ante una
permutación: si se barajara el orden temporal de una ventana, el vector de
características sería idéntico.

**Redundancia conocida:** $E = \mu^2 + \sigma^2$ (con $\sigma^2$ la varianza
poblacional) y $\text{RMS} = \sqrt{E}$, así que `_energy` y `_rms` son
combinaciones exactas de `_mean` y `_variance`. Se conservan porque el
enunciado pide la energía, pero `scripts/calidad_datos.py` las señala.

---

## 5. Extracción de características: opciones

```bash
# El comando que genera el CSV del proyecto:
# ventanas de 1.5 s con 50 % de solape, las 6 características del enunciado
python scripts/extraer_caracteristicas.py \
    --modo ventana --ventana-s 1.5 --solape 0.5 --conjunto basico

# Conjunto ampliado (184 características), por si se quiere explorar más
python scripts/extraer_caracteristicas.py \
    --modo ventana --ventana-s 1.5 --conjunto completo \
    --salida data/processed/caracteristicas_extendido.csv

# Un vector por grabación completa, sin enventanar (200 filas)
python scripts/extraer_caracteristicas.py --modo grabacion
```

Con enventanado, las 3 ventanas de una grabación comparten `recording_id` y son
casi idénticas entre sí, así que **es obligatorio** partir los datos agrupando
por esa columna. Repartirlas entre entrenamiento y prueba infla la exactitud
(medido en este dataset: 94.5 % sin agrupar frente a 93.0 % agrupando).

El extractor limpia automáticamente cada grabación: elimina filas con valores
faltantes, ordena por tiempo, quita marcas de tiempo duplicadas y **corta las
discontinuidades temporales** quedándose con el tramo continuo más largo. Todo
queda registrado en `data/processed/reporte_extraccion.json`.

---

## 6. Captura de nuevos datos

1. Instala la librería correcta desde el gestor de librerías del IDE de Arduino:
   `Arduino_LSM9DS1` (Nano 33 BLE Sense original) o `Arduino_BMI270_BMM150`
   (REV2), y descomenta la línea correspondiente en el `.ino`.
2. Carga `arduino/captura_imu/captura_imu.ino`.
3. Abre el Monitor Serie **una sola vez** para leer la línea
   `# tasa_maxima_medida_hz,<valor>`: es la tasa real que entrega el sensor.
   Ajusta `FS_OBJETIVO_HZ` en el sketch a un valor **igual o menor** y vuelve a
   cargarlo. **Cierra el Monitor Serie** antes del paso siguiente (bloquea el
   puerto).
4. Desde la raíz del repositorio: `python scripts/capturar_serial.py`.
5. Haz el movimiento de forma fluida y consistente, 30–40 repeticiones por gesto.
   Varía la mano y la postura entre sesiones; si todas las repeticiones se hacen
   seguidas sin soltar la placa, el modelo aprende la postura y no el gesto.
6. Vuelve a ejecutar `extraer_caracteristicas.py` y `calidad_datos.py`.

Para que el dataset sea más sólido conviene añadir **más de un participante**:
con uno solo, la exactitud medida no dice nada sobre cómo generalizará a otra
persona.

---

## 7. Cuaderno de clasificación

1. Sube el repositorio completo a GitHub.
2. Abre `notebooks/clasificacion_gestos_imu.ipynb` en Google Colab.
3. Cambia `REPO_URL` por la URL de tu repositorio.
4. *Entorno de ejecución* → *Ejecutar todas*.

El cuaderno sigue los cuatro puntos del enunciado: clona el repositorio, elige la
proyección 2D/3D que mejor separa las clases, parte el dataset agrupando por
grabación y entrena un clasificador geométrico programado desde cero en
`scripts/clasificador_geometrico.py`.

### Elección del espacio reducido

Criterio de Fisher $J = \operatorname{traza}(S_W^{-1} S_B)$ sobre las 600 ventanas:

| Proyección | $J$ | Balanced accuracy (VC agrupada) |
|---|---|---|
| **LDA-3D** | **138.3** | **83.0 %** |
| LDA-2D | 135.7 | 74.2 % |
| Mejor trío crudo (`ax_energy`, `az_mean`, `gyro_mag_mean`) | 56.7 | 78.7 % |
| PCA-3D | 7.8 | 72.3 % |
| PCA-2D | 4.8 | 48.8 % |

LDA gana con mucha diferencia porque es el único método que maximiza $J$
directamente; PCA maximiza la varianza total, que no es lo mismo que separar
clases.

Un detalle a tener en cuenta: $J$ es una **traza**, así que está dominado por
las primeras direcciones. Por eso LDA-2D y LDA-3D tienen $J$ casi idéntico
(135.7 frente a 138.3) pero se clasifican muy distinto (74.2 % frente a 83.0 %):
la tercera dirección aporta poco a la traza global y sin embargo es la que separa
`derecha` de `izquierda`. Conviene usar $J$ para preseleccionar candidatos y
decidir con validación cruzada.

### Resultados

Validación cruzada agrupada de 5 pliegues, métrica *balanced accuracy*:

| Clasificador | Balanced accuracy |
|---|---|
| SVM-RBF en LDA-3D *(referencia externa)* | 87.0 % ± 1.7 |
| **Gaussiano lineal en LDA-3D** *(programado en el proyecto)* | **85.5 % ± 3.8** |
| **Gaussiano cuadrático en LDA-3D** *(programado en el proyecto)* | **83.0 % ± 2.5** |
| Gaussiano cuadrático en LDA-2D | 74.2 % ± 1.6 |

El clasificador propio queda a 1.5–4 puntos del SVM de scikit-learn, pero sus
fronteras son ecuaciones explícitas que se pueden escribir en el informe.

`derecha` e `izquierda` son el par que más se confunde: son el mismo movimiento
lateral y se distinguen sobre todo por el signo de $g_z$.

Estas cifras corresponden a **un solo participante y una sola sesión**. No son
una estimación del rendimiento con un usuario nuevo.

---

## 8. Correcciones aplicadas

`INFORME_CORRECCIONES.md` detalla los 12 defectos encontrados en la versión
anterior, con la evidencia numérica de cada uno. El más grave: la entropía
estaba mal calculada y producía valores negativos de hasta −452.

---

## Integrantes

- [NOMBRES DE LOS INTEGRANTES]
