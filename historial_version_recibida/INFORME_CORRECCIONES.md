# Informe de correcciones

Auditoría del proyecto de clasificación de gestos con IMU: qué estaba mal en
`caracteristicas_gestos.csv` y en los scripts que lo generaban, con la evidencia
numérica de cada punto y la corrección aplicada.

Los datos crudos (`data/raw/`) **no se han modificado**. Todo lo que sigue son
errores de cálculo, de configuración o de metodología.

---

## Resumen

| # | Defecto | Gravedad | Estado |
|---|---|---|---|
| 1 | Entropía de Shannon mal calculada (densidades en lugar de probabilidades) | **Crítica** | Corregido |
| 2 | `recording_id` no incluía el gesto: colisionaba entre las 5 clases | **Crítica** | Corregido |
| 3 | El SVM se entrenaba sobre 2 componentes de PCA | **Alta** | Corregido |
| 4 | Frecuencia de muestreo declarada 50 Hz; la real es 31.25 Hz | **Alta** | Corregido |
| 5 | El enventanado descartaba en silencio el 14 % final de cada gesto | Alta | Corregido |
| 6 | Dos grabaciones con discontinuidades temporales no detectadas | Alta | Corregido |
| 7 | La energía es combinación exacta de media y varianza | Media | Documentado |
| 8 | Ningún archivo con NaN se recuperaba: se descartaba entero | Media | Corregido |
| 9 | Estimadores sesgados y mezcla de convenciones (`ddof`) | Media | Corregido |
| 10 | Ninguna característica sensible al orden temporal | Media | Corregido |
| 11 | Una sola partición aleatoria como evaluación | Media | Corregido |
| 12 | `savefig` sobre carpetas inexistentes; versiones fijadas incompatibles con Colab | Baja | Corregido |

---

## 1. Entropía de Shannon mal calculada — **el error grave**

### Código anterior

```python
def calculate_entropy(series):
    hist, _ = np.histogram(series, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))
```

### Por qué está mal

`density=True` devuelve **densidades** $f_b$, que cumplen

$$\sum_{b=1}^{B} f_b \,\Delta = 1, \qquad \Delta = \text{ancho del bin}$$

no probabilidades. La entropía de Shannon exige **probabilidades**:

$$p_b = \frac{n_b}{N}, \qquad \sum_{b=1}^{B} p_b = 1,
\qquad H = -\sum_{b=1}^{B} p_b \log_2 p_b$$

Evaluar $-\sum_b f_b \log f_b$ no es ni entropía de Shannon ni entropía
diferencial (esa sería $-\sum_b f_b \log_2 (f_b)\,\Delta$, con el ancho del bin
como factor). Las consecuencias son tres:

1. **Da valores negativos.** Si el rango del canal es pequeño (acelerómetro,
   ~1 g), entonces $\Delta \ll 1$, luego $f_b > 1$, luego $\log f_b > 0$, luego
   $H < 0$. La entropía de Shannon discreta cumple $H \ge 0$ siempre.
2. **Depende de las unidades.** El resultado cambia —y cambia de signo— según se
   exprese la señal en g, en m/s² o en mg.
3. **Diverge con el número de bins**, en lugar de converger.

### Evidencia

Señal `ax` de `data/raw/abajo/participante_01_sesion_01_rep_001.csv`, la misma
señal física expresada en tres unidades:

| Unidad | Fórmula anterior | Fórmula corregida (bits) |
|---|---|---|
| g | **−6.4866** | 3.3824 |
| m/s² (×9.81) | **+1.9838** | 3.3847 |
| mg (×1000) | **+0.0720** | 3.3929 |

La misma señal, tres respuestas distintas y hasta un cambio de signo. La versión
corregida es invariante al cambio de escala (las diferencias en el cuarto
decimal son efectos de redondeo del propio histograma).

Sensibilidad al número de bins, sobre `ax` en g:

| Bins | Fórmula anterior | $H/\log_2 B$ corregida |
|---|---|---|
| 8 | −4.4873 | 0.8241 |
| 10 | −6.4866 | 0.8076 |
| 16 | −10.1076 | 0.8456 |
| 32 | −24.9906 | 0.8386 |

La versión anterior no converge; la corregida sí.

### Efecto sobre el CSV anterior

Rangos de las columnas `*_entropy` en `caracteristicas_gestos.csv` antes de la
corrección:

| Columna | mín | máx |
|---|---|---|
| `ax_entropy` | −314.60 | 0.62 |
| `az_entropy` | −452.99 | 3.19 |
| `acc_mag_entropy` | −427.79 | 2.96 |
| `gx_entropy` | 0.025 | 1.37 |
| `gyro_mag_entropy` | 0.032 | 1.93 |

Los canales del acelerómetro estaban en $[-453,\,3]$ y los del giroscopio en
$[0,\,2]$ — órdenes de magnitud y signos distintos para la misma magnitud
teórica, sencillamente porque el giroscopio tiene un rango numérico mayor. Tras
estandarizar, esas columnas dominaban la distancia del kernel RBF sin significar
nada.

### Corrección

```python
conteos, _ = np.histogram(x, bins=bins)      # density=False
p = conteos / conteos.sum()                  # probabilidades
p = p[p > 0]                                 # 0*log(0) = 0
H = -np.sum(p * np.log2(p))                  # bits, 0 <= H <= log2(B)
```

Resultado en el CSV nuevo: todas las entropías en **[0.11, 3.93] bits**, dentro
de la cota teórica $\log_2 16 = 4$. `scripts/calidad_datos.py` comprueba esta
cota en cada ejecución.

### Nota honesta sobre la exactitud

Con las mismas 6 características, la versión rota daba **94.5 %** de balanced
accuracy y la corregida da **91.0 %**. La entropía rota funcionaba de forma
accidental como un indicador del rango de la señal, y eso ayudaba a separar
clases.

Eso no es motivo para conservarla: el valor no es interpretable, no es
reproducible (cambia con las unidades y con el número de bins) y no se puede
defender en un informe. Con el conjunto de características ampliado y correcto
se llega a **97.5 %**, por encima de la versión rota y con cada número
justificable.

---

## 2. `recording_id` colisionaba entre gestos

### Código anterior

```python
recording_id = f"{participant_id}_{session_id}_{tokens[5]}"   # -> "01_01_001"
```

El identificador no contenía el gesto, y como los cinco directorios usan los
mismos nombres de archivo (`participante_01_sesion_01_rep_001.csv`), la
repetición 001 de los cinco gestos compartía el id `01_01_001`.

### Evidencia

En el CSV anterior: **200 filas pero solo 40 `recording_id` distintos**. Cada id
agrupaba exactamente 5 filas, una de cada gesto:

```
       gesture recording_id  window_id
0        abajo    01_01_001          0
40      arriba    01_01_001          0
80     circulo    01_01_001          0
120    derecha    01_01_001          0
160  izquierda    01_01_001          0
```

`source_file` tenía el mismo problema: guardaba solo el nombre base, sin la
carpeta del gesto.

### Consecuencia

El cuaderno usaba `GroupShuffleSplit(groups=recording_id)` precisamente para
evitar fuga de datos. Con los ids colisionando, cada grupo contenía las cinco
clases a la vez, así que el agrupamiento no separaba nada de lo que pretendía
separar: solo obligaba a que las cinco clases de una misma repetición cayeran
siempre juntas.

### Corrección

`recording_id = f"{gesto}_p{participante}_s{sesion}_r{rep}"` → 200 ids únicos.
Se añaden `repetition_id` y `segment_id`, y `source_file` pasa a incluir la
carpeta (`abajo/participante_01_...csv`). `calidad_datos.py` verifica que ningún
`recording_id` aparezca en más de un gesto.

---

## 3. El clasificador se entrenaba sobre 2 componentes de PCA

### Código anterior

```python
svm.fit(X_train_pca, y_train)          # X_train_pca tiene 2 columnas
```

El PCA a 2 dimensiones se había calculado para poder dibujar las fronteras de
decisión, y luego se usó como entrada real del clasificador.

### Evidencia

Con el dataset corregido, las 2 primeras componentes principales retienen el
**42.9 %** de la varianza. Hacen falta **31** componentes para llegar al 90 %.
Comparación con validación cruzada agrupada de 5 pliegues:

| Entrada del SVM | Balanced accuracy |
|---|---|
| 184 características | **97.5 % ± 2.2** |
| PCA de 20 componentes | 97.0 % ± 1.9 |
| PCA de 2 componentes | **83.0 % ± 3.7** |

Unos 14 puntos porcentuales perdidos por clasificar sobre una proyección que
existía solo para hacer una figura.

### Corrección

El modelo se entrena con todas las características dentro de un `Pipeline`
(`VarianceThreshold` → `StandardScaler` → `SVC`). El PCA queda reservado para
visualizar, y el cuaderno lo dice explícitamente. Se añade la curva de varianza
acumulada para que la decisión quede justificada con un número.

---

## 4. Frecuencia de muestreo declarada incorrecta

El `.ino`, el README y la constante `FS = 50` del extractor declaraban ~50 Hz.

### Evidencia

Medido sobre las 200 grabaciones: intervalo entre muestras de **32 ms** de
mediana, es decir

$$f_s = \frac{1000\ \text{ms/s}}{32\ \text{ms}} = 31.25\ \text{Hz}$$

Coherente con lo demás: 3.0 s de captura × 31.25 Hz ≈ 94 muestras, y los
archivos tienen entre 92 y 96.

### Por qué ocurría

```cpp
if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) { ... }
delay(18);   // "1000 ms / 50 Hz = 20 ms"
```

`delay(18)` es un retardo **adicional** al tiempo de espera del sensor y al de
impresión por el puerto serie. El periodo total es la suma de los tres y no está
controlado por nada.

### Por qué importa

`FS` solo se usaba en un comentario en el código anterior, así que no afectaba a
las 6 características antiguas. Pero **sí afecta a cualquier característica
espectral**: con $f_s$ mal, la frecuencia dominante $f[k] = k f_s / N$ sale
escalada por un factor 1.6, y la frecuencia de Nyquist real es 15.6 Hz, no 25 Hz.

### Corrección

- El extractor **estima $f_s$ por grabación** a partir de la mediana de $\Delta t$
  y la guarda en la columna `fs_hz`. No hay ninguna constante que suponer.
- El `.ino` usa una planificación de periodo fijo con `micros()`
  (`siguiente_us += PERIODO_US`), que no acumula deriva, y **mide e informa** la
  tasa real del sensor al arrancar en lugar de suponerla.
- README y cuaderno declaran 31.25 Hz.

---

## 5. El enventanado descartaba el final de cada gesto

### Código anterior

```python
WINDOW_SIZE = 80    # comentario: "2 segundos * 50 Hz = 100 muestras"
STEP_SIZE = 40
for start in range(0, num_muestras - WINDOW_SIZE + 1, STEP_SIZE):
```

Tres problemas a la vez:

1. El comentario dice 100 muestras y el código pone 80.
2. Con ~93 muestras por archivo, `range(0, 93-80+1, 40) = range(0, 14, 40)`
   produce **un solo valor**, `start = 0`. Es decir, **una ventana por archivo**,
   no varias. Confirmado: 200 archivos → 200 filas.
3. Esa única ventana cubre las muestras 0 a 79. Las 13 restantes —el **14 %
   final del gesto**— se descartaban en silencio. En un gesto de 3 s, eso es el
   último medio segundo: justo donde termina el movimiento.

Además, todo el texto del cuaderno sobre fuga entre ventanas de una misma
grabación describía una situación que no se estaba dando.

### Corrección

El modo por defecto es `--modo grabacion`: **un vector por archivo**, usando la
grabación completa. Cada archivo ya es un gesto segmentado de 3 s; no hay razón
para partirlo. El modo `--modo ventana` sigue disponible, con la duración
expresada en segundos (se convierte a muestras con la $f_s$ real de cada
grabación) y con una ventana final alineada al extremo para no perder la cola.

---

## 6. Grabaciones con discontinuidades temporales no detectadas

Las únicas comprobaciones del extractor anterior eran
`df.shape[0] < WINDOW_SIZE` y `df.isnull().values.any()`. Dos archivos pasaban
ambas y contenían datos de una captura anterior:

| Archivo | Problema |
|---|---|
| `derecha/participante_01_sesion_01_rep_001.csv` | Salto de **130 623 ms** tras la 2.ª muestra |
| `izquierda/participante_01_sesion_01_rep_001.csv` | Salto de **2 695 ms** tras la 1.ª muestra |

En el primero, las dos primeras filas tienen marcas de tiempo de 498 394 ms y
498 419 ms, y la tercera salta a 629 042 ms: dos épocas distintas separadas por
más de dos minutos, dentro del mismo archivo.

### Causa

`scripts/capturar_serial.py` llamaba a `ser.reset_input_buffer()`, que vacía el
buffer de pyserial pero no necesariamente el del sistema operativo ni el del
conversor USB-serie. La placa transmite de forma continua, también durante la
cuenta atrás y la pausa entre repeticiones, así que quedaban líneas antiguas en
la cola.

### Corrección

- **Al extraer:** se detecta cualquier salto $\Delta t > 5 \times
  \operatorname{mediana}(\Delta t)$, se corta la serie ahí y se conserva el tramo
  continuo más largo. Las grabaciones corregidas quedan listadas en
  `reporte_extraccion.json`.
- **Al capturar:** tras vaciar el buffer se **drena** el puerto durante 0.4 s
  descartando todo lo que llegue, y antes de guardar se valida la continuidad
  temporal. Una repetición con menos de 32 muestras válidas se descarta y se
  avisa para repetirla.

---

## 7. La energía es una combinación exacta de media y varianza

Para cualquier señal se cumple exactamente

$$E = \frac{1}{N}\sum_{i=0}^{N-1} x[i]^2 = \mu^2 + \sigma^2$$

con $\sigma^2$ la varianza poblacional (`ddof=0`). Verificado sobre el CSV
anterior: el residuo máximo $\big|E - (\mu^2 + \sigma^2)\big|$ es del orden de
$10^{-16}$ para `ax`, $10^{-12}$ para `gy` — es decir, cero salvo redondeo de
punto flotante.

Así que 8 de las 48 columnas del dataset anterior (una sexta parte) no aportaban
ninguna información nueva. No invalida el modelo, pero infla la dimensionalidad
y distorsiona la interpretación del PCA, que reparte varianza entre columnas
duplicadas.

Se conserva porque el enunciado pide la energía, pero queda documentado en el
código, en el README y en `reporte_extraccion.json`, y `calidad_datos.py` lista
todos los pares con $|r| > 0.98$.

---

## 8. Un solo NaN descartaba el archivo entero

```python
if df.shape[0] < WINDOW_SIZE or df.isnull().values.any():
    archivos_descartados += 1
    continue
```

Una única celda vacía —una línea truncada por ruido en el puerto serie— tiraba
las 93 muestras del archivo. Con el dataset actual esto no llegó a activarse (no
hay NaN), pero es una pérdida de datos innecesaria en cuanto aparezca uno.

Ahora se eliminan **las filas** afectadas, se ordena por tiempo, se quitan las
marcas de tiempo duplicadas y solo se descarta la grabación si quedan menos de
32 muestras. Los errores de lectura se registran con su motivo, en lugar de
contarse en silencio con un `except: continue`.

---

## 9. Estimadores sesgados y convenciones mezcladas

El código anterior usaba `np.var(data)` (varianza poblacional, `ddof=0`) y
`stats.skew` / `stats.kurtosis` con `bias=True` por defecto. Mezclaba entonces
la convención poblacional y la muestral sin declararlo.

Ahora:

- Varianza y desviación estándar **muestrales** (`ddof=1`):
  $s^2 = \frac{1}{N-1}\sum_i (x[i]-\mu)^2$.
- Asimetría y curtosis **corregidas por sesgo** (`bias=False`):
  $$G_1 = \frac{\sqrt{N(N-1)}}{N-2}\, g_1, \qquad
    G_2 = \frac{(N+1)\,g_2 + 6}{(N-2)(N-3)}\,(N-1)$$
- Se comprueba $N \ge 4$ y $s^2 > 0$ antes de calcular los momentos de orden
  superior, para no producir NaN silenciosos.

---

## 10. Ninguna característica dependía del orden de las muestras

Las 6 características anteriores —media, varianza, curtosis, asimetría,
entropía, energía— son todas **invariantes ante una permutación** de las
muestras. Si se barajara el orden temporal de una grabación, el vector de
características sería idéntico.

Eso es un problema de fondo para este dataset concreto: `arriba` y `abajo` son
aproximadamente el mismo movimiento invertido en el tiempo. La media y la
asimetría capturan parte de la diferencia de signo, pero nada capturaba
la **secuencia**.

Se añaden `_t_argmax` y `_t_argmin` (posición temporal normalizada del máximo y
del mínimo), `_zcr`, `_mean_abs_diff` y las características espectrales. La
importancia del Random Forest sobre el dataset final confirma que sirven:
`gz_t_argmax` y `gz_t_argmin` aparecen entre las 10 más importantes de 184.

---

## 11. La evaluación dependía de una sola partición aleatoria

El cuaderno hacía un único `GroupShuffleSplit(n_splits=1, test_size=0.25)`, es
decir 50 muestras de prueba. Con ese tamaño, cada acierto o fallo mueve la
exactitud 2 puntos porcentuales, y el resultado depende mucho de
`random_state=42`.

Ahora se usa `StratifiedGroupKFold` de 5 pliegues: cada muestra se predice
exactamente una vez estando fuera de su propio entrenamiento, y se reporta media
± desviación estándar. La partición única se mantiene solo como reporte
secundario. El cuaderno comprueba además con un `assert` que no hay grabaciones
compartidas entre entrenamiento y prueba en ningún pliegue.

---

## 12. Detalles operativos

- `plt.savefig('results/figures/...')` se ejecutaba sin garantizar que la carpeta
  existiera. Ahora se crean `results/figures`, `results/metrics` y
  `results/models` al principio del cuaderno.
- `requirements.txt` fijaba versiones exactas (`numpy==1.24.3`,
  `scikit-learn==1.3.0`) incompatibles con las que Colab trae preinstaladas.
  Sustituidas por cotas mínimas.
- Llamadas a `seaborn` con `palette=` sin `hue=`, que emiten avisos de
  obsolescencia en seaborn ≥ 0.13. Corregidas.
- `CARPETA_BASE = '../data/raw'` en el script de captura obligaba a ejecutarlo
  desde `scripts/`, mientras el extractor usaba `data/raw` y había que
  ejecutarlo desde la raíz. Ahora **todos** los scripts se ejecutan desde la raíz.
- El modelo guardado ahora es un diccionario con el `Pipeline`, la lista ordenada
  de columnas de entrada y las clases, para que se pueda usar sin adivinar el
  orden de las características.
- `revision.py` (5 líneas de conteos) se sustituye por `scripts/calidad_datos.py`,
  que audita las señales crudas y la tabla de características, y devuelve un
  código de salida distinto de cero si falla alguna comprobación crítica.

---

## Verificación del resultado

```
$ python scripts/calidad_datos.py

[ok]    clases perfectamente balanceadas (40 grabaciones por gesto)
[ok]    sin NaN ni infinitos
[ok]    ninguna columna constante
[ok]    todas las entropías son no negativas
[ok]    todas las entropías respetan la cota log2(B)
[ok]    |a| media = 1.013 g, coherente con la gravedad
[ok]    cada recording_id pertenece a un único gesto (200 grabaciones)
[ok]    segment_id único en todas las filas
[ok]    el dataset supera todas las comprobaciones críticas
```

Control de coherencia física de las medias por gesto (lo que la versión anterior
no permitía verificar, porque las entropías tapaban todo lo demás):

| Gesto | `az_mean` [g] | `gz_mean` [°/s] | `sma_gyro` [°/s] |
|---|---|---|---|
| abajo | **−0.943** | 0.84 | 50.6 |
| arriba | **+0.967** | 0.53 | 60.8 |
| circulo | 0.296 | **−19.56** | **254.7** |
| derecha | 0.990 | −2.04 | 74.0 |
| izquierda | 0.987 | **+9.26** | 56.7 |

`abajo` y `arriba` se distinguen por el signo de $a_z$ (la placa se orienta hacia
abajo o hacia arriba respecto de la gravedad); `izquierda` y `derecha` por el
signo de $g_z$; `circulo` por una actividad giroscópica global entre tres y cinco
veces mayor. Es exactamente lo que cabe esperar de la física de cada gesto, y
sirve como comprobación independiente de que las características miden lo que se
supone que miden.
