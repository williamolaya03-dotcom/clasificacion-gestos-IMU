#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraccion de caracteristicas para el dataset de gestos con IMU
(Arduino Nano 33 BLE Sense).

Lee las grabaciones crudas de  data/raw/<gesto>/*.csv  y produce
data/processed/caracteristicas_gestos.csv  con un vector de
caracteristicas por segmento.

--------------------------------------------------------------------------
CONVENCIONES DE NOTACION
--------------------------------------------------------------------------
Para un segmento de N muestras  x[0], x[1], ..., x[N-1]  de un canal:

  media                mu    = (1/N) * sum_{i=0}^{N-1} x[i]

  varianza muestral    s^2   = (1 / (N-1)) * sum_{i=0}^{N-1} (x[i] - mu)^2
  desviacion estandar  s     = sqrt(s^2)

  valor eficaz (RMS)   RMS   = sqrt( (1/N) * sum_{i=0}^{N-1} x[i]^2 )

  energia media        E     = (1/N) * sum_{i=0}^{N-1} x[i]^2 = RMS^2
        Identidad exacta:  E = mu^2 + sigma^2,  con  sigma^2 la varianza
        poblacional (ddof = 0).  Es decir, la energia NO aporta informacion
        nueva si ya se tienen la media y la varianza: es una dependencia
        no lineal de ambas (la media aparece al cuadrado).  Se conserva porque el enunciado del proyecto la
        pide, pero se documenta su colinealidad (ver COL_COLINEALES).

  asimetria (skewness) g1    = m3 / m2^(3/2)
        con  m_k = (1/N) * sum_{i=0}^{N-1} (x[i] - mu)^k
        Se usa la version corregida por sesgo (bias = False) de SciPy:
            G1 = sqrt(N*(N-1)) / (N-2) * g1

  curtosis de Fisher   g2    = m4 / m2^2 - 3
        Version corregida por sesgo (bias = False) de SciPy:
            G2 = ((N+1)*g2 + 6) * (N-1) / ((N-2)*(N-3))
        Con esta convencion, una distribucion normal da  G2 = 0.

  entropia de Shannon  H     = - sum_{b=1}^{B} p_b * log2(p_b)
        donde  p_b = n_b / N  es la PROBABILIDAD del bin b del histograma
        (n_b = numero de muestras que caen en el bin b, B = numero de bins,
        sum_b p_b = 1) y los terminos con  p_b = 0  se omiten porque
        lim_{p->0} p*log2(p) = 0.
        Unidades: bits.  Rango: 0 <= H <= log2(B).
        Se reporta ademas la version normalizada  H_norm = H / log2(B),
        acotada a [0, 1] e independiente del numero de bins.

  tasa de cruces por cero (sobre la senal centrada  y[i] = x[i] - mu):
        ZCR = (1 / (N-1)) * sum_{i=1}^{N-1} 1{ sign(y[i]) != sign(y[i-1]) }

  variacion media absoluta (proxy discreto de la derivada):
        MAD_diff = (1 / (N-1)) * sum_{i=1}^{N-1} | x[i] - x[i-1] |

  desviacion absoluta mediana (robusta):
        MAD = mediana( | x[i] - mediana(x) | )

  rango intercuartilico:
        IQR = Q3 - Q1  (percentiles 75 y 25)

DOMINIO DE LA FRECUENCIA (sobre la senal centrada y enventanada con Hann):
        X[k] = sum_{i=0}^{N-1} w[i] * (x[i] - mu) * exp(-j*2*pi*k*i/N)
        P[k] = |X[k]|^2                      (espectro de potencia)
        f[k] = k * fs / N                    (frecuencia del bin k, en Hz)
        p[k] = P[k] / sum_{m=1}^{K} P[m]     (potencia normalizada, k >= 1)

  frecuencia dominante   f_dom      = f[ argmax_{k>=1} P[k] ]
  centroide espectral    f_centroid = sum_{k>=1} f[k] * p[k]
  ancho de banda         f_spread   = sqrt( sum_{k>=1} (f[k]-f_centroid)^2 * p[k] )
  entropia espectral     H_esp      = ( - sum_{k>=1} p[k]*log2(p[k]) ) / log2(K)

CARACTERISTICAS ENTRE EJES:
  correlacion de Pearson   r_uv = cov(u,v) / (s_u * s_v)
  area de magnitud de senal (SMA), p.ej. para el acelerometro:
        SMA_acc = (1/N) * sum_{i=0}^{N-1} ( |ax[i]| + |ay[i]| + |az[i]| )

--------------------------------------------------------------------------
Uso:
    python scripts/extraer_caracteristicas.py
    python scripts/extraer_caracteristicas.py --modo ventana --ventana-s 1.5
    python scripts/extraer_caracteristicas.py --conjunto basico
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover
    sys.exit("Falta SciPy.  Instala con:  pip install -r requirements.txt")


# ==========================================================================
# CONFIGURACION
# ==========================================================================

DIR_RAW = os.path.join("data", "raw")
DIR_PROCESSED = os.path.join("data", "processed")
ARCHIVO_SALIDA = os.path.join(DIR_PROCESSED, "caracteristicas_gestos.csv")
ARCHIVO_REPORTE = os.path.join(DIR_PROCESSED, "reporte_extraccion.json")

COLUMNAS_REQUERIDAS = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]
EJES_ACC = ["ax", "ay", "az"]
EJES_GYR = ["gx", "gy", "gz"]

# Canales sobre los que se calculan las caracteristicas por senal.
CANALES = EJES_ACC + EJES_GYR + ["acc_mag", "gyro_mag"]

# --- Limpieza -------------------------------------------------------------
# Una discontinuidad temporal es un salto  dt > FACTOR_HUECO * mediana(dt).
# Ocurre cuando el buffer del puerto serie entrega lineas de una captura
# anterior (ver scripts/capturar_serial.py).
FACTOR_HUECO = 5.0
MIN_MUESTRAS = 32          # segmento minimo aceptable tras limpiar
MIN_DT_MS = 1.0            # dt no positivo => marcas de tiempo invalidas

# --- Entropia -------------------------------------------------------------
BINS_ENTROPIA = 16         # B en H = -sum p_b log2 p_b

# --- Saturacion del sensor (para el reporte de calidad, no descarta) ------
LIMITE_ACC_G = 4.0         # fondo de escala tipico del acelerometro (+-4 g)
LIMITE_GYR_DPS = 2000.0    # fondo de escala tipico del giroscopio (+-2000 dps)
UMBRAL_SATURACION = 0.98   # se marca como saturada si |x| >= 0.98 * fondo

# Caracteristicas que son combinacion exacta de otras.
# energia = media^2 + varianza_poblacional  y  rms = sqrt(energia).
COL_COLINEALES = ("energy", "rms")


@dataclass
class Config:
    modo: str = "ventana"        # "grabacion" | "ventana"
    ventana_s: float = 1.5         # duracion de ventana (modo "ventana")
    solape: float = 0.5            # solape fraccional (modo "ventana")
    conjunto: str = "basico"     # "completo" | "basico"
    bins: int = BINS_ENTROPIA
    dir_raw: str = DIR_RAW
    salida: str = ARCHIVO_SALIDA
    reporte: str = ARCHIVO_REPORTE
    incidencias: list = field(default_factory=list)


# ==========================================================================
# ESTADISTICOS ELEMENTALES
# ==========================================================================

def entropia_shannon(x: np.ndarray, bins: int):
    """Entropia de Shannon discreta de la distribucion de amplitudes.

        H = - sum_{b=1}^{B} p_b * log2(p_b),   p_b = n_b / N

    Devuelve (H en bits, H normalizada en [0, 1]).

    NOTA SOBRE EL ERROR CORREGIDO
    -----------------------------
    La version anterior usaba  np.histogram(x, bins=10, density=True), que
    devuelve DENSIDADES  f_b  (con  sum_b f_b * ancho_bin = 1), no
    probabilidades, y luego evaluaba  -sum f_b*log(f_b).  Eso no es la
    entropia de Shannon ni la entropia diferencial:

      * Si el rango del canal es pequeno (acelerometro, ~1 g), entonces
        ancho_bin << 1  =>  f_b > 1  =>  log(f_b) > 0  =>  H < 0.
        De ahi los valores negativos (-314, -452, ...) del CSV anterior.
      * Si el rango es grande (giroscopio, ~2000 dps), entonces  f_b << 1
        y el resultado queda entre 0 y 2.
      * El resultado depende del ANCHO del bin, es decir de las unidades del
        canal, de forma no monotona: no es comparable entre canales.

    La entropia diferencial correcta seria
        h = - sum_b f_b * log2(f_b) * ancho_bin
    pero tambien depende de las unidades y puede ser negativa, por lo que
    aqui se usa la version discreta (probabilidades), que es adimensional,
    esta acotada en [0, log2(B)] y es la habitual para caracterizar la
    "dispersion de forma" de una senal.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan

    rango = float(np.ptp(x))
    if not np.isfinite(rango) or rango == 0.0:
        # Senal constante: toda la masa en un unico bin => H = 0.
        return 0.0, 0.0

    conteos, _ = np.histogram(x, bins=bins)           # density=False  <-- clave
    total = float(conteos.sum())
    if total <= 0:
        return np.nan, np.nan
    p = conteos.astype(float) / total                 # sum_b p_b = 1
    p = p[p > 0.0]                                    # 0*log(0) = 0
    h = float(-np.sum(p * np.log2(p)))
    return h, h / np.log2(bins)


def tasa_cruces_cero(x: np.ndarray) -> float:
    """ZCR sobre la senal centrada:  (1/(N-1)) * numero de cambios de signo."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    y = x - np.mean(x)
    s = np.sign(y)
    s[s == 0] = 1.0                     # los ceros exactos no cuentan
    return float(np.count_nonzero(np.diff(s) != 0) / (x.size - 1))


def caracteristicas_espectrales(x: np.ndarray, fs: float) -> dict:
    """Frecuencia dominante, centroide, ancho de banda y entropia espectral.

    Se centra la senal (se elimina la componente de continua) y se aplica
    una ventana de Hann antes de la FFT para reducir la fuga espectral.
    El bin k = 0 se excluye de todas las metricas porque, tras centrar,
    no lleva informacion util.
    """
    vacio = {"f_dom": np.nan, "f_centroid": np.nan,
             "f_spread": np.nan, "spec_entropy": np.nan}
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8 or not np.isfinite(fs) or fs <= 0:
        return vacio

    y = (x - np.mean(x)) * np.hanning(n)
    potencia = np.abs(np.fft.rfft(y)) ** 2
    frecuencias = np.fft.rfftfreq(n, d=1.0 / fs)

    potencia = potencia[1:]          # descartar la componente de continua
    frecuencias = frecuencias[1:]
    if potencia.size == 0:
        return vacio
    total = float(potencia.sum())
    if total <= 0.0:
        return vacio

    p = potencia / total             # sum_k p[k] = 1
    f_dom = float(frecuencias[int(np.argmax(potencia))])
    f_centroid = float(np.sum(frecuencias * p))
    f_spread = float(np.sqrt(np.sum(((frecuencias - f_centroid) ** 2) * p)))

    p_pos = p[p > 0.0]
    h = float(-np.sum(p_pos * np.log2(p_pos)))
    h_norm = h / np.log2(p.size) if p.size > 1 else 0.0

    return {"f_dom": f_dom, "f_centroid": f_centroid,
            "f_spread": f_spread, "spec_entropy": h_norm}


def caracteristicas_canal(x: np.ndarray, fs: float, cfg: Config) -> dict:
    """Vector de caracteristicas de un unico canal."""
    x = np.asarray(x, dtype=float)
    n = x.size

    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else np.nan      # varianza muestral
    std = float(np.sqrt(var)) if np.isfinite(var) else np.nan
    energia = float(np.mean(x ** 2))                          # = mu^2 + var_pob

    f = {
        "mean": mu,
        "variance": var,
        "std": std,
        "energy": energia,
        "rms": float(np.sqrt(energia)),
    }

    # Momentos de orden superior: la correccion de sesgo de la curtosis
    # requiere N >= 4 y varianza no nula.
    if n >= 4 and np.isfinite(var) and var > 0:
        f["skewness"] = float(sp_stats.skew(x, bias=False))
        f["kurtosis"] = float(sp_stats.kurtosis(x, fisher=True, bias=False))
    else:
        f["skewness"] = np.nan
        f["kurtosis"] = np.nan

    # Entropia en bits, acotada en [0, log2(B)].  No se emite ademas la
    # version normalizada H/log2(B) porque, con B fijo, es la misma columna
    # multiplicada por una constante (correlacion exacta r = 1) y solo
    # duplicaria la dimensionalidad.
    f["entropy"] = entropia_shannon(x, cfg.bins)[0]

    if cfg.conjunto == "basico":
        # Las 6 caracteristicas del enunciado original, ya corregidas.
        return {k: f[k] for k in
                ("mean", "variance", "kurtosis", "skewness", "entropy", "energy")}

    # --- Conjunto completo -------------------------------------------------
    q1, mediana, q3 = np.percentile(x, [25, 50, 75])
    f["min"] = float(np.min(x))
    f["max"] = float(np.max(x))
    f["range"] = float(np.ptp(x))
    f["median"] = float(mediana)
    f["iqr"] = float(q3 - q1)
    f["mad"] = float(np.median(np.abs(x - mediana)))
    f["zcr"] = tasa_cruces_cero(x)
    f["mean_abs_diff"] = float(np.mean(np.abs(np.diff(x)))) if n > 1 else np.nan
    # Posicion temporal normalizada del pico: 0 = inicio, 1 = fin del segmento.
    # Es una familia sensible al ORDEN; tambien lo son ZCR, diferencias y espectro,
    # necesaria para distinguir gestos que son inversos en el tiempo
    # (por ejemplo "arriba" frente a "abajo").
    f["t_argmax"] = float(np.argmax(x) / (n - 1)) if n > 1 else np.nan
    f["t_argmin"] = float(np.argmin(x) / (n - 1)) if n > 1 else np.nan

    f.update(caracteristicas_espectrales(x, fs))
    return f


def correlacion(u: np.ndarray, v: np.ndarray) -> float:
    """Coeficiente de Pearson  r = cov(u,v) / (s_u * s_v), robusto a s = 0."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.size < 3:
        return np.nan
    su, sv = np.std(u, ddof=1), np.std(v, ddof=1)
    if su == 0.0 or sv == 0.0:
        return np.nan
    cov = float(np.sum((u - u.mean()) * (v - v.mean())) / (u.size - 1))
    return cov / (su * sv)


# ==========================================================================
# PROCESAMIENTO DE UN SEGMENTO
# ==========================================================================

def procesar_segmento(seg: pd.DataFrame, fs: float, metadatos: dict,
                      cfg: Config) -> dict:
    caract = dict(metadatos)
    caract["n_muestras"] = int(len(seg))
    caract["fs_hz"] = float(fs)
    caract["duracion_s"] = float(len(seg) / fs) if fs > 0 else np.nan

    for canal in CANALES:
        for nombre, valor in caracteristicas_canal(seg[canal].values, fs, cfg).items():
            caract[f"{canal}_{nombre}"] = valor

    if cfg.conjunto == "completo":
        # Correlaciones entre ejes: capturan la ORIENTACION del movimiento en
        # el plano, que ninguna caracteristica de un solo canal describe.
        for a, b in (("ax", "ay"), ("ax", "az"), ("ay", "az"),
                     ("gx", "gy"), ("gx", "gz"), ("gy", "gz")):
            caract[f"corr_{a}_{b}"] = correlacion(seg[a].values, seg[b].values)

        # Area de magnitud de senal: intensidad global del movimiento.
        caract["sma_acc"] = float(
            np.mean(np.abs(seg["ax"]) + np.abs(seg["ay"]) + np.abs(seg["az"])))
        caract["sma_gyro"] = float(
            np.mean(np.abs(seg["gx"]) + np.abs(seg["gy"]) + np.abs(seg["gz"])))

    return caract


# ==========================================================================
# CARGA Y LIMPIEZA DE UNA GRABACION
# ==========================================================================

def cargar_y_limpiar(ruta: str, cfg: Config):
    """Devuelve (dataframe limpio o None, fs estimada, diagnostico)."""
    diag = {"archivo": ruta.replace(os.sep, "/"), "motivo": None,
            "n_original": 0, "n_final": 0, "muestras_eliminadas": 0,
            "nan_eliminados": 0, "hueco_max_ms": 0.0, "saturacion": 0}

    try:
        df = pd.read_csv(ruta)
    except Exception as exc:
        diag["motivo"] = f"ilegible: {exc}"
        return None, np.nan, diag

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        diag["motivo"] = f"faltan columnas: {faltantes}"
        return None, np.nan, diag

    df = df[COLUMNAS_REQUERIDAS].apply(pd.to_numeric, errors="coerce")
    diag["n_original"] = int(len(df))

    # 1) Eliminar solo las FILAS con NaN, no el archivo completo.
    n_antes = len(df)
    df = df.dropna().reset_index(drop=True)
    diag["nan_eliminados"] = int(n_antes - len(df))

    # 2) Ordenar por tiempo y quitar marcas de tiempo duplicadas.
    df = (df.sort_values("timestamp_ms")
            .drop_duplicates(subset="timestamp_ms", keep="first")
            .reset_index(drop=True))

    if len(df) < MIN_MUESTRAS:
        diag["motivo"] = f"muy corto tras limpiar ({len(df)} < {MIN_MUESTRAS})"
        return None, np.nan, diag

    # 3) Cortar discontinuidades temporales y quedarse con el tramo continuo
    #    mas largo.  Detecta el buffer serie obsoleto (saltos de segundos).
    dt = np.diff(df["timestamp_ms"].values.astype(float))
    dt_mediana = float(np.median(dt))
    diag["hueco_max_ms"] = float(dt.max()) if dt.size else 0.0

    if dt_mediana < MIN_DT_MS:
        diag["motivo"] = f"marcas de tiempo invalidas (dt mediano = {dt_mediana} ms)"
        return None, np.nan, diag

    cortes = np.where(dt > FACTOR_HUECO * dt_mediana)[0]
    if cortes.size:
        limites = [0, *(cortes + 1), len(df)]
        tramos = [(limites[i], limites[i + 1]) for i in range(len(limites) - 1)]
        ini, fin = max(tramos, key=lambda t: t[1] - t[0])
        df = df.iloc[ini:fin].reset_index(drop=True)
        if len(df) < MIN_MUESTRAS:
            diag["motivo"] = ("tramo continuo mas largo demasiado corto "
                              f"({len(df)} < {MIN_MUESTRAS})")
            return None, np.nan, diag
        dt = np.diff(df["timestamp_ms"].values.astype(float))
        dt_mediana = float(np.median(dt))

    diag["muestras_eliminadas"] = diag["n_original"] - len(df)

    # 4) Frecuencia de muestreo estimada a partir de la MEDIANA de dt
    #    (robusta frente a retrasos puntuales del puerto serie):
    #        fs = 1000 / mediana(dt)   [Hz]
    fs = 1000.0 / dt_mediana

    # 5) Magnitudes (invariantes a la orientacion del dispositivo):
    #        |a| = sqrt(ax^2 + ay^2 + az^2)
    #        |g| = sqrt(gx^2 + gy^2 + gz^2)
    df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    df["gyro_mag"] = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)

    # 6) Marcar saturacion del sensor (se informa, no se descarta).
    sat_acc = (df[EJES_ACC].abs() >= UMBRAL_SATURACION * LIMITE_ACC_G).any(axis=1)
    sat_gyr = (df[EJES_GYR].abs() >= UMBRAL_SATURACION * LIMITE_GYR_DPS).any(axis=1)
    diag["saturacion"] = int((sat_acc | sat_gyr).sum())
    diag["n_final"] = int(len(df))
    diag["fs_hz"] = fs

    return df, fs, diag


# ==========================================================================
# PRINCIPAL
# ==========================================================================

def listar_archivos(dir_raw: str):
    rutas = []
    for gesto in sorted(os.listdir(dir_raw)):
        sub = os.path.join(dir_raw, gesto)
        if not os.path.isdir(sub):
            continue
        for nombre in sorted(os.listdir(sub)):
            if nombre.lower().endswith(".csv"):
                rutas.append(os.path.join(sub, nombre))
    return rutas


def main(cfg: Config) -> int:
    print("=== EXTRACCION DE CARACTERISTICAS ===")
    print(f"modo={cfg.modo}  conjunto={cfg.conjunto}  bins_entropia={cfg.bins}")

    if not os.path.isdir(cfg.dir_raw):
        print(f"ERROR: no existe el directorio {cfg.dir_raw}")
        return 1

    os.makedirs(os.path.dirname(cfg.salida) or ".", exist_ok=True)
    rutas = listar_archivos(cfg.dir_raw)
    if not rutas:
        print(f"ERROR: no se encontraron CSV en {cfg.dir_raw}/<gesto>/")
        return 1

    filas, diagnosticos, descartados, fs_estimadas = [], [], [], []

    for ruta in rutas:
        gesto = os.path.basename(os.path.dirname(ruta))
        nombre = os.path.basename(ruta)
        raiz = os.path.splitext(nombre)[0]

        tokens = raiz.split("_")
        if len(tokens) >= 6 and tokens[0] == "participante":
            participante, sesion, rep = tokens[1], tokens[3], tokens[5]
        else:
            participante = sesion = rep = "NA"

        # CORRECCION IMPORTANTE
        # ---------------------
        # El identificador anterior era  f"{participante}_{sesion}_{rep}",
        # que NO incluye el gesto: la repeticion 001 de los 5 gestos
        # compartia el id "01_01_001".  Con GroupShuffleSplit eso obliga a
        # que las 5 clases caigan siempre en la misma particion y destruye
        # el sentido del agrupamiento.  Ahora el gesto forma parte del id.
        recording_id = f"{gesto}_p{participante}_s{sesion}_r{rep}"
        source_file = os.path.join(gesto, nombre).replace(os.sep, "/")

        df, fs, diag = cargar_y_limpiar(ruta, cfg)
        diag["gesture"] = gesto
        diagnosticos.append(diag)

        if df is None:
            descartados.append(diag)
            continue

        fs_estimadas.append(fs)

        # --- Definicion de los segmentos ----------------------------------
        if cfg.modo == "grabacion":
            # Cada archivo es UN gesto ya segmentado (~3 s): un vector por
            # grabacion.  Evita partir el gesto por la mitad y evita
            # descartar el tramo final, como hacia el ventaneo anterior.
            segmentos = [(0, len(df))]
        else:
            largo = max(MIN_MUESTRAS, int(round(cfg.ventana_s * fs)))
            paso = max(1, int(round(largo * (1.0 - cfg.solape))))
            if len(df) < largo:
                segmentos = [(0, len(df))]      # no se descarta la grabacion
            else:
                segmentos = [(i, i + largo)
                             for i in range(0, len(df) - largo + 1, paso)]
                # Cola final: si sobran muestras suficientes se anade una
                # ultima ventana alineada al final.  El codigo anterior
                # descartaba silenciosamente hasta el 14 % del gesto.
                if (segmentos[-1][1] < len(df)
                        and len(df) - segmentos[-1][1] >= paso // 2):
                    segmentos.append((len(df) - largo, len(df)))

        for idx, (ini, fin) in enumerate(segmentos):
            metadatos = {
                "gesture": gesto,
                "participant_id": participante,
                "session_id": sesion,
                "repetition_id": rep,
                "recording_id": recording_id,
                "window_id": idx,
                "segment_id": f"{recording_id}_w{idx:02d}",
                "source_file": source_file,
            }
            filas.append(procesar_segmento(
                df.iloc[ini:fin].reset_index(drop=True), fs, metadatos, cfg))

    if not filas:
        print("ERROR: no se extrajo ningun segmento valido.")
        return 1

    salida = pd.DataFrame(filas)

    # Orden estable de columnas: metadatos primero, caracteristicas despues.
    meta = ["gesture", "participant_id", "session_id", "repetition_id",
            "recording_id", "window_id", "segment_id", "source_file",
            "n_muestras", "fs_hz", "duracion_s"]
    resto = [c for c in salida.columns if c not in meta]
    salida = salida[meta + sorted(resto)]
    salida.to_csv(cfg.salida, index=False)

    # ------------------------- Reporte ------------------------------------
    fs_arr = np.array(fs_estimadas, dtype=float)
    reporte = {
        "modo": cfg.modo,
        "conjunto": cfg.conjunto,
        "bins_entropia": cfg.bins,
        "ventana_s": cfg.ventana_s if cfg.modo == "ventana" else None,
        "solape": cfg.solape if cfg.modo == "ventana" else None,
        "archivo_caracteristicas": cfg.salida.replace(os.sep, "/"),
        "archivos_encontrados": len(rutas),
        "archivos_procesados": len(rutas) - len(descartados),
        "archivos_descartados": len(descartados),
        "segmentos": int(len(salida)),
        "n_caracteristicas": int(len(resto)),
        "nan_en_salida": int(salida.isnull().sum().sum()),
        "fs_hz_mediana": round(float(np.median(fs_arr)), 3) if fs_arr.size else None,
        "fs_hz_min": round(float(fs_arr.min()), 3) if fs_arr.size else None,
        "fs_hz_max": round(float(fs_arr.max()), 3) if fs_arr.size else None,
        "segmentos_por_clase": salida["gesture"].value_counts().to_dict(),
        "grabaciones_corregidas": [
            {k: d[k] for k in ("archivo", "muestras_eliminadas", "hueco_max_ms")}
            for d in diagnosticos if d.get("muestras_eliminadas", 0) > 0],
        "grabaciones_con_saturacion": [
            {"archivo": d["archivo"], "muestras_saturadas": d["saturacion"]}
            for d in diagnosticos if d.get("saturacion", 0) > 0],
        "descartados": descartados,
        "caracteristicas_colineales": {
            "descripcion": ("energy = mean^2 + varianza_poblacional  y  "
                            "rms = sqrt(energy).  Se conservan por requisito "
                            "del enunciado, pero son redundantes."),
            "sufijos": list(COL_COLINEALES),
        },
    }
    with open(cfg.reporte, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    print("\n=== RESUMEN ===")
    print(f"Archivos encontrados : {reporte['archivos_encontrados']}")
    print(f"Archivos procesados  : {reporte['archivos_procesados']}")
    print(f"Archivos descartados : {reporte['archivos_descartados']}")
    print(f"Segmentos extraidos  : {reporte['segmentos']}")
    print(f"Caracteristicas      : {reporte['n_caracteristicas']}")
    print(f"NaN en la salida     : {reporte['nan_en_salida']}")
    print(f"fs estimada (Hz)     : mediana={reporte['fs_hz_mediana']}  "
          f"[{reporte['fs_hz_min']}, {reporte['fs_hz_max']}]")
    print("Segmentos por clase  :")
    for clase, conteo in sorted(reporte["segmentos_por_clase"].items()):
        print(f"   - {clase:<10} {conteo}")
    if reporte["grabaciones_corregidas"]:
        print("\nGrabaciones con discontinuidades corregidas:")
        for d in reporte["grabaciones_corregidas"]:
            print(f"   - {d['archivo']}: -{d['muestras_eliminadas']} muestras "
                  f"(hueco maximo {d['hueco_max_ms']:.0f} ms)")
    if reporte["grabaciones_con_saturacion"]:
        print("\nGrabaciones con muestras en saturacion del sensor:")
        for d in reporte["grabaciones_con_saturacion"]:
            print(f"   - {d['archivo']}: {d['muestras_saturadas']} muestra(s)")
    if descartados:
        print("\nDescartados:")
        for d in descartados:
            print(f"   - {d['archivo']}: {d['motivo']}")
    print(f"\nCSV      -> {cfg.salida}")
    print(f"Reporte  -> {cfg.reporte}")
    return 0


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Extraccion de caracteristicas IMU")
    p.add_argument("--modo", choices=["grabacion", "ventana"], default="ventana",
                   help="grabacion: un vector por archivo. "
                        "ventana: enventanado deslizante (por defecto).")
    p.add_argument("--ventana-s", type=float, default=1.5,
                   help="Duracion de la ventana en segundos (modo ventana).")
    p.add_argument("--solape", type=float, default=0.5,
                   help="Solape fraccional entre ventanas, en [0, 1).")
    p.add_argument("--conjunto", choices=["completo", "basico"], default="basico",
                   help="completo: todas las caracteristicas. "
                        "basico: solo las 6 del enunciado, corregidas.")
    p.add_argument("--bins", type=int, default=BINS_ENTROPIA,
                   help="Numero de bins del histograma para la entropia.")
    p.add_argument("--raw", default=DIR_RAW)
    p.add_argument("--salida", default=ARCHIVO_SALIDA)
    p.add_argument("--reporte", default=ARCHIVO_REPORTE)
    a = p.parse_args()
    if not 0.0 <= a.solape < 1.0:
        p.error("--solape debe estar en [0, 1)")
    if a.bins < 2:
        p.error("--bins debe ser >= 2")
    return Config(modo=a.modo, ventana_s=a.ventana_s, solape=a.solape,
                  conjunto=a.conjunto, bins=a.bins, dir_raw=a.raw,
                  salida=a.salida, reporte=a.reporte)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
