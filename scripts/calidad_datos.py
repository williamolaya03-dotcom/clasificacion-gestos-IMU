#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control de calidad del dataset de gestos con IMU.

Reemplaza al antiguo  revision.py,  que solo imprimia el balance de clases.
Este script audita las DOS capas del dataset:

  1. Senales crudas   (data/raw/<gesto>/*.csv)
     - numero de muestras, duracion, frecuencia de muestreo real
     - marcas de tiempo no monotonas o duplicadas
     - discontinuidades temporales (buffer serie obsoleto)
     - valores faltantes y saturacion del sensor

  2. Tabla de caracteristicas (data/processed/caracteristicas_gestos.csv)
     - balance de clases, NaN e infinitos
     - columnas constantes (varianza cero) -> inutiles para el clasificador
     - pares de columnas con |r| > UMBRAL_CORR -> redundancia
     - rango de las entropias (deben estar en [0, log2(B)])
     - fugas de agrupamiento: recording_id repetido entre gestos
     - relacion numero de caracteristicas / numero de muestras

Uso:
    python scripts/calidad_datos.py
    python scripts/calidad_datos.py --sin-raw
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DIR_RAW = os.path.join("data", "raw")
ARCHIVO_CARACT = os.path.join("data", "processed", "caracteristicas_gestos.csv")

COLUMNAS_CRUDAS = ["timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]
META = ["gesture", "participant_id", "session_id", "repetition_id",
        "recording_id", "window_id", "segment_id", "source_file",
        "n_muestras", "fs_hz", "duracion_s"]

FACTOR_HUECO = 5.0
UMBRAL_CORR = 0.98
LIMITE_ACC_G = 4.0
LIMITE_GYR_DPS = 2000.0
UMBRAL_SATURACION = 0.98

OK, AVISO, ERROR = "[ok]   ", "[aviso]", "[error]"


def _leer_bins(ruta_reporte: str):
    """Numero de bins usado para la entropia, segun el reporte de extraccion."""
    try:
        import json
        with open(ruta_reporte, encoding="utf-8") as fh:
            return int(json.load(fh).get("bins_entropia"))
    except Exception:
        return None


def titulo(texto: str) -> None:
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ==========================================================================
# 1. SENALES CRUDAS
# ==========================================================================

def auditar_raw(dir_raw: str) -> int:
    titulo("1. SENALES CRUDAS")
    if not os.path.isdir(dir_raw):
        print(f"{ERROR} no existe {dir_raw}")
        return 1

    registros, problemas = [], []
    for gesto in sorted(os.listdir(dir_raw)):
        sub = os.path.join(dir_raw, gesto)
        if not os.path.isdir(sub):
            continue
        for nombre in sorted(os.listdir(sub)):
            if not nombre.lower().endswith(".csv"):
                continue
            ruta = os.path.join(sub, nombre)
            rel = f"{gesto}/{nombre}"
            try:
                df = pd.read_csv(ruta)
            except Exception as exc:
                problemas.append((rel, f"ilegible: {exc}"))
                continue

            faltan = [c for c in COLUMNAS_CRUDAS if c not in df.columns]
            if faltan:
                problemas.append((rel, f"faltan columnas {faltan}"))
                continue

            n_nan = int(df[COLUMNAS_CRUDAS].isnull().sum().sum())
            if n_nan:
                problemas.append((rel, f"{n_nan} valores faltantes"))

            t = df["timestamp_ms"].values.astype(float)
            if np.any(np.diff(t) < 0):
                problemas.append((rel, "marcas de tiempo no monotonas"))
            n_dup = int(df["timestamp_ms"].duplicated().sum())
            if n_dup:
                problemas.append((rel, f"{n_dup} marcas de tiempo duplicadas"))

            dt = np.diff(t)
            if dt.size == 0:
                problemas.append((rel, "menos de 2 muestras"))
                continue
            dt_med = float(np.median(dt))
            fs = 1000.0 / dt_med if dt_med > 0 else np.nan

            huecos = int(np.count_nonzero(dt > FACTOR_HUECO * dt_med))
            if huecos:
                problemas.append(
                    (rel, f"{huecos} discontinuidad(es) temporal(es), "
                          f"hueco maximo {dt.max():.0f} ms "
                          f"(mediana dt = {dt_med:.0f} ms)"))

            sat = int(((df[["ax", "ay", "az"]].abs()
                        >= UMBRAL_SATURACION * LIMITE_ACC_G).any(axis=1) |
                       (df[["gx", "gy", "gz"]].abs()
                        >= UMBRAL_SATURACION * LIMITE_GYR_DPS).any(axis=1)).sum())
            if sat:
                problemas.append((rel, f"{sat} muestra(s) en saturacion del sensor"))

            registros.append({"gesto": gesto, "n": len(df), "fs_hz": fs,
                              "dur_s": (t[-1] - t[0]) / 1000.0})

    if not registros:
        print(f"{ERROR} no se encontro ninguna grabacion valida")
        return 1

    res = pd.DataFrame(registros)
    print(f"\nGrabaciones: {len(res)}   Gestos: {res['gesto'].nunique()}\n")
    resumen = res.groupby("gesto").agg(
        grabaciones=("n", "size"),
        muestras_min=("n", "min"),
        muestras_mediana=("n", "median"),
        muestras_max=("n", "max"),
        fs_mediana_hz=("fs_hz", "median"),
        duracion_mediana_s=("dur_s", "median"))
    print(resumen.round(2).to_string())

    fs_global = float(res["fs_hz"].median())
    print(f"\nFrecuencia de muestreo real (mediana global): {fs_global:.2f} Hz")
    print(f"Rango entre grabaciones: [{res['fs_hz'].min():.2f}, "
          f"{res['fs_hz'].max():.2f}] Hz")

    conteos = res["gesto"].value_counts()
    if conteos.max() - conteos.min() == 0:
        print(f"\n{OK} clases perfectamente balanceadas "
              f"({conteos.min()} grabaciones por gesto)")
    else:
        print(f"\n{AVISO} clases desbalanceadas: min={conteos.min()}, "
              f"max={conteos.max()}. Usa balanced_accuracy en la evaluacion.")

    if problemas:
        print(f"\n{AVISO} {len(problemas)} incidencia(s) en las senales crudas:")
        for rel, msg in problemas:
            print(f"   - {rel}: {msg}")
        print("\n   extraer_caracteristicas.py corrige automaticamente las "
              "discontinuidades\n   quedandose con el tramo continuo mas largo "
              "de cada grabacion.")
    else:
        print(f"\n{OK} sin incidencias en las senales crudas")
    return 0


# ==========================================================================
# 2. TABLA DE CARACTERISTICAS
# ==========================================================================

def auditar_caracteristicas(ruta: str) -> int:
    titulo("2. TABLA DE CARACTERISTICAS")
    if not os.path.isfile(ruta):
        print(f"{ERROR} no existe {ruta}. Ejecuta primero "
              f"scripts/extraer_caracteristicas.py")
        return 1

    df = pd.read_csv(ruta)
    columnas_meta = [c for c in META if c in df.columns]
    X = df.drop(columns=columnas_meta)
    n, p = X.shape

    print(f"\nFilas (segmentos): {n}")
    print(f"Caracteristicas  : {p}")
    print(f"Metadatos        : {columnas_meta}")

    print("\n--- Balance de clases ---")
    print(df["gesture"].value_counts().sort_index().to_string())

    fallos = 0

    # --- Valores no finitos ------------------------------------------------
    n_nan = int(X.isnull().sum().sum())
    n_inf = int(np.isinf(X.select_dtypes(include=[np.number]).values).sum())
    print("\n--- Valores no finitos ---")
    if n_nan == 0 and n_inf == 0:
        print(f"{OK} sin NaN ni infinitos")
    else:
        fallos += 1
        print(f"{ERROR} NaN: {n_nan}   infinitos: {n_inf}")
        cols = X.columns[X.isnull().any()].tolist()
        if cols:
            print(f"        columnas con NaN: {cols[:15]}")

    # --- Columnas constantes ----------------------------------------------
    desv = X.std(ddof=1)
    constantes = desv[desv.fillna(0) < 1e-12].index.tolist()
    print("\n--- Columnas constantes (varianza cero) ---")
    if constantes:
        print(f"{AVISO} {len(constantes)} columna(s) sin informacion: "
              f"{constantes[:15]}")
        print("        StandardScaler las convierte en NaN o cero. "
              "Filtralas con VarianceThreshold.")
    else:
        print(f"{OK} ninguna columna constante")

    # --- Entropias ---------------------------------------------------------
    print("\n--- Entropias ---")
    cols_h = [c for c in X.columns if c.endswith("_entropy")]
    if cols_h:
        vmin, vmax = float(X[cols_h].min().min()), float(X[cols_h].max().max())
        print(f"  H (bits)      : [{vmin:.4f}, {vmax:.4f}]")
        if vmin < -1e-9:
            fallos += 1
            print(f"{ERROR} hay entropias NEGATIVAS. La entropia de Shannon "
                  f"discreta cumple H >= 0.")
            print("        Sintoma tipico de usar np.histogram(..., density=True) "
                  "en vez de\n        probabilidades p_b = n_b / N.")
        else:
            print(f"{OK} todas las entropias son no negativas")
        # Cota superior teorica: H <= log2(B).  B se lee del reporte de
        # extraccion si esta disponible.
        bins = _leer_bins(os.path.join(os.path.dirname(ruta),
                                       "reporte_extraccion.json"))
        if bins:
            cota = np.log2(bins)
            print(f"  cota teorica  : H <= log2({bins}) = {cota:.4f}")
            if vmax > cota + 1e-9:
                fallos += 1
                print(f"{ERROR} alguna entropia supera log2(B): revisa el calculo")
            else:
                print(f"{OK} todas las entropias respetan la cota log2(B)")
    cols_he = [c for c in X.columns if c.endswith("_spec_entropy")]
    if cols_he:
        vmin, vmax = float(X[cols_he].min().min()), float(X[cols_he].max().max())
        print(f"  H espectral   : [{vmin:.4f}, {vmax:.4f}]  (debe estar en [0, 1])")

    # --- Coherencia fisica -------------------------------------------------
    print("\n--- Coherencia fisica ---")
    if "acc_mag_mean" in X.columns:
        m = X["acc_mag_mean"]
        print(f"  |a| media por segmento: [{m.min():.3f}, {m.max():.3f}] g "
              f"(mediana {m.median():.3f} g)")
        if 0.6 <= m.median() <= 1.6:
            print(f"{OK} coherente con la gravedad (~1 g) mas la aceleracion "
                  f"del gesto")
        else:
            print(f"{AVISO} la magnitud media se aleja de 1 g: revisa unidades "
                  f"o calibracion")
    for col, limite, unidad in (("ax_max", LIMITE_ACC_G, "g"),
                                ("gx_max", LIMITE_GYR_DPS, "dps")):
        if col in X.columns and float(X[col].abs().max()) > limite:
            print(f"{AVISO} {col} supera el fondo de escala de {limite} {unidad}")

    # --- Redundancia -------------------------------------------------------
    print(f"\n--- Redundancia entre caracteristicas (|r| > {UMBRAL_CORR}) ---")
    Xn = X.select_dtypes(include=[np.number]).drop(columns=constantes,
                                                   errors="ignore")
    if Xn.shape[1] >= 2:
        R = Xn.corr().abs().values
        iu = np.triu_indices_from(R, k=1)
        pares = [(Xn.columns[i], Xn.columns[j], R[i, j])
                 for i, j in zip(*iu) if np.isfinite(R[i, j]) and R[i, j] > UMBRAL_CORR]
        pares.sort(key=lambda t: -t[2])
        print(f"  pares muy correlacionados: {len(pares)}")
        for a, b, r in pares[:10]:
            print(f"     {a:28s} <-> {b:28s} r = {r:.4f}")
        if len(pares) > 10:
            print(f"     ... y {len(pares) - 10} pares mas")
        if pares:
            print("\n  Recordatorio: energy = mean^2 + varianza_poblacional y "
                  "rms = sqrt(energy),\n  asi que son redundantes por "
                  "construccion. No invalidan el modelo, pero\n  inflan la "
                  "dimensionalidad y sesgan la interpretacion del PCA.")

    # --- Fugas de agrupamiento --------------------------------------------
    print("\n--- Agrupamiento e integridad de los identificadores ---")
    if "recording_id" in df.columns:
        cruce = df.groupby("recording_id")["gesture"].nunique()
        colisiones = cruce[cruce > 1]
        if len(colisiones):
            fallos += 1
            print(f"{ERROR} {len(colisiones)} recording_id aparecen en mas de "
                  f"un gesto.")
            print("        GroupShuffleSplit agrupara clases distintas y el "
                  "agrupamiento pierde sentido.")
            print(f"        Ejemplos: {colisiones.index[:5].tolist()}")
        else:
            print(f"{OK} cada recording_id pertenece a un unico gesto "
                  f"({df['recording_id'].nunique()} grabaciones)")
        vpr = df.groupby("recording_id").size()
        print(f"  segmentos por grabacion: min={vpr.min()}, "
              f"mediana={int(vpr.median())}, max={vpr.max()}")
        if vpr.max() == 1:
            print("  (1 segmento por grabacion: no hay solape entre segmentos, "
                  "asi que\n   una particion estratificada normal ya es honesta; "
                  "el agrupamiento sigue\n   siendo obligatorio en modo ventana.)")
    if "segment_id" in df.columns:
        n_dup = int(df["segment_id"].duplicated().sum())
        if n_dup:
            fallos += 1
            print(f"{ERROR} {n_dup} segment_id duplicados")
        else:
            print(f"{OK} segment_id unico en todas las filas")

    # --- Dimensionalidad ---------------------------------------------------
    print("\n--- Dimensionalidad ---")
    print(f"  caracteristicas / muestras = {p} / {n} = {p / n:.2f}")
    if p >= n:
        print(f"{AVISO} hay tantas o mas caracteristicas que muestras.")
        print("        Es viable, pero exige regularizacion (SVM con C moderado),")
        print("        reduccion de dimension o seleccion de caracteristicas, y")
        print("        validacion cruzada en vez de una unica particion.")
    elif p > n / 5:
        print(f"{AVISO} dimensionalidad alta frente al numero de muestras.")
    else:
        print(f"{OK} relacion razonable")

    titulo("RESULTADO")
    if fallos == 0:
        print(f"{OK} el dataset supera todas las comprobaciones criticas.")
    else:
        print(f"{ERROR} {fallos} comprobacion(es) critica(s) fallida(s).")
    return 0 if fallos == 0 else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Control de calidad del dataset IMU")
    p.add_argument("--raw", default=DIR_RAW)
    p.add_argument("--caracteristicas", default=ARCHIVO_CARACT)
    p.add_argument("--sin-raw", action="store_true",
                   help="Omitir la auditoria de las senales crudas.")
    a = p.parse_args()

    codigo = 0
    if not a.sin_raw:
        codigo |= auditar_raw(a.raw)
    codigo |= auditar_caracteristicas(a.caracteristicas)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
