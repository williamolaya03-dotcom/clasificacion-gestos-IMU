#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades para DESCARGAR, CARGAR y VISUALIZAR el dataset de gestos con IMU.

Pensado para usarse desde Google Colab, desde un notebook local o desde la
terminal.  No requiere que el repositorio este clonado previamente.

Ejemplo minimo
--------------
    from scripts.dataset_utils import descargar_dataset, cargar_caracteristicas
    from scripts.dataset_utils import graficar_gestos_ejemplo, graficar_pca

    raiz = descargar_dataset("https://github.com/USUARIO/REPO")
    df   = cargar_caracteristicas(raiz)
    graficar_gestos_ejemplo(raiz)
    graficar_pca(df)

Desde la terminal
-----------------
    python scripts/dataset_utils.py --repo https://github.com/USUARIO/REPO
    python scripts/dataset_utils.py --local . --figuras results/figures
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("MPLBACKEND") is None and not hasattr(sys, "ps1"):
    matplotlib.use("Agg")          # permite ejecutar sin entorno grafico
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Constantes del dataset
# --------------------------------------------------------------------------
GESTOS = ["abajo", "arriba", "circulo", "derecha", "izquierda"]
EJES_ACC = ["ax", "ay", "az"]
EJES_GYR = ["gx", "gy", "gz"]

RUTA_CARACTERISTICAS = os.path.join("data", "processed", "caracteristicas_gestos.csv")
RUTA_RAW = os.path.join("data", "raw")

# Columnas que NO son caracteristicas (identificadores y metadatos).
COLUMNAS_META = ["gesture", "participant_id", "session_id", "repetition_id",
                 "recording_id", "window_id", "segment_id", "source_file",
                 "n_muestras", "fs_hz", "duracion_s"]

_COLORES = {"abajo": "#d62728", "arriba": "#2ca02c", "circulo": "#9467bd",
            "derecha": "#1f77b4", "izquierda": "#ff7f0e"}


# ==========================================================================
# 1. DESCARGA
# ==========================================================================

def _normalizar_repo(repo_url: str) -> str:
    """Quita el sufijo .git y las barras finales de la URL del repositorio."""
    url = repo_url.strip().rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def descargar_dataset(repo_url: str,
                      destino: str | None = None,
                      rama: str = "main",
                      metodo: str = "auto",
                      forzar: bool = False) -> str:
    """Descarga el repositorio del dataset y devuelve la ruta a su raiz.

    Parametros
    ----------
    repo_url : URL de GitHub, por ejemplo
               "https://github.com/usuario/clasificacion-gestos-imu"
    destino  : carpeta local donde dejar el repositorio.  Si es None se usa
               el nombre del repositorio.
    rama     : rama a descargar ("main", "master", ...).
    metodo   : "zip"  descarga el archivo comprimido por HTTPS (no necesita
                      git instalado; es lo mas rapido en Colab),
               "git"  usa `git clone --depth 1`,
               "auto" intenta "zip" y, si falla, "git".
    forzar   : si es True y la carpeta destino ya existe, la borra y vuelve
               a descargar.  Si es False y ya existe, no descarga nada.

    Devuelve
    --------
    Ruta (str) a la raiz del repositorio descargado.
    """
    url = _normalizar_repo(repo_url)
    if destino is None:
        destino = url.rsplit("/", 1)[-1]

    if os.path.isdir(destino):
        if not forzar:
            print(f"[descarga] '{destino}' ya existe; se reutiliza. "
                  f"Usa forzar=True para volver a descargar.")
            return destino
        shutil.rmtree(destino)

    intentos = ["zip", "git"] if metodo == "auto" else [metodo]
    errores = []
    for m in intentos:
        try:
            if m == "zip":
                _descargar_zip(url, destino, rama)
            elif m == "git":
                _descargar_git(url, destino, rama)
            else:
                raise ValueError(f"metodo desconocido: {m}")
            print(f"[descarga] listo ({m}) -> {os.path.abspath(destino)}")
            return destino
        except Exception as exc:
            errores.append(f"{m}: {exc}")

    raise RuntimeError(
        "No se pudo descargar el repositorio.\n  " + "\n  ".join(errores) +
        "\nComprueba la URL, que el repositorio sea publico y el nombre de "
        "la rama (main / master).")


def _descargar_zip(url: str, destino: str, rama: str) -> None:
    zip_url = f"{url}/archive/refs/heads/{rama}.zip"
    print(f"[descarga] bajando {zip_url}")
    with urllib.request.urlopen(zip_url, timeout=120) as respuesta:
        datos = respuesta.read()
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        raiz_zip = z.namelist()[0].split("/")[0]
        temporal = destino + "__tmp"
        shutil.rmtree(temporal, ignore_errors=True)
        z.extractall(temporal)
    shutil.move(os.path.join(temporal, raiz_zip), destino)
    shutil.rmtree(temporal, ignore_errors=True)


def _descargar_git(url: str, destino: str, rama: str) -> None:
    print(f"[descarga] git clone --depth 1 -b {rama} {url}")
    subprocess.run(["git", "clone", "--depth", "1", "-b", rama,
                    f"{url}.git", destino],
                   check=True, capture_output=True, text=True)


# ==========================================================================
# 2. CARGA
# ==========================================================================

def cargar_caracteristicas(raiz: str = ".") -> pd.DataFrame:
    """Carga data/processed/caracteristicas_gestos.csv desde `raiz`."""
    ruta = os.path.join(raiz, RUTA_CARACTERISTICAS)
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No se encontro {ruta}.\nEjecuta primero: "
            f"python scripts/extraer_caracteristicas.py")
    df = pd.read_csv(ruta)
    print(f"[carga] {len(df)} segmentos, "
          f"{len(separar_X_y(df)[0].columns)} caracteristicas, "
          f"{df['gesture'].nunique()} gestos")
    return df


def separar_X_y(df: pd.DataFrame):
    """Devuelve (X, y, grupos): caracteristicas, etiqueta y recording_id."""
    meta = [c for c in COLUMNAS_META if c in df.columns]
    X = df.drop(columns=meta)
    y = df["gesture"]
    grupos = df["recording_id"] if "recording_id" in df.columns else None
    return X, y, grupos


def listar_grabaciones(raiz: str = ".", gesto: str | None = None):
    """Lista las rutas de las grabaciones crudas, opcionalmente de un gesto."""
    base = os.path.join(raiz, RUTA_RAW)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No se encontro {base}")
    gestos = [gesto] if gesto else sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    rutas = []
    for g in gestos:
        carpeta = os.path.join(base, g)
        if not os.path.isdir(carpeta):
            continue
        rutas += [os.path.join(carpeta, f)
                  for f in sorted(os.listdir(carpeta)) if f.endswith(".csv")]
    return rutas


def cargar_senal(ruta: str) -> pd.DataFrame:
    """Carga una grabacion cruda y anade tiempo relativo y magnitudes.

        t[i]    = (timestamp_ms[i] - timestamp_ms[0]) / 1000     [s]
        |a|[i]  = sqrt(ax[i]^2 + ay[i]^2 + az[i]^2)              [g]
        |g|[i]  = sqrt(gx[i]^2 + gy[i]^2 + gz[i]^2)              [dps]
    """
    try:
        from .extraer_caracteristicas import Config, cargar_y_limpiar
    except ImportError:
        from extraer_caracteristicas import Config, cargar_y_limpiar
    df, fs, diagnostico = cargar_y_limpiar(ruta, Config())
    if df is None:
        raise ValueError(f"No se puede visualizar {ruta}: {diagnostico['motivo']}")
    df["t_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    return df


def resumen_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por gesto: conteo, duracion y magnitudes medias."""
    columnas = [c for c in ("duracion_s", "n_muestras", "fs_hz",
                            "acc_mag_mean", "gyro_mag_mean", "sma_gyro")
                if c in df.columns]
    resumen = df.groupby("gesture")[columnas].mean().round(3)
    resumen.insert(0, "segmentos", df["gesture"].value_counts().sort_index())
    return resumen


# ==========================================================================
# 3. VISUALIZACION
# ==========================================================================

def _guardar(fig, guardar_en: str | None):
    if guardar_en:
        os.makedirs(os.path.dirname(guardar_en) or ".", exist_ok=True)
        fig.savefig(guardar_en, dpi=140, bbox_inches="tight")
        print(f"[figura] {guardar_en}")
    return fig


def graficar_senal(ruta: str, guardar_en: str | None = None):
    """Acelerometro, giroscopio y magnitudes de UNA grabacion."""
    df = cargar_senal(ruta)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for eje, color in zip(EJES_ACC, ("#d62728", "#2ca02c", "#1f77b4")):
        ax[0].plot(df["t_s"], df[eje], label=eje, color=color, lw=1.3)
    ax[0].set_ylabel("Aceleracion [g]")
    ax[0].set_title(os.path.basename(ruta))

    for eje, color in zip(EJES_GYR, ("#d62728", "#2ca02c", "#1f77b4")):
        ax[1].plot(df["t_s"], df[eje], label=eje, color=color, lw=1.3)
    ax[1].set_ylabel("Vel. angular [dps]")

    ax[2].plot(df["t_s"], df["acc_mag"], label="|a| [g]", color="#111111", lw=1.4)
    ax2b = ax[2].twinx()
    ax2b.plot(df["t_s"], df["gyro_mag"], label="|g| [dps]",
              color="#9467bd", lw=1.2, alpha=0.8)
    ax[2].set_ylabel("|a| [g]")
    ax2b.set_ylabel("|g| [dps]")
    ax[2].set_xlabel("Tiempo [s]")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    ax2b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_gestos_ejemplo(raiz: str = ".", repeticion: int = 1,
                            canal: str = "acc", guardar_en: str | None = None):
    """Una grabacion de cada gesto, en una rejilla comparativa.

    canal : "acc" (ax, ay, az) o "gyro" (gx, gy, gz).
    """
    ejes = EJES_ACC if canal == "acc" else EJES_GYR
    unidad = "g" if canal == "acc" else "dps"

    base = os.path.join(raiz, RUTA_RAW)
    gestos = sorted(d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)))
    fig, axes = plt.subplots(len(gestos), 1, figsize=(11, 2.2 * len(gestos)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for a, gesto in zip(axes, gestos):
        rutas = listar_grabaciones(raiz, gesto)
        if not rutas:
            continue
        ruta = rutas[min(repeticion - 1, len(rutas) - 1)]
        df = cargar_senal(ruta)
        for eje, color in zip(ejes, ("#d62728", "#2ca02c", "#1f77b4")):
            a.plot(df["t_s"], df[eje], label=eje, color=color, lw=1.2)
        a.set_ylabel(f"{gesto}\n[{unidad}]", fontsize=9)
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("Tiempo [s]")
    fig.suptitle(f"Comparacion de gestos - {'acelerometro' if canal == 'acc' else 'giroscopio'}"
                 f" (repeticion {repeticion})", y=1.0)
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_distribucion_clases(df: pd.DataFrame, guardar_en: str | None = None):
    """Numero de segmentos por gesto."""
    conteos = df["gesture"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(conteos.index, conteos.values,
           color=[_COLORES.get(g, "#777777") for g in conteos.index])
    for i, v in enumerate(conteos.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Segmentos")
    ax.set_title("Distribucion de clases")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_caracteristicas(df: pd.DataFrame, columnas=None,
                             guardar_en: str | None = None):
    """Diagramas de caja de varias caracteristicas, separados por gesto."""
    if columnas is None:
        columnas = [c for c in ("acc_mag_std", "gyro_mag_std", "az_mean",
                                "gz_mean", "sma_gyro", "acc_mag_entropy")
                    if c in df.columns]
    columnas = [c for c in columnas if c in df.columns]
    if not columnas:
        raise ValueError("Ninguna de las columnas indicadas existe en el DataFrame")

    n = len(columnas)
    filas = int(np.ceil(n / 3))
    fig, axes = plt.subplots(filas, min(n, 3), figsize=(5 * min(n, 3), 3.6 * filas),
                             squeeze=False)
    gestos = sorted(df["gesture"].unique())

    for k, col in enumerate(columnas):
        a = axes[k // 3][k % 3]
        datos = [df.loc[df["gesture"] == g, col].dropna().values for g in gestos]
        try:                      # Matplotlib >= 3.9
            bp = a.boxplot(datos, tick_labels=gestos, patch_artist=True, widths=0.6)
        except TypeError:         # Matplotlib < 3.9
            bp = a.boxplot(datos, labels=gestos, patch_artist=True, widths=0.6)
        for parche, g in zip(bp["boxes"], gestos):
            parche.set_facecolor(_COLORES.get(g, "#cccccc"))
            parche.set_alpha(0.65)
        for mediana in bp["medians"]:
            mediana.set_color("black")
        a.set_title(col, fontsize=10)
        a.tick_params(axis="x", rotation=45, labelsize=8)
        a.grid(axis="y", alpha=0.3)

    for k in range(n, filas * min(n, 3)):
        axes[k // 3][k % 3].axis("off")
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_matriz_correlacion(df: pd.DataFrame, columnas=None, maximo: int = 30,
                                guardar_en: str | None = None):
    """Matriz de correlacion de Pearson entre caracteristicas."""
    X, _, _ = separar_X_y(df)
    if columnas is not None:
        X = X[[c for c in columnas if c in X.columns]]
    elif X.shape[1] > maximo:
        # Se muestran las de mayor varianza tras estandarizar por rango.
        rango = X.max() - X.min()
        X = X[rango.sort_values(ascending=False).index[:maximo]]

    R = X.corr()
    fig, ax = plt.subplots(figsize=(0.42 * len(R) + 3, 0.42 * len(R) + 2))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(R)))
    ax.set_yticks(range(len(R)))
    ax.set_xticklabels(R.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(R.columns, fontsize=7)
    ax.set_title(f"Correlacion de Pearson ({len(R)} caracteristicas)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_pca(df: pd.DataFrame, guardar_en: str | None = None):
    """Proyeccion PCA a 2 componentes, coloreada por gesto.

    Solo para VISUALIZAR.  No se debe entrenar el clasificador unicamente
    sobre estas 2 componentes: se pierde la mayor parte de la varianza.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X, y, _ = separar_X_y(df)
    X = X.loc[:, X.std(ddof=1) > 1e-12]           # descartar columnas constantes
    Xs = StandardScaler().fit_transform(X)        # z[i,j] = (x[i,j]-mu_j)/s_j
    Z = PCA(n_components=2, random_state=42)
    P = Z.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for gesto in sorted(y.unique()):
        m = (y == gesto).values
        ax.scatter(P[m, 0], P[m, 1], label=gesto, s=42, alpha=0.8,
                   edgecolor="k", linewidth=0.4,
                   color=_COLORES.get(gesto, None))
    var = Z.explained_variance_ratio_ * 100
    ax.set_xlabel(f"CP1 ({var[0]:.1f} % de la varianza)")
    ax.set_ylabel(f"CP2 ({var[1]:.1f} % de la varianza)")
    ax.set_title(f"PCA - varianza acumulada en 2 componentes: {var.sum():.1f} %")
    ax.legend(title="Gesto")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, guardar_en)


def graficar_espectro(raiz: str = ".", repeticion: int = 1, canal: str = "gyro_mag",
                      guardar_en: str | None = None):
    """Densidad espectral de potencia de un canal, un gesto por curva.

        X[k] = FFT{ (x[i] - media(x)) * hann[i] }
        P[k] = |X[k]|^2 / sum_m |X[m]|^2      (potencia normalizada)
        f[k] = k * fs / N
    """
    base = os.path.join(raiz, RUTA_RAW)
    gestos = sorted(d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)))
    fig, ax = plt.subplots(figsize=(9, 5))
    fs = np.nan

    for gesto in gestos:
        rutas = listar_grabaciones(raiz, gesto)
        if not rutas:
            continue
        df = cargar_senal(rutas[min(repeticion - 1, len(rutas) - 1)])
        x = df[canal].values.astype(float)
        dt = np.median(np.diff(df["timestamp_ms"].values.astype(float)))
        fs = 1000.0 / dt
        y = (x - x.mean()) * np.hanning(len(x))
        P = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(x), d=1.0 / fs)
        P, f = P[1:], f[1:]
        if P.sum() > 0:
            P = P / P.sum()
        ax.plot(f, P, label=gesto, lw=1.5, color=_COLORES.get(gesto, None))

    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Potencia normalizada")
    ax.set_title(f"Espectro de '{canal}' (fs = {fs:.1f} Hz, "
                 f"Nyquist = {fs / 2:.1f} Hz)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _guardar(fig, guardar_en)


# ==========================================================================
# 4. CLI
# ==========================================================================

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Descarga, carga y visualiza el dataset de gestos IMU")
    p.add_argument("--repo", help="URL del repositorio de GitHub a descargar.")
    p.add_argument("--local", help="Raiz de un repositorio ya presente en disco.")
    p.add_argument("--rama", default="main")
    p.add_argument("--figuras", default=os.path.join("results", "figures"),
                   help="Carpeta donde guardar las figuras.")
    a = p.parse_args()

    if not a.repo and not a.local:
        p.error("indica --repo <url> o --local <ruta>")

    raiz = a.local if a.local else descargar_dataset(a.repo, rama=a.rama)
    df = cargar_caracteristicas(raiz)

    print("\n=== RESUMEN POR GESTO ===")
    print(resumen_dataset(df).to_string())

    fig_dir = a.figuras
    graficar_distribucion_clases(df, os.path.join(fig_dir, "distribucion_clases.png"))
    graficar_gestos_ejemplo(raiz, canal="acc",
                            guardar_en=os.path.join(fig_dir, "senales_acc.png"))
    graficar_gestos_ejemplo(raiz, canal="gyro",
                            guardar_en=os.path.join(fig_dir, "senales_gyro.png"))
    graficar_espectro(raiz, guardar_en=os.path.join(fig_dir, "espectro_gyro_mag.png"))
    graficar_caracteristicas(df, guardar_en=os.path.join(fig_dir, "caracteristicas.png"))
    graficar_matriz_correlacion(df, guardar_en=os.path.join(fig_dir, "correlacion.png"))
    graficar_pca(df, guardar_en=os.path.join(fig_dir, "pca.png"))
    plt.close("all")
    print(f"\nFiguras guardadas en {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
