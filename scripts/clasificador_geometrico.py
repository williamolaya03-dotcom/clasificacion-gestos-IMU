#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador geometrico en un espacio reducido de 2 o 3 dimensiones, y
busqueda de la proyeccion que mejor separa las clases.

Responde a dos puntos del enunciado:

  * "Grafica de dispersion con la agrupacion de caracteristicas que arroje la
     mejor distribucion de clases en un espacio 2D o 3D"
        -> buscar_mejor_proyeccion()

  * "Proponer y programar un clasificador a partir de funciones que limiten
     el espacio 2D o 3D"
        -> ClasificadorGaussianoCuadratico, implementado desde cero con NumPy.

--------------------------------------------------------------------------
EL CLASIFICADOR: FRONTERAS CUADRATICAS EXPLICITAS
--------------------------------------------------------------------------
Se modela cada clase c con una distribucion normal multivariante en el
espacio reducido z (de dimension D = 2 o 3):

    p(z | c) = 1 / ( (2*pi)^(D/2) * |S_c|^(1/2) )
               * exp( -1/2 * (z - mu_c)^T * S_c^(-1) * (z - mu_c) )

Aplicando la regla de Bayes y tomando logaritmo, la funcion discriminante
de la clase c es (se descartan los terminos constantes comunes):

    g_c(z) = -1/2 * (z - mu_c)^T * S_c^(-1) * (z - mu_c)
             -1/2 * ln|S_c|
             + ln P(c)

donde
    mu_c  = (1/N_c) * sum_{i in c} z_i                  vector de medias
    S_c   = (1/(N_c - 1)) * sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
    P(c)  = N_c / N                                     probabilidad a priori

La decision es    clase(z) = argmax_c  g_c(z).

LA FRONTERA entre dos clases i y j es el lugar geometrico donde ambas
funciones se igualan:

    g_i(z) - g_j(z) = 0

Desarrollando el producto matricial, esa diferencia es un polinomio de
segundo grado en z:

    f_ij(z) = z^T * A_ij * z  +  b_ij^T * z  +  c_ij  =  0

con
    A_ij = -1/2 * ( S_i^(-1) - S_j^(-1) )
    b_ij = S_i^(-1) * mu_i - S_j^(-1) * mu_j
    c_ij = -1/2 * ( mu_i^T S_i^(-1) mu_i - mu_j^T S_j^(-1) mu_j )
           -1/2 * ( ln|S_i| - ln|S_j| )
           + ln P(i) - ln P(j)

Es decir, cada par de clases queda separado por una **conica** en 2D
(elipse, parabola o hiperbola) o una **cuadrica** en 3D (elipsoide,
paraboloide o hiperboloide).  Son literalmente "funciones que limitan el
espacio", y el metodo coeficientes_frontera() las devuelve para poder
escribirlas en el informe.

Caso particular: si se fuerza una unica matriz de covarianza comun para
todas las clases (parametro covarianza="comun"), entonces  S_i = S_j,  con
lo cual  A_ij = 0  y la frontera se vuelve un HIPERPLANO:

    b_ij^T * z + c_ij = 0

REGULARIZACION
--------------
S_c se estima con pocas muestras por clase, asi que puede quedar mal
condicionada.  Se aplica un encogimiento hacia la covarianza global:

    S_c_reg = (1 - lambda) * S_c + lambda * S_global

con  0 <= lambda <= 1  (parametro `reg`).
"""

from __future__ import annotations

import itertools

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:                                        # pragma: no cover
    plt = None


# ==========================================================================
# CLASIFICADOR
# ==========================================================================

class ClasificadorGaussianoCuadratico:
    """Clasificador de fronteras cuadraticas, implementado desde cero.

    Parametros
    ----------
    reg : float en [0, 1]
        Encogimiento de la covarianza de cada clase hacia la global.
    covarianza : "propia" | "comun"
        "propia" -> una S_c por clase; fronteras cuadraticas (conicas).
        "comun"  -> una unica S para todas; fronteras lineales (rectas/planos).
    """

    def __init__(self, reg: float = 0.05, covarianza: str = "propia"):
        if not 0.0 <= reg <= 1.0:
            raise ValueError("reg debe estar en [0, 1]")
        if covarianza not in ("propia", "comun"):
            raise ValueError('covarianza debe ser "propia" o "comun"')
        self.reg = reg
        self.covarianza = covarianza

    # ---------------------------------------------------------------- ajuste
    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y)
        if Z.ndim != 2:
            raise ValueError("Z debe ser una matriz (n_muestras, n_dimensiones)")

        self.clases_ = np.unique(y)
        self.n_dim_ = Z.shape[1]

        # Covarianza global, usada como referencia de la regularizacion.
        S_global = np.cov(Z.T, ddof=1)
        if S_global.ndim == 0:
            S_global = S_global.reshape(1, 1)

        # Covarianza comun agrupada (pooled), si se pide frontera lineal:
        #   S = (1/(N - K)) * sum_c sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
        if self.covarianza == "comun":
            acumulado = np.zeros((self.n_dim_, self.n_dim_))
            for c in self.clases_:
                A = Z[y == c]
                D = A - A.mean(axis=0)
                acumulado += D.T @ D
            S_comun = acumulado / (len(Z) - len(self.clases_))

        self.mu_, self.S_, self.S_inv_, self.logdet_, self.log_prior_ = {}, {}, {}, {}, {}

        for c in self.clases_:
            A = Z[y == c]
            if len(A) <= self.n_dim_:
                raise ValueError(
                    f"La clase '{c}' tiene {len(A)} muestras y el espacio tiene "
                    f"{self.n_dim_} dimensiones: no alcanza para estimar S_c.")

            self.mu_[c] = A.mean(axis=0)

            if self.covarianza == "comun":
                S = S_comun.copy()
            else:
                S = np.cov(A.T, ddof=1)
                if S.ndim == 0:
                    S = S.reshape(1, 1)
                # Encogimiento:  S_reg = (1 - lambda) S_c + lambda S_global
                S = (1.0 - self.reg) * S + self.reg * S_global

            self.S_[c] = S
            self.S_inv_[c] = np.linalg.inv(S)
            signo, logdet = np.linalg.slogdet(S)
            if signo <= 0:
                raise np.linalg.LinAlgError(
                    f"La covarianza de la clase '{c}' no es definida positiva. "
                    f"Sube `reg` o reduce la dimension.")
            self.logdet_[c] = logdet
            self.log_prior_[c] = np.log(len(A) / len(Z))

        return self

    # ----------------------------------------------------------- prediccion
    def funciones_discriminantes(self, Z):
        """Matriz (n_muestras, n_clases) con g_c(z) para cada muestra."""
        Z = np.asarray(Z, dtype=float)
        columnas = []
        for c in self.clases_:
            D = Z - self.mu_[c]
            # Distancia de Mahalanobis al cuadrado, muestra a muestra:
            #   d2_i = (z_i - mu_c)^T * S_c^(-1) * (z_i - mu_c)
            d2 = np.einsum("ij,jk,ik->i", D, self.S_inv_[c], D)
            columnas.append(-0.5 * d2 - 0.5 * self.logdet_[c] + self.log_prior_[c])
        return np.column_stack(columnas)

    def predict(self, Z):
        return self.clases_[np.argmax(self.funciones_discriminantes(Z), axis=1)]

    def predict_proba(self, Z):
        """Probabilidades a posteriori mediante softmax de los discriminantes."""
        G = self.funciones_discriminantes(Z)
        G = G - G.max(axis=1, keepdims=True)      # estabilidad numerica
        E = np.exp(G)
        return E / E.sum(axis=1, keepdims=True)

    def score(self, Z, y):
        return float(np.mean(self.predict(Z) == np.asarray(y)))

    # ------------------------------------------------------------- fronteras
    def coeficientes_frontera(self, clase_i, clase_j):
        """Coeficientes (A, b, c) de la frontera  z^T A z + b^T z + c = 0.

        Devuelve un diccionario con A_ij, b_ij y c_ij según las fórmulas del
        encabezado del módulo.  En 2D, con z = (z1, z2), la ecuación explícita es

            A[0,0]*z1^2 + 2*A[0,1]*z1*z2 + A[1,1]*z2^2
            + b[0]*z1 + b[1]*z2 + c = 0
        """
        Si, Sj = self.S_inv_[clase_i], self.S_inv_[clase_j]
        mi, mj = self.mu_[clase_i], self.mu_[clase_j]

        A = -0.5 * (Si - Sj)
        b = Si @ mi - Sj @ mj
        c = (-0.5 * (mi @ Si @ mi - mj @ Sj @ mj)
             - 0.5 * (self.logdet_[clase_i] - self.logdet_[clase_j])
             + self.log_prior_[clase_i] - self.log_prior_[clase_j])

        return {"A": A, "b": b, "c": float(c),
                "tipo": "lineal" if np.allclose(A, 0, atol=1e-12) else "cuadratica"}

    def imprimir_fronteras(self, decimales: int = 4):
        """Escribe todas las fronteras por pares en forma legible."""
        for ci, cj in itertools.combinations(self.clases_, 2):
            f = self.coeficientes_frontera(ci, cj)
            A, b, c = f["A"], f["b"], f["c"]
            print(f"\nFrontera {ci} / {cj}  ({f['tipo']}):")
            if self.n_dim_ == 2:
                print(f"  {A[0,0]:+.{decimales}f} z1^2 "
                      f"{2*A[0,1]:+.{decimales}f} z1 z2 "
                      f"{A[1,1]:+.{decimales}f} z2^2 "
                      f"{b[0]:+.{decimales}f} z1 "
                      f"{b[1]:+.{decimales}f} z2 "
                      f"{c:+.{decimales}f} = 0")
            elif self.n_dim_ == 3:
                print(f"  {A[0,0]:+.{decimales}f} z1^2 "
                      f"{A[1,1]:+.{decimales}f} z2^2 "
                      f"{A[2,2]:+.{decimales}f} z3^2")
                print(f"  {2*A[0,1]:+.{decimales}f} z1 z2 "
                      f"{2*A[0,2]:+.{decimales}f} z1 z3 "
                      f"{2*A[1,2]:+.{decimales}f} z2 z3")
                print(f"  {b[0]:+.{decimales}f} z1 "
                      f"{b[1]:+.{decimales}f} z2 "
                      f"{b[2]:+.{decimales}f} z3 "
                      f"{c:+.{decimales}f} = 0")
            else:
                print(f"  A =\n{np.round(A, decimales)}")
                print(f"  b = {np.round(b, decimales)}   c = {round(c, decimales)}")


# ==========================================================================
# BUSQUEDA DE LA MEJOR PROYECCION 2D / 3D
# ==========================================================================

def criterio_fisher(Z, y) -> float:
    """Criterio de separabilidad  J = traza( S_W^(-1) * S_B ).

        S_W = sum_c sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T     (intra-clase)
        S_B = sum_c N_c (mu_c - mu)(mu_c - mu)^T                (entre clases)

    Cuanto mayor es J, mas separadas estan las clases en relacion con su
    dispersion interna.  Es adimensional y no depende de la escala.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    d = Z.shape[1]
    mu = Z.mean(axis=0)
    S_W = np.zeros((d, d))
    S_B = np.zeros((d, d))
    for c in np.unique(y):
        A = Z[y == c]
        m = A.mean(axis=0)
        D = A - m
        S_W += D.T @ D
        dm = (m - mu).reshape(-1, 1)
        S_B += len(A) * (dm @ dm.T)
    S_W += 1e-9 * np.eye(d)
    return float(np.trace(np.linalg.solve(S_W, S_B)))


def mejores_combinaciones(X, y, n_dim: int = 3, top: int = 10):
    """Busca exhaustivamente las mejores combinaciones de n_dim caracteristicas.

    Devuelve una lista [(J, (col1, col2, ...)), ...] ordenada de mayor a menor.
    Las columnas se estandarizan antes, para que el criterio no dependa de las
    unidades de cada caracteristica.
    """
    import pandas as pd
    X = pd.DataFrame(X)
    Z = ((X - X.mean()) / X.std(ddof=1)).values
    columnas = list(X.columns)

    resultados = []
    for indices in itertools.combinations(range(len(columnas)), n_dim):
        resultados.append((criterio_fisher(Z[:, indices], y),
                           tuple(columnas[i] for i in indices)))
    resultados.sort(key=lambda t: -t[0])
    return resultados[:top]


# ==========================================================================
# GRAFICAS
# ==========================================================================

_COLORES = {"abajo": "#d62728", "arriba": "#2ca02c", "circulo": "#9467bd",
            "derecha": "#1f77b4", "izquierda": "#ff7f0e"}


def graficar_dispersion_2d(Z, y, etiquetas=("z1", "z2"), titulo="",
                           modelo=None, resolucion: int = 300, ax=None):
    """Dispersion en 2D y, si se pasa `modelo`, sus regiones de decision."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6.5))

    if modelo is not None:
        m = 0.12 * (Z.max(axis=0) - Z.min(axis=0))
        x_min, x_max = Z[:, 0].min() - m[0], Z[:, 0].max() + m[0]
        y_min, y_max = Z[:, 1].min() - m[1], Z[:, 1].max() + m[1]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucion),
                             np.linspace(y_min, y_max, resolucion))
        malla = np.c_[xx.ravel(), yy.ravel()]
        pred = modelo.predict(malla)
        indice = {c: i for i, c in enumerate(modelo.clases_)}
        ax.contourf(xx, yy, np.vectorize(indice.get)(pred).reshape(xx.shape),
                    levels=np.arange(-0.5, len(modelo.clases_)), alpha=0.18,
                    colors=[_COLORES.get(c, "#888888") for c in modelo.clases_])
        ax.contour(xx, yy, np.vectorize(indice.get)(pred).reshape(xx.shape),
                   levels=np.arange(-0.5, len(modelo.clases_)),
                   colors="k", linewidths=0.7, alpha=0.6)

    for c in sorted(np.unique(y)):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], label=c, s=40, alpha=0.85,
                   edgecolor="k", linewidth=0.35, color=_COLORES.get(c))
    ax.set_xlabel(etiquetas[0])
    ax.set_ylabel(etiquetas[1])
    ax.set_title(titulo)
    ax.legend(title="Gesto", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def graficar_dispersion_3d(Z, y, etiquetas=("z1", "z2", "z3"), titulo="",
                           elev: float = 20, azim: float = -60, ax=None):
    """Dispersion en 3D con un elipsoide de covarianza por clase."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    if ax is None:
        fig = plt.figure(figsize=(9.5, 8))
        ax = fig.add_subplot(111, projection="3d")

    for c in sorted(np.unique(y)):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], label=c, s=28, alpha=0.75,
                   edgecolor="k", linewidth=0.25, color=_COLORES.get(c))
        _elipsoide(ax, Z[m], color=_COLORES.get(c, "#888888"))

    ax.set_xlabel(etiquetas[0])
    ax.set_ylabel(etiquetas[1])
    ax.set_zlabel(etiquetas[2])
    ax.set_title(titulo)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(title="Gesto", fontsize=9)
    return ax


def _elipsoide(ax, A, color, n_sigma: float = 1.5, n: int = 22):
    """Superficie de nivel  (z-mu)^T S^(-1) (z-mu) = n_sigma^2  de una clase."""
    if len(A) < 4:
        return
    mu = A.mean(axis=0)
    S = np.cov(A.T, ddof=1)
    valores, vectores = np.linalg.eigh(S)
    valores = np.clip(valores, 1e-12, None)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    esfera = np.array([np.outer(np.cos(u), np.sin(v)).ravel(),
                       np.outer(np.sin(u), np.sin(v)).ravel(),
                       np.outer(np.ones_like(u), np.cos(v)).ravel()])
    puntos = (vectores @ np.diag(n_sigma * np.sqrt(valores)) @ esfera).T + mu
    ax.plot_surface(puntos[:, 0].reshape(n, n), puntos[:, 1].reshape(n, n),
                    puntos[:, 2].reshape(n, n), color=color, alpha=0.10,
                    linewidth=0, shade=False)
