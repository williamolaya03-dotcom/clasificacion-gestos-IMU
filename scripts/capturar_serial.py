#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adquisicion de gestos desde el Arduino Nano 33 BLE Sense por puerto serie.

Guarda cada repeticion en
    data/raw/<gesto>/participante_XX_sesion_YY_rep_ZZZ.csv
con las columnas  timestamp_ms,ax,ay,az,gx,gy,gz.

--------------------------------------------------------------------------
CORRECCIONES RESPECTO A LA VERSION ANTERIOR
--------------------------------------------------------------------------
1) BUFFER SERIE OBSOLETO.  La placa transmite de forma continua, tambien
   durante la cuenta atras y la pausa entre repeticiones.  `reset_input_buffer()`
   vacia el buffer de pyserial, pero puede quedar informacion en el buffer del
   sistema operativo y del conversor USB-serie.  El resultado eran archivos
   cuya primera o primeras muestras venian de una captura anterior:

     data/raw/derecha/..._rep_001.csv  ->  salto de 130 623 ms tras la 2a muestra
     data/raw/izquierda/..._rep_001.csv ->  salto de 2 695 ms tras la 1a muestra

   Ahora, tras vaciar el buffer, se DRENA el puerto durante DRENAJE_S
   descartando todo lo que llegue, y ademas se valida la continuidad
   temporal antes de guardar.

2) VALIDACION DE CONTINUIDAD.  Se calcula dt[i] = t[i] - t[i-1] y se corta
   la serie en cualquier salto  dt > FACTOR_HUECO * mediana(dt),  quedandose
   con el tramo continuo mas largo.  Si el tramo resultante es demasiado
   corto, la repeticion se descarta y se puede repetir.

3) INFORME DE LA TASA REAL.  Se calcula  fs = 1000 / mediana(dt)  y se
   muestra por pantalla, en lugar de suponer 50 Hz.

4) La lectura ya no depende de `ser.in_waiting`: se usa el tiempo de espera
   del propio puerto, que no consume CPU en un bucle vacio.
"""

import csv
import os
import statistics
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("Falta pyserial.  Instala con:  pip install pyserial")

# ==========================================================================
# CONFIGURACION DE USUARIO
# ==========================================================================
PUERTO_SERIAL = 'COM16'      # Windows: 'COM16'  |  Linux: '/dev/ttyACM0'
                             # macOS:   '/dev/cu.usbmodemXXXX'
BAUDIOS = 115200
DURACION_CAPTURA = 3.0       # segundos de grabacion por repeticion
REPETICIONES = 40            # repeticiones por ejecucion
CARPETA_BASE = os.path.join('data', 'raw')

DRENAJE_S = 0.4              # tiempo descartando datos antes de grabar
FACTOR_HUECO = 5.0           # dt > FACTOR_HUECO * mediana(dt) = discontinuidad
MIN_MUESTRAS = 32            # por debajo de esto la repeticion se descarta
GESTOS_VALIDOS = ('izquierda', 'derecha', 'arriba', 'abajo', 'circulo')
COLUMNAS = ['timestamp_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']


# ==========================================================================
# FUNCIONES AUXILIARES
# ==========================================================================

def listar_puertos():
    puertos = list(list_ports.comports())
    if not puertos:
        return "   (no se detecto ningun puerto serie)"
    return "\n".join(f"   {p.device}  -  {p.description}" for p in puertos)


def drenar(ser, segundos: float) -> int:
    """Descarta todo lo que llegue durante `segundos`.  Devuelve lineas tiradas."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    tiradas = 0
    fin = time.time() + segundos
    while time.time() < fin:
        if ser.in_waiting:
            ser.read(ser.in_waiting)
            tiradas += 1
    ser.reset_input_buffer()
    return tiradas


def leer_durante(ser, segundos: float):
    """Lee lineas validas (7 campos numericos) durante `segundos`."""
    muestras = []
    fin = time.time() + segundos
    ser.readline()                       # descartar la linea parcial en curso
    while time.time() < fin:
        try:
            linea = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            continue
        if not linea or linea.startswith('#'):
            continue                     # linea informativa del sketch
        campos = linea.split(',')
        if len(campos) != 7:
            continue
        try:
            muestras.append([float(v) for v in campos])
        except ValueError:
            continue                     # linea truncada o con ruido
    return muestras


def validar_continuidad(muestras):
    """Corta discontinuidades temporales y devuelve (tramo, informe)."""
    if len(muestras) < 2:
        return muestras, {'fs_hz': None, 'huecos': 0, 'descartadas': 0}

    t = [m[0] for m in muestras]
    dt = [t[i] - t[i - 1] for i in range(1, len(t))]
    dt_mediana = statistics.median(dt)
    if dt_mediana <= 0:
        return muestras, {'fs_hz': None, 'huecos': 0, 'descartadas': 0}

    cortes = [i for i, d in enumerate(dt) if d > FACTOR_HUECO * dt_mediana]
    if cortes:
        limites = [0] + [i + 1 for i in cortes] + [len(muestras)]
        tramos = [(limites[k], limites[k + 1]) for k in range(len(limites) - 1)]
        ini, fin = max(tramos, key=lambda p: p[1] - p[0])
        tramo = muestras[ini:fin]
    else:
        tramo = muestras

    if len(tramo) >= 2:
        t2 = [m[0] for m in tramo]
        dt2 = [t2[i] - t2[i - 1] for i in range(1, len(t2))]
        fs = 1000.0 / statistics.median(dt2)
    else:
        fs = None

    return tramo, {'fs_hz': fs,
                   'huecos': len(cortes),
                   'descartadas': len(muestras) - len(tramo)}


# ==========================================================================
# PROGRAMA PRINCIPAL
# ==========================================================================

def main() -> int:
    print("=== SCRIPT DE ADQUISICION DE GESTOS IMU ===")
    print("\nPuertos serie detectados:")
    print(listar_puertos())

    puerto = input(f"\nPuerto serie [{PUERTO_SERIAL}]: ").strip() or PUERTO_SERIAL
    participante = input("ID del participante (ej. 01): ").strip() or "01"
    sesion = input("ID de la sesion (ej. 01): ").strip() or "01"
    gesto = input(f"Gesto {GESTOS_VALIDOS}: ").strip().lower()
    if gesto not in GESTOS_VALIDOS:
        print(f"AVISO: '{gesto}' no esta en la lista de gestos del proyecto. "
              f"Se usara igualmente.")
    if not gesto:
        print("ERROR: el nombre del gesto no puede estar vacio.")
        return 1

    destino = os.path.join(CARPETA_BASE, gesto)
    os.makedirs(destino, exist_ok=True)

    try:
        ser = serial.Serial(puerto, BAUDIOS, timeout=1.0)
    except Exception as exc:
        print(f"\nERROR al abrir {puerto}: {exc}")
        print("Comprueba que el Arduino este conectado y que el Monitor Serie "
              "del IDE este CERRADO (bloquea el puerto).")
        return 1

    print(f"\nConectado a {puerto} a {BAUDIOS} baudios.")
    time.sleep(2.0)                       # la placa se reinicia al abrir el puerto
    drenar(ser, 1.0)

    input("\nPresiona ENTER para comenzar las capturas...")

    guardadas, descartadas, tasas = 0, 0, []
    try:
        for rep in range(1, REPETICIONES + 1):
            nombre = (f"participante_{participante}_sesion_{sesion}"
                      f"_rep_{rep:03d}.csv")
            ruta = os.path.join(destino, nombre)
            if os.path.exists(ruta):
                print(f"[{rep:03d}] ya existe, se omite.")
                continue

            print(f"\n[Repeticion {rep}/{REPETICIONES}] preparate...")
            for i in (3, 2, 1):
                print(f"   {i}...")
                time.sleep(1)

            # Purga del buffer JUSTO antes de grabar: evita mezclar muestras
            # de la repeticion anterior con las de esta.
            drenar(ser, DRENAJE_S)

            print("   CAPTURANDO - haz el gesto")
            muestras = leer_durante(ser, DURACION_CAPTURA)
            print("   fin de captura.")

            tramo, informe = validar_continuidad(muestras)

            if len(tramo) < MIN_MUESTRAS:
                descartadas += 1
                print(f"   DESCARTADA: solo {len(tramo)} muestras validas "
                      f"(minimo {MIN_MUESTRAS}). Repite este gesto.")
                continue

            if informe['descartadas']:
                print(f"   AVISO: se recortaron {informe['descartadas']} "
                      f"muestra(s) por discontinuidad temporal.")

            with open(ruta, 'w', newline='', encoding='utf-8') as fh:
                escritor = csv.writer(fh)
                escritor.writerow(COLUMNAS)
                for m in tramo:
                    escritor.writerow([int(m[0])] + [f"{v:.4f}" for v in m[1:]])

            guardadas += 1
            if informe['fs_hz']:
                tasas.append(informe['fs_hz'])
            fs_txt = (f"{informe['fs_hz']:.1f} Hz" if informe['fs_hz'] else "n/d")
            print(f"   guardado: {ruta}  ({len(tramo)} muestras, fs = {fs_txt})")

            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        ser.close()

    print("\n=== PROCESO COMPLETADO ===")
    print(f"Repeticiones guardadas : {guardadas}")
    print(f"Repeticiones descartadas: {descartadas}")
    if tasas:
        print(f"Frecuencia de muestreo real: mediana "
              f"{statistics.median(tasas):.2f} Hz  "
              f"[{min(tasas):.2f}, {max(tasas):.2f}]")
        print("Anota esta cifra en el README y en el informe: es la fs real "
              "del dataset.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
