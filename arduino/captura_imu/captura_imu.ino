/*
 * Captura de datos IMU - Proyecto de clasificacion de gestos
 * Arduino Nano 33 BLE Sense (original o REV2)
 *
 * Formato de salida (una linea por muestra):
 *     timestamp_ms,ax,ay,az,gx,gy,gz
 *   - timestamp_ms : milisegundos desde el arranque de la placa
 *   - ax, ay, az   : aceleracion en g       (1 g = 9.81 m/s^2)
 *   - gx, gy, gz   : velocidad angular en grados por segundo (dps)
 *
 * Las lineas que empiezan por '#' son informativas: el script de captura
 * las descarta porque no tienen 7 campos.
 *
 * ------------------------------------------------------------------
 * CORRECCIONES RESPECTO A LA VERSION ANTERIOR
 * ------------------------------------------------------------------
 * 1) La version anterior usaba  delay(18)  y afirmaba en los comentarios
 *    que muestreaba a 50 Hz.  El periodo real medido en los datos ya
 *    grabados es de 32 ms, es decir  fs = 1000/32 = 31.25 Hz,  no 50 Hz.
 *    Motivo: delay(18) es un retardo ADICIONAL al tiempo que tarda el
 *    bucle en esperar a que la IMU tenga dato nuevo y en imprimir por el
 *    puerto serie.  El periodo total es la suma de las tres cosas, y no
 *    esta controlado.
 *
 *    Aqui se usa una planificacion de periodo fijo con micros():
 *        siguiente_us = siguiente_us + PERIODO_US
 *    de modo que el intervalo entre muestras no acumula deriva.
 *
 * 2) Se anade una medicion real de la tasa maxima del sensor al arrancar,
 *    para no suponerla.  Si la tasa configurada supera lo que el sensor
 *    entrega, el sketch avisa por el puerto serie.
 *
 * 3) La aceleracion se imprime con 4 decimales.  Con 2 decimales la
 *    resolucion efectiva era de 0.01 g, muy por encima de la del sensor,
 *    y eso cuantiza las senales pequenas.
 */

// LIBRERIA DE LA IMU
// Nano 33 BLE Sense ORIGINAL  -> Arduino_LSM9DS1.h
// Nano 33 BLE Sense REV2      -> Arduino_BMI270_BMM150.h
//#include <Arduino_LSM9DS1.h>
#include <Arduino_BMI270_BMM150.h>

// --------------------------------------------------------------------
// CONFIGURACION
// --------------------------------------------------------------------
// Frecuencia de muestreo objetivo.  Ponla igual o por debajo de la tasa
// que la placa informa al arrancar (linea "# tasa_maxima_medida_hz").
// Debe coincidir con la que declares en el README y en el informe.
const uint32_t FS_OBJETIVO_HZ = 50;
const uint32_t PERIODO_US = 1000000UL / FS_OBJETIVO_HZ;

const uint8_t DECIMALES_ACC = 4;   // resolucion de impresion del acelerometro
const uint8_t DECIMALES_GYR = 2;   // resolucion de impresion del giroscopio

uint32_t siguiente_us = 0;

// Ultimo valor valido leido (se reutiliza si el sensor aun no tiene dato
// nuevo, para que la serie temporal quede uniformemente muestreada).
float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

// --------------------------------------------------------------------

void medirTasaMaxima() {
  // Cuenta cuantas muestras nuevas entrega la IMU en 1 segundo.
  uint32_t n = 0;
  uint32_t t0 = millis();
  float a, b, c;
  while (millis() - t0 < 1000) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(a, b, c);
      IMU.readGyroscope(a, b, c);
      n++;
    }
  }
  Serial.print("# tasa_maxima_medida_hz,");
  Serial.println(n);
  if (n < FS_OBJETIVO_HZ) {
    Serial.print("# AVISO: la tasa objetivo (");
    Serial.print(FS_OBJETIVO_HZ);
    Serial.println(" Hz) supera la del sensor. Baja FS_OBJETIVO_HZ.");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("# ERROR: no se pudo inicializar la IMU");
    while (1);
  }

  Serial.print("# fs_objetivo_hz,");
  Serial.println(FS_OBJETIVO_HZ);
  Serial.print("# tasa_acelerometro_hz,");
  Serial.println(IMU.accelerationSampleRate());
  Serial.print("# tasa_giroscopio_hz,");
  Serial.println(IMU.gyroscopeSampleRate());

  medirTasaMaxima();

  Serial.println("# columnas,timestamp_ms,ax,ay,az,gx,gy,gz");
  siguiente_us = micros();
}

void loop() {
  // Refrescar los valores en cuanto el sensor tenga dato nuevo.
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }

  // Emitir exactamente una muestra cada PERIODO_US.
  // La resta con signo maneja correctamente el desbordamiento de micros().
  if ((int32_t)(micros() - siguiente_us) >= 0) {
    siguiente_us += PERIODO_US;

    Serial.print(millis());          Serial.print(',');
    Serial.print(ax, DECIMALES_ACC); Serial.print(',');
    Serial.print(ay, DECIMALES_ACC); Serial.print(',');
    Serial.print(az, DECIMALES_ACC); Serial.print(',');
    Serial.print(gx, DECIMALES_GYR); Serial.print(',');
    Serial.print(gy, DECIMALES_GYR); Serial.print(',');
    Serial.println(gz, DECIMALES_GYR);
  }
}
