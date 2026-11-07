# Contexto completo de la red neuronal de muografía (versión final V16)

Documento de traspaso para un asistente que va a trabajar sobre este proyecto. Describe qué hace la red, cómo se generan los datos de entrenamiento, cómo se entrena, cómo se aplica a los datos reales y cómo se presentan los resultados. Todas las rutas son relativas a la carpeta `NN2/`, y todos los valores numéricos están copiados del código. **Si algo de aquí no coincide con el código, manda el código.**

Fuera de alcance: el ajuste rígido de barras ("Nivel A": `ajuste_barras.py`, `validar_ajuste_barras.py` y la parte `rigid_fit` de `postproceso.py` y `figuras_tesis.py`). Existe, pero **no se usa** en el trabajo que sigue. No lo modifiques ni lo tomes como parte del método.

---

## 0. Reglas de trabajo con el usuario

- El usuario es Daniel Corredor, estudiante de posgrado de la Universidad de los Andes (física experimental de muografía). Escribe y prefiere respuestas en **español**.
- **Figuras y resultados para la tesis, en inglés.** Unidades de las imágenes: *Average Scattering Density* [rad²/cm] = varianza / 0.5 cm, en escala logarítmica.
- Nombres de las muestras en la tesis:
  - $\mathcal{M}_1$ = malla de varillas (carpetas `Mallado_*`). Hay dos rellenos: **arena** (`Mallado_arena`) y **concreto** (`Mallado_mortero`). El código dice "mortero", pero en la tesis se llama **concreto**; nunca escribas "mortar".
  - $\mathcal{M}_2$ = muestra de grosores (`Grosores_mortero`).
- En las figuras de tesis: títulos cortos (solo qué se grafica, sin nombre de muestra ni detalles), **sin guías de la geometría** dibujadas encima (sesgan al lector) y en la orientación de la tesis (ver §7.3).
- **No se pueden tomar más mediciones reales.** La toma más corta posible es de 16 h, y las muestras están medidas.
- No hay tiempo para más entrenamientos: **V16 es la versión final.** Cualquier propuesta debe funcionar con los pesos existentes.
- Entorno:
  - **PC del usuario:** Windows, solo CPU, PyTorch 2.1 CPU. Inferencia, figuras y análisis se hacen aquí.
  - **Servidor `labhep`:** 2× RTX 4000 Ada, Python 3.8, torch 2.4.1, repositorio en `/home/d.corredorl/proyecto/codigo/NN_github` (antes `.../NN2`). Solo se usa para entrenar.
  - El repositorio GitHub `DaniCorredor/NN` es público. Flujo: PC → `git push` → servidor `git pull`; los resultados vuelven con `git push` desde el servidor.
- Lo pesado no se sube a git (`.gitignore`): `datasets/`, `*.pth`, voxelizaciones completas. Sí se suben `modelos/*_fp16.pt` (pesos de la red en float16, ~74 MB).

---

## 1. Problema físico

- **Instrumento:** hodoscopio de muones cósmicos de 57×57 cm². Cuatro planos de detección en z = 0, 40, 128 y 168 cm; la muestra va en medio.
- **Medida por muón:** se reconstruyen las trazas de entrada y de salida, y se calculan el ángulo de dispersión θ y el punto PoCA (*Point of Closest Approach*) entre las dos trazas.
- **Voxelización:** la región de interés (ROI) es de 56×56×60 cm (z entre 54 y 114 cm), con vóxeles de 0.5 cm, es decir una grilla de **112×112×120**. Cada muón se asigna al vóxel de su PoCA, y cada vóxel guarda la lista de θ de sus muones.
- **Cortes usados** (archivos `voxel_data_n_*_nhits_1_disMax_4cm_poca_angle_15.0.npz`):
  - DOCA (distancia entre trazas) ≤ 4 cm;
  - ángulo de incidencia < 15°;
  - ángulo de dispersión entre la resolución y el máximo geométrico (82°);
  - PoCA dentro de la ROI.
- **Estadística:** muy baja. ~6.3 muones por píxel (columna de 0.5×0.5 cm) por día en el centro del detector, y mucho menos en los bordes. Las tomas reales son de ~20 días.
- **Física:** la dispersión múltiple crece con el material atravesado, $T = \int dl/X_0$ (longitudes de radiación). El hierro ($X_0$ = 1.757 cm) dispersa mucho más que el concreto ($X_0$ ≈ 13 cm a 2 g/cm³).
- **Objetivo de la red:** a partir del mapa ruidoso de unos pocos días, recuperar el mapa de material y localizar el hierro (varillas de refuerzo) dentro del concreto.

### Muestras reales (`datos_reales/`)

| Muestra | Descripción | Tomas acumuladas (días) |
|---|---|---|
| $\mathcal{M}_1$ arena (`Mallado_arena_20d`) | Malla de varillas #6 (Ø1.9 cm) con espaciado irregular (4/6/8/10 cm) y profundidad variable, en arena; bloque de ~47×47×21 cm | 6.9, 13.9, 19.6, 21.6 |
| $\mathcal{M}_1$ concreto (`Mallado_mortero_20d`) | La misma malla en concreto (~2.0 g/cm³, 103 kg) | 12.0, 18.9 |
| $\mathcal{M}_2$ (`Grosores_mortero_20d`) | Bloque de concreto de 49×30×25 cm (~1.8 g/cm³, 69 kg) con 4 barras de hierro de 10 cm de largo y Ø 6, 4, 2 y 1 cm, cada una seguida de un tubo vacío de 5 cm del mismo diámetro | 6.5, 13.6, 20.6 |
| Plomo (`plomo_bloques_84h`) | Bloques de plomo libres de 20, 10 y 5 cm de espesor (solo para calibración) | 3.5 |
| Hodoscopio vacío (`vacio_exp_0.7d`, `toma_libre_voxelizacion_exp.npz`) | Toma experimental de 16 h sin muestra (solo para calibrar los bordes) | 0.7 |

- **Tomas acumuladas:** la toma de 13.6 d contiene a la de 6.5 d, y en cada vóxel la lista de la toma corta es exactamente el comienzo de la lista de la toma larga.
- **Intervalos independientes** (`*_int_a-bd_*`): se construyen tomando, en cada vóxel, solo los muones posteriores a la toma anterior. Ejemplos: 0–6.5, 6.5–13.6 y 13.6–20.6 d. No comparten ningún muón.
- **Posición de las muestras:** se colocaron en el hodoscopio sin coordenadas exactas. Las posiciones de las barras que hay en el código (`GEOMETRIAS` en `inference_unet_2_real.py`) son **estimaciones** ajustadas a los perfiles medidos. Se usan solo para métricas y **no se dibujan** en las figuras de tesis.

---

## 2. Qué entra y qué sale de la red

### 2.1 Entrada (2 canales, 112×112)
1. **Media recortada de θ² por columna (x, y):**
   - se juntan los θ² de todos los muones de la columna, en cualquier z (los vóxeles vacíos no cuentan);
   - se ordenan y se descarta el 10% más alto (`FRACCION_RECORTE = 0.10`) con recorte fraccionario, como en `media_recortada()` de `data_generator.py`;
   - se promedia el resto.
   - Unidades: rad². Archivos: `datos_reales/<muestra>_<dias>d_varrec_xy.npy`, generados por `preparar_recortada.py`.
   - **Normalización:** `normalizar_entrada(v) = clip(v / MAX_VAR, 0, 1)`, sin restar pedestal.
   - **Por qué recortar:** quita las colas de dispersiones extremas y multiplica la señal/ruido de las barras por 1.4–2.4 frente a la media simple.
2. **Número de muones por columna, N** (`*_N_xy.npy`):
   - normalizado con `normalizar_muones(N) = clip(log1p(N) / log1p(256), 0, 1)`;
   - le dice a la red cuánto confiar en cada píxel.

### 2.2 Salidas (3 cabezales, 112×112)
1. **Varianza "mejorada":** la varianza esperada con N → ∞ y sin desenfoque. Normalizada como `clip((y − PEDESTAL) / MAX_VAR, 0, 1)`; en la inferencia se desnormaliza sumando el pedestal de nuevo.
2. **Densidad:** $T = \int dl/X_0$ proyectado, sin desenfoque. Normalizada por `MAX_DENSITY`.
3. **Probabilidad de hierro:** son logits; la probabilidad es sigmoide(logits). Un píxel se considera hierro si p > **0.35**, el umbral calibrado en el conjunto de prueba (`resultados/V16/test_evaluation/iron_threshold.json`).

**Normalización** (`limites_normalizacion()`):
- $T_{max}$ = 100 cm de hierro = 56.8;
- `MAX_VAR = detector_response(T_max) × 1.1`;
- `MAX_DENSITY = T_max × 1.1`.

---

## 3. Generación de los datos de entrenamiento (`data_generator.py`)

**Es simulación simplificada, no GEANT4.** No se transportan muones uno a uno: se calcula directamente el mapa que mediría el detector, con piezas calibradas con datos reales. Cada escena tarda menos de 1 s.

### 3.1 Escena 3D
- **Grilla:** `GRID = (112, 112, 120)`, `VOXEL_CM = 0.5`. Cada vóxel guarda el 1/X₀ [1/cm] de su material:
  - aire = 0;
  - hierro = 0.568;
  - concreto ∈ `INV_X0_CONCRETO = (0.068, 0.080)`, sorteado por escena;
  - arena ∈ `INV_X0_ARENA = (0.055, 0.068)`, sorteado por escena.
- **Bloque anfitrión** (`sample_block`):
  - lados en x e y de 25–60 cm, espesor de 18–28 cm;
  - centro desplazado ±8 cm en x e y, y ±10 cm en z;
  - 80% concreto y 20% arena.
- **Tipo de escena** (`build_scene`), por `i % 100`:

  | Tipo | Porcentaje |
  |---|---|
  | Malla | 32% |
  | Grosores | 22% |
  | Barras sueltas | 18% |
  | Bloque sin hierro | 8% |
  | Barras + huecos | 6% |
  | Solo huecos | 6% |
  | Aire | 4% |
  | Barras en aire | 4% |

  El 40% de los bloques con barras lleva además tubos vacíos de PVC (Ø1.5–3 cm, cerca del fondo y de un borde).
- **Malla** (`add_rebar_mesh`):
  - de 2 a 6 barras por dirección; en el 25% de los casos, una sola dirección;
  - diámetro de 1–3 cm por dirección;
  - espaciado regular (40%) o irregular (60%);
  - profundidad que puede variar barra a barra (hasta 8 cm);
  - extremos que sobresalen 0–5 cm del bloque o terminan 1–4 cm antes;
  - un 10% de barras faltantes.
- **Grosores** (`add_thickness_row`):
  - fila de 2 a 5 barras paralelas, de 6–15 cm de largo y Ø 1–6 cm (ordenadas por grosor en el 50% de los casos), separadas 4–12 cm;
  - en el 60% de los casos cada barra continúa con un tubo vacío de 3–7 cm.
- **Barras sueltas** (`add_isolated_bars`):
  - de 1 a 6 barras, de 5–40 cm de largo y Ø 1–6 cm;
  - la mitad alineadas con x o y, la otra mitad con azimut libre;
  - el 30% inclinadas hasta ±15°.
- **Huecos** (`add_void_clusters`): una esfera principal de Ø 1–7 cm más 0–4 esferas menores solapadas. Solo vacían la matriz, nunca el hierro.

### 3.2 Proyección (`simulate_transmission`)
- **64 direcciones** (`NUM_ANGULOS = 64`):
  - ángulo cenital sorteado del histograma **medido** (`calibracion/zenith_hist.npz`, que ya incluye flujo y aceptancia; media de 8.2°), con corte en 15°;
  - azimut uniforme.
- **Cada dirección:** proyección inclinada en la que cada capa z se desplaza tan θ·(z − z_pivote), con el pivote en el centro del bloque, y se multiplica por la longitud del camino 0.5 cm / cos θ.
- **Resultado:** el promedio de las 64 proyecciones da $T(x,y)$, más un segundo mapa $T_{hierro}$ con solo el aporte del hierro.

### 3.3 Lo que se guarda por escena (`.npz`)
- `x_limpio`: entrada **sin ruido**:
  1. $T_{det} = T − (1 − u)\,T_{hierro}$, con u ∈ `FACTOR_HIERRO = (0.55, 0.95)`. El contraste real de las barras es menor que el ideal (efecto PoCA medido).
  2. Desenfoque gaussiano (PSF) con σ ∈ `PSF_SIGMA_RANGO_CM = (0.6, 1.0)` cm.
  3. Se aplica `measured_variance(..., clean=True)` = `detector_response(T) + PEDESTAL`.
- `y`: varianza objetivo = respuesta de $T_{det}$ **sin desenfoque** (+ pedestal).
- `density`: el $T$ verdadero (con el hierro completo, sin desenfoque).
- `mascara`: columnas con hierro, dilatadas 2 px (para ponderar la pérdida). `hierro`: columnas con hierro exactas (objetivo de la segmentación).
- `T_hierro` (float16) y `scene` (tipo de escena).
- Solo en el conjunto de prueba: una medida ruidosa fija `x`, `muones` y `dias`.

### 3.4 Respuesta del detector (`detector_response`, calibrada en `calibrar_respuesta.py`)
$$V = V_0 + A\,T^{\gamma}, \qquad V_0 = 1.5\cdot10^{-3}\ \text{rad}^2,\quad A = 2.1666\cdot10^{-3}\ \text{rad}^2,\quad \gamma = 0.5284$$
- Es la escala de la **media** de θ² (no la recortada). V₀ es el valor en aire, es decir, la resolución angular del hodoscopio.
- Se ajustó con 5 regiones de material conocido **sin barras**:
  - plomo de 20, 10 y 5 cm (T = 35.6, 17.8, 8.9);
  - concreto de $\mathcal{M}_2$ (25 cm a 1.82 g/cm³, T = 1.71);
  - concreto de $\mathcal{M}_1$ (21 cm a 2.0 g/cm³, T = 1.58); X₀ másico del concreto = 26.6 g/cm².
- Sustituye a Highland (V ∝ T), que no describe el detector: crece más lento por los cortes angulares, la hipótesis de un único punto de dispersión de PoCA y la absorción.
- Validación: en el concreto de calibración de $\mathcal{M}_2$ coincide con lo medido al 0.6%, y el ruido simulado coincide con el medido (`figura_entrenamiento.py`).

### 3.5 Ruido (`agregar_ruido`, aplicado al cargar cada muestra)
1. **Días:** 80% entre 15 y 25 d, 20% entre 3 y 15 d (`sample_observation_days`).
2. **Muones por píxel:** $N = \text{Poisson}(\text{días} × a(x,y) × r(x,y))$.
   - $a$ = `calibracion/aceptancia_xy.npy`, en muones/día/píxel (~6.3 en el centro). Sale de una toma libre simulada del hodoscopio, suavizada y escalada con la tasa medida en el centro.
   - $r = \max(0.3,\ 1 + k\cdot c_{loc})$ con k ∈ `REDISTRIBUCION_N = (0.15, 0.45)`, donde $c_{loc}$ es el contraste local de la varianza (suavizado de 2 cm). PoCA concentra muones en el material denso.
3. **Bandas fijas del detector** (`exceso_bordes`, calibradas en `calibrar_bordes.py` con la toma vacía y el aire de las tomas de plomo y $\mathcal{M}_2$). Exceso aditivo, igual en aire y en concreto:
   - banda gaussiana en y ≈ 9.6 cm (σ ≈ 2.3 cm);
   - escalón sigmoide desde y ≈ 49.5 cm;
   - subida exponencial hacia x = 0;
   - en cada muestra se sortean la amplitud (×0.3–2.0), el ancho (×0.7–1.4) y la posición (±1.5 cm); x = 56 recibe como mínimo el 15% de la subida de x = 0.
4. **Estimador con ruido real:** cada píxel se multiplica por un factor tomado de `calibracion/banco_ruido.npz`.
   - El banco tiene, para cada N de 1 a 400 y cada clase (aire, concreto o plomo), 1024 medias recortadas de θ² de **muones reales** normalizados a media 1 (`theta2_por_muon.npz`).
   - La clase se sortea según el T del píxel, al invertir la respuesta del detector.
   - Para N > 400, la dispersión se reduce como 1/√N.
   - Esto reproduce las colas reales y el cambio de escala del recorte: E[recortada]/E[media] ≈ 0.49 en concreto y ≈ 0.6 en aire.

**Detalle importante:** `x_limpio` e `y` están en la escala de la media. El banco convierte a la escala de la media recortada, que es la de los datos reales.

---

## 4. Red neuronal (`Unet.py`)

`RRDB_UNet_Deterministic(in_channels=2, base_dim=32, num_rrdb=3, con_hierro=True)`.

- **Codificador:** `DoubleConv` (Conv3×3 → GroupNorm(8) → LeakyReLU(0.2), dos veces, más atención por canal) y 3 reducciones (AvgPool2 + DoubleConv) con 32 → 64 → 128 → 256 canales.
- **Cuello:** atención espacial (kernel 7) y 3 bloques **RRDB** (Residual-in-Residual Dense Block, crecimiento de 128 canales, escala residual de 0.2).
- **Decodificador:** 3 subidas (Upsample bilineal + Conv3×3), concatenación con las conexiones de salto y DoubleConv.
- **Tres cabezales**, cada uno Conv3×3 (16 canales) → LeakyReLU → Conv1×1 (1 canal): `head_variance`, `head_density` y `head_iron` (logits).
- **Tamaños:** 112 es divisible por 8, así que no hace falta relleno. La inferencia rellena a múltiplos de 16 por seguridad (`pad_to_multiple`).

---

## 5. Entrenamiento (`train_Unet_2.py`)

- **Datos:** `datasets/dataset_muografia_V11`, 10 000 escenas (solo están en el servidor). División fija con semilla 42: **70% entrenamiento / 30% validación**.
- **En cada carga:**
  - aumentos de datos sobre la escena limpia **antes** del ruido (volteos y rotaciones de 90°), para que la aceptancia quede fija en coordenadas del detector;
  - después, ruido nuevo, es decir, una medición distinta en cada época. En validación, el ruido tiene semilla fija por muestra.
- **Pérdidas:**
  - Varianza y densidad: `HybridGradientLoss`, que suma:
    - L1 ponderado, más 2 veces la L1 de los gradientes en x e y (castiga los bordes borrosos);
    - peso por píxel = 1 + 4·`mascara` (`PESO_BARRAS = 4`), normalizado por su media.
  - Hierro: `IronSegmentationLoss` = BCE + (1 − Dice) + 0.5·(1 − clDice). clDice usa un esqueleto diferenciable (Shit et al., 2021) y premia la continuidad de las barras.
  - **Total** = L_var + L_dens + 0.1·L_hierro (`PESO_HIERRO = 0.1`).
- **Optimización:**
  - AdamW, lr = 5·10⁻⁴, weight_decay = 10⁻⁴;
  - tasa de aprendizaje con CosineAnnealing hasta 10⁻⁶ en 200 épocas;
  - lote de 16, recorte del gradiente a 0.5;
  - early stopping con paciencia de 25 (no se activó);
  - GPU `cuda:1`.
- **Selección del modelo:** se guarda el de mejor pérdida de validación (`checkpoints/V16/best_model.pth`, solo en el servidor). En el PC se usa **`modelos/V16_fp16.pt`**, los mismos pesos en float16, que `cargar_modelo()` acepta.
- **Resultado V16:**
  - mejor época 199; diferencia entre entrenamiento y validación de 2.4%, estable, **sin sobreajuste**;
  - curvas en `checkpoints/V16/historial.csv` y `training_loss_curve.png`.

---

## 6. Evaluación en simulación (`evaluar_test.py`)

- **Conjunto de prueba:** `datasets/dataset_muografia_Test_V13`, 500 escenas con ruido fijo y semilla 12345, nunca vistas en el entrenamiento.
- **Métricas:**
  - MAE de densidad, dentro y fuera de las barras;
  - contraste de barra recuperado;
  - Dice, precisión y recall del hierro;
  - fracción de falso hierro.
  - Todo por tipo de escena y por rango de días.
- **Umbral de hierro:** se barre de 0.20 a 0.95 y se elige el de mejor Dice medio. **Resultado: 0.35, con Dice 0.742.**
- **V16 total:**
  - Dice 0.74, precisión 0.76, recall 0.76;
  - contraste recuperado 0.81;
  - falso hierro 2.3%.
  - Por escena, el peor caso es la malla (Dice 0.68).
- Resultados en `resultados/V16/test_evaluation/`.

---

## 7. Aplicación a datos reales

### 7.1 Preprocesamiento (`preparar_recortada.py`, en el PC; ya ejecutado)
- Lee `../Datos/<muestra>/voxel_data_n_<archivos>_nhits_1_disMax_4cm_poca_angle_15.0.npz` (claves `voxel_theta`, `grid_counts`, `grid_score`). Los días se calculan como archivos / 144.
- **Salidas:**
  - `datos_reales/<muestra>_<dias>d_{varrec,var,N}_xy.npy` (media recortada, media simple, N) para tomas acumuladas e intervalos;
  - `calibracion/theta2_por_muon.npz` (θ² reales para el banco de ruido);
  - `calibracion/aceptancia_xy.npy`.

### 7.2 Inferencia (`inference_unet_2_real.py`)
- `cargar_modelo(ruta)` detecta solo los pesos fp16 y el cabezal de hierro.
- `inferir_mapa(model, device, varrec, N)` devuelve `(varianza, densidad, prob_hierro)` en unidades físicas.
- `run_inference()` procesa todos los `*_varrec_xy.npy` y genera:
  - las figuras `*_network.png`;
  - las comparaciones entre tomas acumuladas;
  - las de intervalos independientes;
  - la correlación de estructura fina entre intervalos.
- `run_filtered_inference()` es solo referencia: corre con datos filtrados, deconvolucionados y con un solo corte. **No mejoran el resultado.**

### 7.3 Convenciones de figura (variables globales de `inference_unet_2_real.py`)
- **Índices:** los mapas `.npy` se indexan `[x, y]` en coordenadas del detector.
- **`TRANSPONER = True`:** dibuja como en la tesis, con el y del detector en horizontal y el x en vertical.
  - $\mathcal{M}_2$: barras verticales, de la más gruesa a la izquierda a la más delgada a la derecha.
  - $\mathcal{M}_1$: la zona con más barras queda abajo a la derecha.
  - Solo cambia el dibujo, no los datos.
- **Ventana:** `LIMITES_CM = (2, 54)`, 52 cm centrados en 28 cm, el centro del detector y de la aceptancia.
- **Escala fija del input:** `ESCALA_ENTRADA_ASD = (0.8e-3, 1.28e-2)` rad²/cm, logarítmica.
- **Mapas de color:**

  | Mapa | Colores |
  |---|---|
  | ASD | `inferno` (log) |
  | ∫dl/X₀ | `YlGnBu` |
  | Probabilidad de hierro | `gray`, con contorno rojo en el umbral |
  | Incertidumbre | `cividis` |

- **`MOSTRAR_GUIA = False`:** la guía de la geometría no se dibuja.
- **`NOMBRES_TESIS`:** usa $\mathcal{M}_1$ (sand), $\mathcal{M}_1$ (concrete) y $\mathcal{M}_2$.
- **Panel "Measured ASD" en las figuras de tesis:** muestra el **estimador clásico** de la tesis, no la entrada de la red.
  - El estimador clásico (`figuras_tesis.mapa_clasico`) es la media de θ² por vóxel, promediada en z con los vóxeles vacíos como 0, con los mismos cortes de 15°.
  - La red siempre recibe la media recortada.

### 7.4 Zona confiable (`confiabilidad.py`)
- **Criterio:** se confía solo donde la **densidad relativa de eventos RED ≥ 0.35** (`RED_MINIMA`).
  - La RED es `calibracion/densidad_relativa_eventos_xy.npy`: el conteo de trayectorias por vóxel de una simulación Monte Carlo del hodoscopio (`../scatter/mapa_hit_count_3d.npy`), proyectado en X-Y y normalizado a 1 en el máximo (centro en (28, 28) cm).
  - En el centro va de ~8 a ~48 cm en cada eje.
- **Justificación:**
  - El ruido por píxel crece como $1/\sqrt{\text{RED}}$, porque N ∝ RED. En RED = 0.35 es ~1.7 veces el del centro.
  - Se comprueba con el ruido medido como std(A − B)/√2 entre intervalos independientes, relativo a la zona con RED ≥ 0.8 (figura 04).
  - Por debajo de ~0.35–0.4, el falso hierro de V16 en $\mathcal{M}_1$ sube a 23–41%.
- **En las figuras:**
  - fuera de la zona, un velo gris difuso (`ATENUACION_ANCHO = 0.08`, `ATENUACION_MAXIMA = 0.7`) y una línea blanca en el límite;
  - con `RECORTAR_A_CONFIABLE = True`, los mapas valen 0 fuera de la zona y la ventana se recorta a ella (carpeta `thesis_figures_2`).

### 7.5 Incertidumbre por remuestreo (`bootstrap_muones.py`)
- 50 repeticiones: en cada columna se remuestrean con reemplazo sus N muones, se recalcula la media recortada y se corre la red.
- **Salidas:**
  - frecuencia de detección = fracción de repeticiones con p > 0.35;
  - desviación estándar de ∫dl/X₀ (incertidumbre estadística).
- **Resultado:** entre el 67% y el 72% del hierro marcado es estable (≥ 90% de las repeticiones). Las manchas falsas de los bordes también son estables: son errores **sistemáticos**, que se manejan con la zona confiable, no con la estadística.
- Salidas guardadas en `resultados/V16/postprocessing/bootstrap/`.

### 7.6 Figuras de tesis (`figuras_tesis.py`)
Regenera, desde los resultados guardados y en menos de unos minutos, `resultados/V16/thesis_figures/` (con velo gris) y `resultados/V16/thesis_figures_2/` (recortada a la zona confiable):

| Figura | Contenido |
|---|---|
| 01 | Curvas de entrenamiento |
| 02–03 | Métricas y ejemplos del conjunto de prueba |
| 04 | RED con el umbral, y ruido frente a RED en las tres muestras |
| 05 | Intervalos independientes |
| 06 | Cuadrícula 2×3: Measured ASD, Enhanced ASD, ∫dl/X₀ / Iron probability, Detection frequency, Statistical uncertainty |
| 07 | Ajuste rígido (fuera de alcance; ignorar) |

`figura_entrenamiento.py` genera `resultados/V16/reunion/training_data_and_detector_response.png`, que explica la cadena de simulación, la calibración de V y la comparación del ruido.

---

## 8. Resultados reales de V16

Contra la geometría estimada, dentro del bloque y con umbral de 0.35:

| Muestra | Dice | Precisión | Recall |
|---|---|---|---|
| $\mathcal{M}_1$ arena | 0.79 | 0.70 | 0.92 |
| $\mathcal{M}_1$ concreto | 0.73 | 0.61 | 0.90 |
| $\mathcal{M}_2$ | 0.51 | 0.34 | 1.00 |

- En $\mathcal{M}_2$ la precisión es baja porque el hierro predicho sale ~1.5–2 cm más ancho por lado que las barras reales.
- $\mathcal{M}_2$: las barras de 6, 4 y 2 cm aparecen en las tres ventanas independientes (posiciones repetibles a ±0.3 cm); la de 1 cm no se resuelve.
- Entre intervalos independientes, la correlación de la estructura fina es de 0.85–0.89 en densidad para $\mathcal{M}_2$, y de 0.46–0.72 en probabilidad para $\mathcal{M}_1$ arena.

---

## 9. Limitaciones conocidas (para tener en cuenta al analizar o redactar)

1. **Une barras partidas:** une huecos de hasta ~6 cm en barras de Ø2 cm (~8 cm en Ø4 cm), por encima de lo que muestran los datos. El entrenamiento casi no tiene barras con huecos colineales, y clDice favorece la continuidad.
2. **El hierro sale más ancho que las barras reales,** por la PSF y la incertidumbre; la red es conservadora.
3. **Bandas fijas del detector** (y ≈ 10, y > 49.5, x → 0): aun modeladas, generan falso hierro en los bordes. En $\mathcal{M}_2$ queda una mancha cerca del escalón superior incluso dentro de RED ≥ 0.35.
4. **Valores absolutos de ∫dl/X₀:** solo son tan buenos como la calibración de V y del contraste del hierro (sorteado al 55–95%). Interpreta los resultados como **estructura** (dónde está el hierro), no como densidad absoluta.
5. **Huecos de aire** (oclusiones): no se detectan de forma confiable y son de baja prioridad.

---

## 10. Mapa rápido de archivos

| Archivo | Para qué |
|---|---|
| `data_generator.py` | Escenas, proyección, respuesta, ruido; constantes calibradas |
| `train_Unet_2.py` | Dataset, pérdidas y entrenamiento (servidor) |
| `Unet.py` | Arquitectura |
| `evaluar_test.py` | Evaluación en simulación y umbral de hierro |
| `inference_unet_2_real.py` | Inferencia real y estilo de figuras |
| `preparar_recortada.py` | Mapas reales (media recortada, N, intervalos) |
| `calibrar_respuesta.py` | Ajuste de V = V₀ + A·T^γ |
| `calibrar_bordes.py` | Bandas del detector |
| `confiabilidad.py` | Zona confiable por RED |
| `bootstrap_muones.py` | Incertidumbre por remuestreo |
| `figuras_tesis.py` | Todas las figuras de tesis |
| `figura_entrenamiento.py` | Figura explicativa de los datos de entrenamiento |
| `calibracion/` | aceptancia, RED, banco de ruido, θ² reales, cenitales, bandas |
| `datos_reales/` | Mapas reales de entrada (`*_varrec_xy.npy`, `*_N_xy.npy`) |
| `modelos/V16_fp16.pt` | Pesos de la red final (usar estos en el PC) |
| `resultados/V16/` | Todas las salidas de V16 |
| `../Datos/` | Voxelizaciones por muón (fuera de NN2, solo en el PC) |

**Uso mínimo en el PC:**
```python
import numpy as np, inference_unet_2_real as inf
model, device = inf.cargar_modelo("modelos/V16_fp16.pt")
v = inf.limpiar(np.load("datos_reales/Grosores_mortero_20d_20.6d_varrec_xy.npy"))
N = inf.limpiar(np.load("datos_reales/Grosores_mortero_20d_20.6d_N_xy.npy"))
varianza, densidad, prob_hierro = inf.inferir_mapa(model, device, v, N)   # mapas [x, y] del detector
```
