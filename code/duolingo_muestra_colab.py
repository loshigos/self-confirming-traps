# -*- coding: utf-8 -*-
"""
Muestra de ~1 millón de filas del conjunto Duolingo half-life regression (Settles & Meeder, 2016)
para el paper Caro_et_al_SelfConfirming_AIJ2027.

Uso en Google Colab: pegar todo en una celda y ejecutar.

La muestra se toma por USUARIOS completos, no por filas sueltas: se conservan todas las
repeticiones de cada usuario elegido, para no romper la historia de práctica y olvido
de cada palabra. La selección usa un hash del identificador del usuario, así que es
reproducible (la misma semilla da la misma muestra).

Salida: duolingo_muestra_1M.csv.gz (unos 30-40 MB), que se descarga al final.
"""
import os, sys, hashlib, json, urllib.request
import pandas as pd

OBJETIVO_FILAS = 1_000_000           # tamaño aproximado de la muestra
TOTAL_FILAS = 12_854_226              # filas del archivo completo (aprox.)
SEMILLA = "caro2027"                  # cambia la muestra si se cambia
DOI = "doi:10.7910/DVN/N8XJME"
ARCHIVO = "learning_traces.13m.csv.gz"
SALIDA = "duolingo_muestra_1M.csv.gz"

# ---------------------------------------------------------------- 1. descarga
def descargar():
    if os.path.exists(ARCHIVO) and os.path.getsize(ARCHIVO) > 300e6:
        print("Archivo ya descargado."); return
    api = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={DOI}"
    meta = json.load(urllib.request.urlopen(api))
    archivos = meta["data"]["latestVersion"]["files"]
    elegido = max(archivos, key=lambda f: f["dataFile"].get("filesize", 0))   # el CSV grande
    fid = elegido["dataFile"]["id"]
    nombre = elegido["dataFile"]["filename"]
    print(f"Descargando {nombre} ({elegido['dataFile'].get('filesize', 0)/1e6:.0f} MB) ...")
    url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
    os.system(f'wget -q --show-progress -O "{ARCHIVO}" "{url}"')
    if not os.path.exists(ARCHIVO) or os.path.getsize(ARCHIVO) < 1e6:
        sys.exit("La descarga falló. Descargue el archivo a mano desde "
                 "https://dataverse.harvard.edu/dataset.xhtml?persistentId=" + DOI +
                 f" y súbalo a Colab con el nombre {ARCHIVO}.")

descargar()

# Algunos servidores entregan el archivo ya descomprimido: se detecta por los primeros bytes.
with open(ARCHIVO, "rb") as fh:
    comprimido = fh.read(2) == b"\x1f\x8b"
compresion = "gzip" if comprimido else None

# ---------------------------------------------------------------- 2. muestra por usuarios
fraccion = OBJETIVO_FILAS / TOTAL_FILAS
def usuario_elegido(u):
    h = int(hashlib.md5((SEMILLA + str(u)).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraccion

partes, leidas = [], 0
for trozo in pd.read_csv(ARCHIVO, compression=compresion, chunksize=500_000):
    leidas += len(trozo)
    usuarios = trozo["user_id"].unique()
    mapa = {u: usuario_elegido(u) for u in usuarios}
    partes.append(trozo[trozo["user_id"].map(mapa)])
    print(f"  leídas {leidas:,} filas, muestra {sum(len(p) for p in partes):,}", end="\r")
muestra = pd.concat(partes, ignore_index=True)
print()

# ---------------------------------------------------------------- 3. resumen y guardado
muestra = muestra.sort_values(["user_id", "lexeme_id", "timestamp"]).reset_index(drop=True)
print(f"Filas en la muestra: {len(muestra):,}")
print(f"Usuarios: {muestra['user_id'].nunique():,} | palabras distintas: {muestra['lexeme_id'].nunique():,}")
print("Idiomas aprendidos:", muestra["learning_language"].value_counts().to_dict())
print("Días desde la última práctica (delta/86400), cuantiles:",
      (muestra["delta"] / 86400).quantile([.1, .5, .9, .99]).round(2).to_dict())
print("Recuerdo medio en la sesión (p_recall):", round(muestra["p_recall"].mean(), 3))

muestra.to_csv(SALIDA, index=False, compression="gzip")
print(f"Guardado {SALIDA} ({os.path.getsize(SALIDA)/1e6:.1f} MB)")

try:
    from google.colab import files
    files.download(SALIDA)
except ImportError:
    pass
