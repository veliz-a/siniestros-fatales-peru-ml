from pathlib import Path

import numpy as np
import pandas as pd
import unicodedata


# ============================================================
# Enriquecimiento relacional ONSV: Personas y Vehiculos
# Proyecto: siniestros-fatales-peru-ml
#
# Este script NO sobrescribe siniestros_enriquecido.csv.
# Genera una nueva version enriquecida para revision:
#   data/procesada/siniestros_enriquecido_onsv.csv
#   data/procesada/trazabilidad_variables_onsv_relacional.csv
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESADA = ROOT / "data" / "procesada"

PATHS = {
    "siniestros_raw": DATA_RAW / "BBDD ONSV - SINIESTROS FATALES 2021-2025 (preliminar).xlsx",
    "personas_raw": DATA_RAW / "BBDD ONSV - PERSONAS 2021-2025 (preliminar) (1).xlsx",
    "vehiculos_raw": DATA_RAW / "BBDD ONSV - VEHICULOS 2021-2025 (preliminar) (1).xlsx",
    "siniestros_enriquecido_actual": DATA_PROCESADA / "siniestros_enriquecido.csv",
    "salida_enriquecido_onsv": DATA_PROCESADA / "siniestros_enriquecido_onsv.csv",
    "salida_trazabilidad": DATA_PROCESADA / "trazabilidad_variables_onsv_relacional.csv",
}


def normalizar_texto(valor):
    """Normaliza texto para comparar categorias: mayusculas, sin tildes y sin espacios repetidos."""
    if pd.isna(valor):
        return ""
    texto = str(valor).strip().upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(texto.split())


def buscar_columna(df, nombre_objetivo):
    """Busca una columna ignorando tildes y diferencias de mayusculas."""
    objetivo = normalizar_texto(nombre_objetivo)
    for col in df.columns:
        if normalizar_texto(col) == objetivo:
            return col
    for col in df.columns:
        if objetivo in normalizar_texto(col):
            return col
    raise KeyError(f"No se encontro la columna esperada: {nombre_objetivo}")


def leer_excel_onsv(path, sheet_name):
    """Lee archivos ONSV, donde los encabezados empiezan despues de 4 filas de notas."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=4)
    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validar_unicidad_por_siniestro(df_agg, nombre):
    if df_agg["CODIGO_SINIESTRO"].duplicated().any():
        dup = df_agg.loc[df_agg["CODIGO_SINIESTRO"].duplicated(), "CODIGO_SINIESTRO"].head().tolist()
        raise ValueError(f"{nombre} tiene codigos duplicados despues de agrupar: {dup}")


def validar_merge(df_antes, df_despues, nombre_fuente, columnas_nuevas):
    """Valida que el merge no altere filas ni distribucion de la variable objetivo."""
    print("=" * 80)
    print(f"VALIDACION POST-MERGE: {nombre_fuente}")
    print("=" * 80)
    print(f"Filas antes:   {len(df_antes):,}")
    print(f"Filas despues: {len(df_despues):,}")

    assert len(df_antes) == len(df_despues), "El merge cambio la cantidad de filas."

    if "CATEGORIA_SEVERIDAD" in df_antes.columns and "CATEGORIA_SEVERIDAD" in df_despues.columns:
        dist_antes = df_antes["CATEGORIA_SEVERIDAD"].value_counts(dropna=False).sort_index().to_dict()
        dist_despues = df_despues["CATEGORIA_SEVERIDAD"].value_counts(dropna=False).sort_index().to_dict()
        assert dist_antes == dist_despues, "Cambio la distribucion de CATEGORIA_SEVERIDAD."
        print("Variable objetivo: inalterada")

    disponibles = [c for c in columnas_nuevas if c in df_despues.columns]
    cobertura = df_despues[disponibles].notna().any(axis=1).mean() * 100 if disponibles else 0
    cruzados = int(df_despues[disponibles].notna().any(axis=1).sum()) if disponibles else 0
    no_cruzados = len(df_despues) - cruzados
    print(f"Cobertura fuente: {cobertura:.2f}% ({cruzados:,} cruzados; {no_cruzados:,} no cruzados)")
    print(f"Columnas nuevas: {len(disponibles)}")
    return cobertura, cruzados, no_cruzados


def agregar_trazabilidad(registros, variables, fuente, archivo_origen, llave_cruce, nivel, cobertura, cruzados, no_cruzados, observaciones):
    for variable in variables:
        registros.append({
            "variable": variable,
            "fuente": fuente,
            "archivo_origen": str(archivo_origen),
            "llave_cruce": llave_cruce,
            "nivel": nivel,
            "cobertura_pct": round(float(cobertura), 2),
            "registros_cruzados": int(cruzados),
            "registros_no_cruzados": int(no_cruzados),
            "observaciones": observaciones,
        })


def preparar_base_con_codigo(df_actual, siniestros_raw):
    """
    La base enriquecida actual no conserva CÓDIGO SINIESTRO.
    Para no rehacer todo el notebook 01, se recupera desde el Excel original
    siempre que el orden y numero de filas coincidan.
    """
    col_codigo_raw = buscar_columna(siniestros_raw, "CODIGO SINIESTRO")

    if "CODIGO_SINIESTRO" in df_actual.columns:
        return df_actual.copy()

    if "CÓDIGO SINIESTRO" in df_actual.columns:
        df = df_actual.rename(columns={"CÓDIGO SINIESTRO": "CODIGO_SINIESTRO"}).copy()
        return df

    assert len(df_actual) == len(siniestros_raw), (
        "No se puede recuperar CODIGO_SINIESTRO por orden porque las filas no coinciden. "
        f"Actual={len(df_actual):,}; raw={len(siniestros_raw):,}"
    )

    df = df_actual.copy()
    df.insert(0, "CODIGO_SINIESTRO", siniestros_raw[col_codigo_raw].astype(str).values)
    print("Se recupero CODIGO_SINIESTRO desde el Excel original de siniestros, manteniendo el orden de filas.")
    return df


def crear_agregados_personas(personas):
    col_codigo = buscar_columna(personas, "CODIGO SINIESTRO")
    col_tipo = buscar_columna(personas, "TIPO PERSONA")
    col_edad = buscar_columna(personas, "EDAD")
    col_sexo = buscar_columna(personas, "SEXO")
    col_estado_lic = buscar_columna(personas, "ESTADO LICENCIA")
    col_dosaje = buscar_columna(personas, "RESULTADO DEL DOSAJE ETILICO CUALITATIVO")

    df = personas.copy()
    df["CODIGO_SINIESTRO"] = df[col_codigo].astype(str)
    df["_TIPO_N"] = df[col_tipo].apply(normalizar_texto)
    df["_SEXO_N"] = df[col_sexo].apply(normalizar_texto)
    df["_ESTADO_LIC_N"] = df[col_estado_lic].apply(normalizar_texto)
    df["_DOSAJE_N"] = df[col_dosaje].apply(normalizar_texto)
    df["_EDAD_NUM"] = pd.to_numeric(df[col_edad], errors="coerce")

    # IMPORTANTE: No se usa GRAVEDAD, LUGAR ATENCION LESIONADO ni LUGAR DE DEFUNCION,
    # porque revelan directamente el resultado del siniestro y generan fuga de informacion.
    es_conductor = df["_TIPO_N"].str.contains("CONDUCTOR", na=False)
    es_conductor_fugado = df["_TIPO_N"].str.contains("CONDUCTOR FUGADO", na=False)
    es_peaton = df["_TIPO_N"].str.contains("PEATON", na=False)
    es_pasajero_ocupante = df["_TIPO_N"].str.contains("PASAJERO|OCUPANTE", na=False)
    lic_problema = df["_ESTADO_LIC_N"].str.contains("SIN LICENCIA|VENCID|SUSPEND|CANCEL|NO VIGENTE", na=False)
    dosaje_pos = df["_DOSAJE_N"].str.contains("POSITIVO", na=False)
    dosaje_eval = df["_DOSAJE_N"].isin(["POSITIVO", "NEGATIVO"])
    sexo_masc = df["_SEXO_N"].eq("MASCULINO")

    base = pd.DataFrame({"CODIGO_SINIESTRO": df["CODIGO_SINIESTRO"].dropna().unique()})
    g = df.groupby("CODIGO_SINIESTRO")
    agg = base.set_index("CODIGO_SINIESTRO")

    agg["PER_TOTAL_PERSONAS"] = g.size()
    agg["PER_N_CONDUCTORES"] = df[es_conductor].groupby("CODIGO_SINIESTRO").size()
    agg["PER_N_CONDUCTORES_FUGADOS"] = df[es_conductor_fugado].groupby("CODIGO_SINIESTRO").size()
    agg["PER_N_PEATONES"] = df[es_peaton].groupby("CODIGO_SINIESTRO").size()
    agg["PER_N_PASAJEROS_OCUPANTES"] = df[es_pasajero_ocupante].groupby("CODIGO_SINIESTRO").size()

    conductores = df[es_conductor].copy()
    agg["PER_EDAD_COND_PROM"] = conductores.groupby("CODIGO_SINIESTRO")["_EDAD_NUM"].mean()
    agg["PER_EDAD_COND_MIN"] = conductores.groupby("CODIGO_SINIESTRO")["_EDAD_NUM"].min()
    agg["PER_EDAD_COND_MAX"] = conductores.groupby("CODIGO_SINIESTRO")["_EDAD_NUM"].max()
    agg["PER_ANY_COND_MASCULINO"] = sexo_masc[es_conductor].groupby(df.loc[es_conductor, "CODIGO_SINIESTRO"]).max().astype(float)
    agg["PER_ANY_LIC_PROBLEMA"] = lic_problema[es_conductor].groupby(df.loc[es_conductor, "CODIGO_SINIESTRO"]).max().astype(float)
    agg["PER_ANY_DOSAJE_POS"] = dosaje_pos[es_conductor].groupby(df.loc[es_conductor, "CODIGO_SINIESTRO"]).max().astype(float)
    agg["PER_ANY_DOSAJE_EVALUADO"] = dosaje_eval[es_conductor].groupby(df.loc[es_conductor, "CODIGO_SINIESTRO"]).max().astype(float)

    cols_count = [
        "PER_N_CONDUCTORES", "PER_N_CONDUCTORES_FUGADOS", "PER_N_PEATONES", "PER_N_PASAJEROS_OCUPANTES"
    ]
    agg[cols_count] = agg[cols_count].fillna(0).astype(int)

    agg = agg.reset_index()
    validar_unicidad_por_siniestro(agg, "personas_agg")
    return agg


def crear_agregados_vehiculos(vehiculos):
    col_codigo = buscar_columna(vehiculos, "CODIGO SINIESTRO")
    col_situacion = buscar_columna(vehiculos, "SITUACION VEHICULO")
    col_modalidad = buscar_columna(vehiculos, "MODALIDAD DE TRANSPORTE")
    col_estado_soat = buscar_columna(vehiculos, "ESTADO SOAT")
    col_estado_citv = buscar_columna(vehiculos, "ESTADO CITV")
    col_posee_seguro = buscar_columna(vehiculos, "POSEE SEGURO")
    col_vehiculo = buscar_columna(vehiculos, "VEHICULO")

    df = vehiculos.copy()
    df["CODIGO_SINIESTRO"] = df[col_codigo].astype(str)
    df["_SITUACION_N"] = df[col_situacion].apply(normalizar_texto)
    df["_MODALIDAD_N"] = df[col_modalidad].apply(normalizar_texto)
    df["_SOAT_N"] = df[col_estado_soat].apply(normalizar_texto)
    df["_CITV_N"] = df[col_estado_citv].apply(normalizar_texto)
    df["_POSEE_SEGURO_N"] = df[col_posee_seguro].apply(normalizar_texto)
    df["_VEHICULO_N"] = df[col_vehiculo].apply(normalizar_texto)

    fugado = df["_SITUACION_N"].str.contains("FUGADO", na=False)
    carga = df["_MODALIDAD_N"].str.contains("CARGA", na=False)
    publico = df["_MODALIDAD_N"].str.contains("PUBLICO|PASAJER|REGULAR", na=False)
    particular = df["_MODALIDAD_N"].str.contains("PARTICULAR", na=False)
    moto = df["_VEHICULO_N"].str.contains("MOTO|TRIMOTO|SCOOTER", na=False)
    auto = df["_VEHICULO_N"].str.contains("AUTOMOVIL|STATION WAGON", na=False)
    camion_bus = df["_VEHICULO_N"].str.contains("CAMION|OMNIBUS|BUS|REMOLCADOR|SEMIREMOLQUE", na=False)
    soat_problema = df["_SOAT_N"].str.contains("NO REGISTRA|VENCID|NO TIENE|SIN", na=False)
    citv_problema = df["_CITV_N"].str.contains("NO REGISTRA|VENCID|NO TIENE|SIN", na=False)
    seguro_problema = df["_POSEE_SEGURO_N"].str.contains("NO|NO ESPECIFICA", na=False)

    base = pd.DataFrame({"CODIGO_SINIESTRO": df["CODIGO_SINIESTRO"].dropna().unique()})
    g = df.groupby("CODIGO_SINIESTRO")
    agg = base.set_index("CODIGO_SINIESTRO")

    agg["VEH_TOTAL"] = g.size()
    agg["VEH_N_FUGADOS"] = df[fugado].groupby("CODIGO_SINIESTRO").size()
    agg["VEH_ANY_FUGADO"] = fugado.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_CARGA"] = carga.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_PUBLICO"] = publico.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_PARTICULAR"] = particular.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_MOTO"] = moto.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_AUTO"] = auto.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_CAMION_BUS"] = camion_bus.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_SOAT_PROBLEMA"] = soat_problema.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_CITV_PROBLEMA"] = citv_problema.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)
    agg["VEH_ANY_SEGURO_PROBLEMA"] = seguro_problema.groupby(df["CODIGO_SINIESTRO"]).max().astype(float)

    agg["VEH_N_FUGADOS"] = agg["VEH_N_FUGADOS"].fillna(0).astype(int)

    agg = agg.reset_index()
    validar_unicidad_por_siniestro(agg, "vehiculos_agg")
    return agg


def main():
    print("ROOT:", ROOT)
    for nombre, path in PATHS.items():
        if nombre.startswith("salida"):
            continue
        print(f"{nombre}: {path} -> {'OK' if path.exists() else 'NO EXISTE'}")

    siniestros_raw = leer_excel_onsv(PATHS["siniestros_raw"], sheet_name="SINIESTROS")
    personas = leer_excel_onsv(PATHS["personas_raw"], sheet_name="PERSONAS INVOLUCRADAS")
    vehiculos = leer_excel_onsv(PATHS["vehiculos_raw"], sheet_name="VEHICULO INVOLUCRADOS")
    df_actual = pd.read_csv(PATHS["siniestros_enriquecido_actual"], encoding="utf-8-sig")

    df = preparar_base_con_codigo(df_actual, siniestros_raw)
    filas_originales = len(df)
    dist_objetivo_original = (
        df["CATEGORIA_SEVERIDAD"].value_counts(dropna=False).sort_index().to_dict()
        if "CATEGORIA_SEVERIDAD" in df.columns else None
    )

    print("\nDimensiones iniciales")
    print(f"Siniestros enriquecido actual: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    print(f"Personas raw:                 {personas.shape[0]:,} filas x {personas.shape[1]:,} columnas")
    print(f"Vehiculos raw:                {vehiculos.shape[0]:,} filas x {vehiculos.shape[1]:,} columnas")

    personas_agg = crear_agregados_personas(personas)
    vehiculos_agg = crear_agregados_vehiculos(vehiculos)

    trazabilidad = []

    cols_personas = [c for c in personas_agg.columns if c != "CODIGO_SINIESTRO"]
    df_antes = df.copy()
    df = df.merge(personas_agg, on="CODIGO_SINIESTRO", how="left", validate="one_to_one")
    cov_p, cruz_p, no_p = validar_merge(df_antes, df, "ONSV Personas involucradas 2021-2025", cols_personas)
    agregar_trazabilidad(
        trazabilidad, cols_personas,
        fuente="ONSV - Personas involucradas en siniestros fatales 2021-2025",
        archivo_origen=PATHS["personas_raw"],
        llave_cruce="CODIGO_SINIESTRO",
        nivel="Siniestro, agregado desde personas involucradas",
        cobertura=cov_p,
        cruzados=cruz_p,
        no_cruzados=no_p,
        observaciones="Variables agregadas por siniestro. Se excluyen GRAVEDAD, LUGAR ATENCION LESIONADO y LUGAR DE DEFUNCION para evitar fuga de informacion.",
    )

    cols_vehiculos = [c for c in vehiculos_agg.columns if c != "CODIGO_SINIESTRO"]
    df_antes = df.copy()
    df = df.merge(vehiculos_agg, on="CODIGO_SINIESTRO", how="left", validate="one_to_one")
    cov_v, cruz_v, no_v = validar_merge(df_antes, df, "ONSV Vehiculos involucrados 2021-2025", cols_vehiculos)
    agregar_trazabilidad(
        trazabilidad, cols_vehiculos,
        fuente="ONSV - Vehiculos involucrados en siniestros fatales 2021-2025",
        archivo_origen=PATHS["vehiculos_raw"],
        llave_cruce="CODIGO_SINIESTRO",
        nivel="Siniestro, agregado desde vehiculos involucrados",
        cobertura=cov_v,
        cruzados=cruz_v,
        no_cruzados=no_v,
        observaciones="Variables agregadas por siniestro: cantidad, modalidad, tipo de vehiculo, SOAT, CITV, seguro y fuga.",
    )

    assert len(df) == filas_originales, "La cantidad final de filas cambio."
    if dist_objetivo_original is not None:
        dist_final = df["CATEGORIA_SEVERIDAD"].value_counts(dropna=False).sort_index().to_dict()
        assert dist_final == dist_objetivo_original, "La distribucion final de CATEGORIA_SEVERIDAD cambio."

    cols_nuevas = cols_personas + cols_vehiculos
    print("\n" + "=" * 80)
    print("VALIDACION GLOBAL FINAL")
    print("=" * 80)
    print(f"Filas finales:       {df.shape[0]:,}")
    print(f"Columnas iniciales:  {df_actual.shape[1]:,}")
    print(f"Columnas finales:    {df.shape[1]:,}")
    print(f"Variables nuevas:    {len(cols_nuevas):,}")
    print("Nuevas variables:")
    for c in cols_nuevas:
        print(f"- {c}: {df[c].notna().mean() * 100:.2f}% cobertura")

    DATA_PROCESADA.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS["salida_enriquecido_onsv"], index=False, encoding="utf-8-sig")
    pd.DataFrame(trazabilidad).to_csv(PATHS["salida_trazabilidad"], index=False, encoding="utf-8-sig")

    print("\nArchivos generados:")
    print(PATHS["salida_enriquecido_onsv"])
    print(PATHS["salida_trazabilidad"])
    print("\nNota: revisar el CSV nuevo antes de reemplazar siniestros_enriquecido.csv.")


if __name__ == "__main__":
    main()
