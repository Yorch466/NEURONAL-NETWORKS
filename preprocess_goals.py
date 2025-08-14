import os, argparse, glob
import numpy as np
import pandas as pd

def mmss_to_sec(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if ":" not in s: 
        try: return float(s)
        except: return np.nan
    m, sec = s.split(":")
    return int(m)*60 + int(sec)

def melt_scores(path, test_name, value_name):
    df = pd.read_csv(path)
    df = df.rename(columns={"sexo":"sex"})
    # columnas de edades (toman exactamente estos encabezados)
    age_cols = [c for c in df.columns if c in ["20-24","25-29","30-34","35-39","40-44","45-49","50+"]]
    out = df.melt(
        id_vars=["sex","prueba","calificacion"] + ([c for c in ["tiempo","tiempo_seg"] if c in df.columns]),
        value_vars=age_cols, var_name="age_band", value_name=value_name
    ).dropna(subset=[value_name])
    out["test"] = test_name
    out = out.rename(columns={"calificacion":"grade"})
    # tiempo en segundos si aplica
    if "time_sec" in [value_name]:
        if "tiempo_seg" in out.columns:
            out["time_sec"] = pd.to_numeric(out["tiempo_seg"], errors="coerce")
        elif "tiempo" in out.columns:
            out["time_sec"] = out["tiempo"].apply(mmss_to_sec)
        out = out.drop(columns=[c for c in ["tiempo","tiempo_seg"] if c in out.columns])
    return out[["sex","test","age_band","grade",value_name]]

def expand_pesotalla(path, step_w=0.5):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "sexo":"sex",
        "categoria":"status",
        "estatura_min_cm":"height_min",
        "estatura_max_cm":"height_max",
        "peso_min_kg":"weight_min",
        "peso_max_kg":"weight_max"
    })
    # asegurar numéricos
    for c in ["height_min","height_max","weight_min","weight_max","peso_ideal_kg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rows = []
    for _,r in df.iterrows():
        if pd.isna(r.height_min) or pd.isna(r.height_max): continue
        hs = list(range(int(r.height_min), int(r.height_max)+1))
        if pd.isna(r.weight_min) or pd.isna(r.weight_max):
            # solo ideal (desnutrición/obesidad extremos): usa solo el ideal si existe
            if not pd.isna(r.peso_ideal_kg):
                for h in hs:
                    rows.append([r["sex"], h, float(r.peso_ideal_kg), r["status"], float(r.peso_ideal_kg)])
            continue
        ws = np.round(np.arange(float(r.weight_min), float(r.weight_max)+1e-6, step_w), 1)
        for h in hs:
            for w in ws:
                rows.append([r["sex"], h, float(w), r["status"], float(r.peso_ideal_kg) if not pd.isna(r.peso_ideal_kg) else np.nan])
    out = pd.DataFrame(rows, columns=["sex","height_cm","weight_kg","status","peso_ideal_kg"])
    return out

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- Aeróbica
    for sx in ["damas","varones"]:
        f = os.path.join(in_dir, f"{sx}_aerobica_3200m.csv")
        if os.path.isfile(f):
            aero = melt_scores(f, "aerobica_3200m", "time_sec")
            aero.to_csv(os.path.join(out_dir, f"{sx}_aerobica_3200m_long.csv"), index=False)

    # --- Flexiones
    for sx in ["damas","varones"]:
        f = os.path.join(in_dir, f"{sx}_flexiones_2min.csv")
        if os.path.isfile(f):
            flex = melt_scores(f, "flexiones_2min", "reps")
            flex.to_csv(os.path.join(out_dir, f"{sx}_flexiones_2min_long.csv"), index=False)

    # --- Abdominales
    for sx in ["damas","varones"]:
        f = os.path.join(in_dir, f"{sx}_abdominales_2min.csv")
        if os.path.isfile(f):
            abs_ = melt_scores(f, "abdominales_2min", "reps")
            abs_.to_csv(os.path.join(out_dir, f"{sx}_abdominales_2min_long.csv"), index=False)

    # --- Peso–talla (rango → rejilla finita)
    pt_candidates = glob.glob(os.path.join(in_dir, "*pesotalla*.csv")) + \
                    glob.glob(os.path.join(in_dir, "*peso*talla*.csv")) + \
                    [os.path.join(in_dir, "comb_pesotalla_long_finite.csv")]
    pt_candidates = [p for p in pt_candidates if os.path.isfile(p)]
    if pt_candidates:
        # usa el primero que encuentre
        pt = expand_pesotalla(pt_candidates[0], step_w=0.5)
        pt.to_csv(os.path.join(out_dir, "comb_pesotalla_long_finite.csv"), index=False)

    print(f"OK. Archivos limpios en: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.in_dir, args.out_dir)
