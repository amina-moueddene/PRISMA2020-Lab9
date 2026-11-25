#!/usr/bin/env python3
"""Lab 9 — Analyser les études et identifier les Research Gaps (PRISMA 2020).

Revue systématique : Machine learning and remote sensing for disease and stress detection in olive.
Corpus : 22 études incluses extraites et évaluées au Lab 8.

Usage : python scripts/analyze_and_gaps.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

STOP_WORDS_FR_EN = {
    "de", "du", "des", "la", "le", "les", "et", "en", "un", "une", "au", "aux", "par", "pour",
    "sur", "dans", "avec", "sans", "non", "pas", "plus", "ou", "a", "à", "ce", "ces", "se", "sa",
    "son", "ses", "leur", "leurs", "qui", "que", "dont", "est", "sont", "the", "and", "of", "to",
    "in", "for", "on", "with", "not", "no", "by", "from", "as", "at", "an", "or", "is", "are",
}


def clean_text(text):
    """Normalise une chaîne libre (minuscules, espaces, abréviations courantes)."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "cnn": "convolutional neural network",
        "lstm": "long short-term memory",
        "svm": "support vector machine",
        "rf": "random forest",
        "lda": "linear discriminant analysis",
        "qda": "quadratic discriminant analysis",
    }
    for abbr, full in replacements.items():
        text = re.sub(rf"\b{abbr}\b", full, text)
    return text


def split_metrics(text):
    """Découpe la chaîne de métriques en liste normalisée."""
    if pd.isna(text) or not str(text).strip() or str(text).strip().lower() in ("none", "nr", "nan"):
        return []
    return [m.strip() for m in re.split(r"[;,]", str(text)) if m.strip()]


def categorize_approach(row):
    ai = str(row.get("ai_family", "")).lower()
    alg = str(row.get("algorithms", "")).lower()
    if "deep learning" in ai:
        if "forest" in alg or "svm" in alg or "geostat" in alg:
            return "Hybride (ML + DL)"
        return "Deep Learning"
    if "classical ml" in ai or "ann" in ai:
        return "Machine Learning Classique"
    return "Autre"


def categorize_modality(mod):
    m = str(mod).lower()
    if "multimodal" in m or "+" in m:
        return "Multimodal (Fusion)"
    if "hyperspectral" in m:
        return "Hyperspectral"
    if "multispectral" in m:
        return "Multispectral"
    if "rgb" in m:
        return "RGB"
    return "Autre"


def categorize_domain(target):
    t = str(target).lower()
    if "xylella" in t and "verticillium" in t:
        return "Multi-pathogènes (Xylella + Verticillium)"
    if "xylella" in t:
        return "Xylella fastidiosa"
    if "verticillium" in t:
        return "Verticillium dahliae"
    if "water" in t or "drought" in t or "physiological" in t:
        return "Stress hydrique & abiotique"
    if "leaf" in t or "disease" in t:
        return "Maladies foliaires"
    return "Stress mixte (biotique/abiotique)"


def extract_numeric_metrics(raw_metrics, master_perf, master_metric):
    txt = str(raw_metrics) if not pd.isna(raw_metrics) else ""
    out = {"OA": np.nan, "Recall": np.nan, "Precision": np.nan, "F1": np.nan, "Kappa": np.nan, "R2": np.nan}

    def norm_val(val_str):
        v = float(val_str)
        return round(v / 100.0, 4) if v > 1.0 else round(v, 4)

    m_oa = re.search(r"(?:OA|overall accuracy|accuracy)[^0-9]{0,12}(\d+(?:\.\d+)?)", txt, re.I)
    if m_oa:
        out["OA"] = norm_val(m_oa.group(1))
    elif str(master_metric).upper() == "OA" and not pd.isna(master_perf):
        out["OA"] = norm_val(master_perf)

    m_rec = re.search(r"(?:recall|sensitivity)[^0-9]{0,10}(\d+(?:\.\d+)?)", txt, re.I)
    if m_rec:
        out["Recall"] = norm_val(m_rec.group(1))
    elif str(master_metric).lower() in ("recall", "sensitivity") and not pd.isna(master_perf):
        out["Recall"] = norm_val(master_perf)

    m_prec = re.search(r"precision[^0-9]{0,10}(\d+(?:\.\d+)?)", txt, re.I)
    if m_prec:
        out["Precision"] = norm_val(m_prec.group(1))

    m_f1 = re.search(r"F1[^0-9]{0,25}(\d+(?:\.\d+)?)", txt, re.I)
    if m_f1:
        out["F1"] = norm_val(m_f1.group(1))

    m_kap = re.search(r"(?:κ|kappa)[^0-9]{0,8}(\d+(?:\.\d+)?)", txt, re.I)
    if m_kap:
        out["Kappa"] = round(float(m_kap.group(1)), 4)

    m_r2 = re.search(r"R²[^0-9]{0,8}(\d+(?:\.\d+)?)", txt, re.I)
    if m_r2:
        out["R2"] = round(float(m_r2.group(1)), 4)

    return pd.Series(out)


def derive_study_limitations(row):
    lims = []
    val = str(row.get("validation_level", ""))
    tr = str(row.get("transfer_test", ""))
    ref = str(row.get("reference_data", "")).lower()
    early = str(row.get("early_detection", "")).lower()
    code = str(row.get("data_code_availability", "")).lower()
    exp = str(row.get("explainability", ""))
    dep = str(row.get("deployment_level", ""))
    met = str(row.get("metrics", "")).lower()

    if val in ("V0", "V1", "V2", "V3") or tr.lower().startswith("none") or tr == "nan":
        lims.append("Absence de transfert spatial ou temporel indépendant")
    elif "recall fell to 10%" in tr.lower() or "poor" in tr.lower():
        lims.append("Chute de généralisation inter-zones ou inter-années")

    if "pcr" not in ref and "water potential" not in ref and "fluorescence" not in ref:
        lims.append("Absence de validation biologique ou physiologique terrain")

    if "symptomatic" in early or "not demonstrated" in early or "not addressed" in early or early == "nr":
        lims.append("Détection limitée au stade symptomatique tardif")

    if code in ("no", "nr", "nan") or "not" in code or "request" in code or "figures only" in code:
        lims.append("Code et données non partagés (faible reproductibilité)")

    if exp in ("nan", "NR", "") or pd.isna(row.get("explainability")):
        lims.append("Manque d'explicabilité des prédictions")

    if dep in ("DRL1", "DRL2"):
        lims.append("Absence de déploiement opérationnel temps réel")

    if "radar chart only" in met or met.startswith("none"):
        lims.append("Métriques quantitatives incomplètes")

    return " ; ".join(lims)


def clean_limitation_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS_FR_EN and len(t) > 2]
    return " ".join(tokens)


def get_ngrams(text, n=2):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def run():
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(DATA / "lab8_consolidated_extraction_qa.csv")
    sm = pd.read_csv(DATA / "studies_master_v2.csv")[["record", "reported_performance", "metric"]]
    df = df.merge(sm, on="record", how="left")

    df["Approche_brut"] = df["ai_family"]
    df["Modèle_brut"] = df["algorithms"]
    df["Métriques_brut"] = df["metrics"]

    df["Approche"] = df["ai_family"].apply(clean_text)
    df["Modèle"] = df["algorithms"].apply(clean_text)
    df["Métriques"] = df["metrics"].apply(clean_text)
    df["Métriques_list"] = df["metrics"].apply(split_metrics)

    df["Catégorie_Approche"] = df.apply(categorize_approach, axis=1)
    df["Catégorie_Modalité"] = df["modality"].apply(categorize_modality)
    df["Catégorie_Domaine"] = df["target"].apply(categorize_domain)

    num_metrics = df.apply(
        lambda r: extract_numeric_metrics(r["metrics"], r["reported_performance"], r["metric"]),
        axis=1,
    )
    df = pd.concat([df, num_metrics], axis=1)

    df["Limitation"] = df.apply(derive_study_limitations, axis=1)
    df["Limitation_clean"] = df["Limitation"].apply(clean_limitation_text)

    # Tableau comparatif (22 études)
    comp_cols = pd.DataFrame({
        "Étude": df["study"],
        "Année": df["year"],
        "Plateforme": df["platform_norm"],
        "Modalité": df["Catégorie_Modalité"],
        "Domaine_Cible": df["Catégorie_Domaine"],
        "Approche": df["Catégorie_Approche"],
        "Modèle": df["Modèle_brut"],
        "Validation": df["validation_level"],
        "Score_QA": df["quality_total_out_of_20"].astype(str) + "/20",
        "Métriques": df["Métriques_brut"].fillna("Non rapporté"),
        "Résultat_principal": df.apply(
            lambda r: f"{r['metric']}={r['reported_performance']:.3f}"
            if pd.notna(r["reported_performance"])
            else "Qualitatif / Radar chart",
            axis=1,
        ),
        "Limitation": df["Limitation"],
    })
    comp_cols.to_csv(OUT / "tableau_comparatif.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "# Tableau Comparatif des 22 Études Incluses (PRISMA 2020 — Lab 9)",
        "",
        "| Étude | Année | Plateforme | Modalité | Domaine Cible | Approche | Modèle | Val. | QA | Résultat principal | Limitations identifiées |",
        "| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |",
    ]
    for _, r in comp_cols.iterrows():
        md_lines.append(
            f"| {r['Étude']} | {r['Année']} | {r['Plateforme']} | {r['Modalité']} | "
            f"{r['Domaine_Cible']} | {r['Approche']} | {r['Modèle']} | {r['Validation']} | "
            f"{r['Score_QA']} | {r['Résultat_principal']} | {r['Limitation']} |"
        )
    (OUT / "tableau_comparatif.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Statistiques descriptives des performances
    stats_approche = df.groupby("Catégorie_Approche")[["OA", "Recall", "Kappa", "R2", "quality_total_out_of_20"]].agg(
        ["count", "mean", "std", "median"]
    )
    stats_approche.to_csv(OUT / "statistiques_performances.csv", encoding="utf-8-sig")

    # Graphique 1 : Tendances temporelles (2015–2026)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5.5))
    all_years = list(range(int(df["year"].min()), int(df["year"].max()) + 1))
    year_counts = df["year"].value_counts().reindex(all_years, fill_value=0)
    colors_yr = sns.color_palette("viridis", len(year_counts))
    ax = sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, palette=colors_yr)
    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2.0, h + 0.12), ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.title("Évolution temporelle des publications incluses (n = 22 études, 2015–2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Année de publication", fontsize=11)
    plt.ylabel("Nombre d'études incluses", fontsize=11)
    plt.ylim(0, max(year_counts.values) + 1.2)
    plt.tight_layout()
    plt.savefig(OUT / "tendances_annuelles.png", dpi=200)
    plt.close()

    # Graphique 2 : Répartition des familles de modèles
    plt.figure(figsize=(11, 6))
    model_mentions = Counter()
    for alg in df["algorithms"]:
        a = str(alg).lower()
        if "svm" in a:
            model_mentions["Support Vector Machine (SVM)"] += 1
        if "random forest" in a:
            model_mentions["Random Forest (RF)"] += 1
        if "lda" in a or "qda" in a or "discriminant" in a:
            model_mentions["Analyse Discriminante (LDA/QDA)"] += 1
        if "cnn" in a or "resnet" in a or "mobilenet" in a:
            model_mentions["CNN / ResNet / MobileNet"] += 1
        if "xgboost" in a or "gradient boosting" in a or "gbm" in a or "gtb" in a:
            model_mentions["Gradient Boosting (XGBoost/GBM)"] += 1
        if "yolo" in a:
            model_mentions["YOLOv5 (Détection d'objets)"] += 1
        if "gaussian process" in a:
            model_mentions["Gaussian Process (GP)"] += 1
        if "ann" in a:
            model_mentions["Artificial Neural Network (ANN)"] += 1
        if "svr" in a or "mlr" in a or "lm" in a:
            model_mentions["Régression (SVR / MLR / LM)"] += 1

    mod_series = pd.Series(dict(model_mentions.most_common(10)))
    colors_mod = sns.color_palette("magma", len(mod_series))
    ax2 = sns.barplot(y=mod_series.index, x=mod_series.values, palette=colors_mod)
    for p in ax2.patches:
        w = int(p.get_width())
        ax2.annotate(f"{w}", (w + 0.15, p.get_y() + p.get_height() / 2.0), va="center", fontweight="bold", fontsize=10)
    plt.title("Top des familles de modèles utilisés dans les 22 études incluses", fontsize=13, fontweight="bold")
    plt.xlabel("Nombre d'études utilisant le modèle", fontsize=11)
    plt.ylabel("Famille de modèle", fontsize=11)
    plt.xlim(0, max(mod_series.values) + 1.5)
    plt.tight_layout()
    plt.savefig(OUT / "top_modeles.png", dpi=200)
    plt.close()

    # Extraction des bigrammes fréquents sur les limitations
    all_bigrams = []
    for txt in df["Limitation_clean"]:
        all_bigrams.extend(get_ngrams(txt, n=2))
    bigram_counts = Counter(all_bigrams)
    lim_df = pd.DataFrame(bigram_counts.most_common(15), columns=["Bigramme_Limitation", "Fréquence"])
    lim_df.to_csv(OUT / "limitations_frequentes.csv", index=False, encoding="utf-8-sig")

    # Comptage par étude distincte (potential_gaps.csv)
    lim_to_studies = {}
    for _, r in df.iterrows():
        for item in [x.strip() for x in r["Limitation"].split(";") if x.strip()]:
            lim_to_studies.setdefault(item, []).append(r["study"])

    gap_rows = []
    for lim_label, studies_list in sorted(lim_to_studies.items(), key=lambda x: -len(x[1])):
        nb = len(studies_list)
        if nb >= 3:
            gap_rows.append({
                "Limitation_Récurrente": lim_label,
                "Nb_études": nb,
                "Pourcentage_corpus": round(100.0 * nb / len(df), 1),
                "Études_concernées": ", ".join(studies_list),
            })
    potential_gaps_df = pd.DataFrame(gap_rows)
    potential_gaps_df.to_csv(OUT / "potential_gaps.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total_studies": int(len(df)),
        "by_approach": df["Catégorie_Approche"].value_counts().to_dict(),
        "by_modality": df["Catégorie_Modalité"].value_counts().to_dict(),
        "by_domain": df["Catégorie_Domaine"].value_counts().to_dict(),
        "by_validation": df["validation_level"].value_counts().to_dict(),
        "mean_qa_score": round(float(df["quality_total_out_of_20"].mean()), 2),
        "median_qa_score": round(float(df["quality_total_out_of_20"].median()), 2),
        "potential_gaps_count": int(len(potential_gaps_df)),
    }
    (OUT / "synthese_lab9.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Lab 9 exécuté avec succès :", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    run()
