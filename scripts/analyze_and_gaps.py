#!/usr/bin/env python3
"""Lab 9 — Analyser les études et identifier les Research Gaps (PRISMA 2020).

Revue systématique : Machine learning and remote sensing for disease and stress
detection in olive (Olea europaea L.).
Corpus : 22 études (23 rapports) incluses après lecture du texte intégral.

Toutes les catégories (plateforme, modalité, cible, famille d'IA, niveaux V et DRL,
données de référence, scores Q1–Q10) sont reprises telles quelles des fichiers de
la revue (S3 extraction, S4 qualité, S5 données codées). Les tables de codage
explicites ci-dessous (cible détaillée, algorithmes, voies de fuite,
explicabilité) reproduisent le codage multi-label du manuscrit de la revue.

Correspondance étapes du lab -> code :
  Étape 1      README.md (introduction, prérequis, exécution)
  Étapes 2–4   load()                             chargement, validation, nettoyage, classification
  Étape 5      comparative_table()                output/tableau_comparatif.csv et .md
  Étape 6      performance_stats()                output/statistiques_performances.csv
  Étapes 7–8   figures()                          output/tendances_annuelles.png, top_modeles.png, validation_et_qa.png
  Étapes 9–10  gaps()                             output/limitations_frequentes.csv, potential_gaps.csv
  Étapes 11–12 write_research_gaps_and_pistes()   output/research_gaps_et_pistes.md (Défis 1 et 2)
  Étape 13     output/verification.md             réponses aux questions Q1–Q3 et défis 1–2

Usage :
  python3 scripts/analyze_and_gaps.py
  python3 scripts/analyze_and_gaps.py --data-dir data --output-dir output --quiet
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)

ROOT: Path = Path(__file__).resolve().parent.parent
DATA: Path = ROOT / "data"
OUT: Path = ROOT / "output"

# ---------------------------------------------------------------------------
# Tables de codage (clé = numéro d'enregistrement du ledger de screening)
# ---------------------------------------------------------------------------
TARGET_DETAIL: dict[int, str] = {
    2: "Verticillium dahliae", 6: "Xylella fastidiosa", 9: "Xylella fastidiosa",
    16: "Verticillium dahliae", 20: "Xylella fastidiosa", 28: "Xylella + Verticillium",
    33: "Maladies foliaires", 37: "Stress mixte biotique/abiotique", 43: "Verticillium dahliae",
    45: "Stress mixte biotique/abiotique", 46: "Xylella + Verticillium", 53: "Stress hydrique / sécheresse",
    64: "Stress hydrique / sécheresse", 65: "Xylella fastidiosa", 73: "Stress hydrique / sécheresse",
    74: "Stress hydrique / sécheresse", 83: "Verticillium dahliae", 85: "Stress hydrique / sécheresse",
    93: "Stress hydrique / sécheresse", 109: "Verticillium dahliae", 125: "Maladies foliaires",
    137: "Stress hydrique / sécheresse",
}

ALGOS: dict[int, list[str]] = {
    2: ["SVM", "Analyse discriminante"], 6: ["SVM", "Boosting"], 9: ["Analyse discriminante"],
    16: ["CNN"], 20: ["Analyse discriminante"], 28: ["Random Forest", "SVM"], 33: ["CNN"],
    37: ["SVM", "Random Forest", "Boosting", "Analyse discriminante"],
    43: ["YOLO"], 45: ["Random Forest", "Boosting"], 46: ["SVM"],
    53: ["Random Forest", "SVM", "Régression linéaire"],
    64: ["Random Forest", "SVM", "Boosting", "kNN", "Arbre de décision", "Régression linéaire"],
    65: ["SVM"], 73: ["Random Forest"],
    74: ["Random Forest", "Boosting", "SVM", "Arbre de décision", "Naive Bayes"],
    83: ["Processus gaussien"], 85: ["SVM"],
    93: ["Boosting", "SVM", "Processus gaussien", "Random Forest", "Régression linéaire", "Réseau de neurones (MLP)"],
    109: ["Réseau de neurones (MLP)"], 125: ["CNN"], 137: ["SVM"],
}

LEAKAGE: dict[str, list[int]] = {
    "Split aléatoire de pixels/ROI dans une même scène": [74, 85, 137, 37],
    "Split aléatoire de mesures répétées des mêmes arbres/zones": [53, 64, 73, 93],
    "Split au niveau image avec augmentation": [33, 125],
    "Sur-échantillonnage sans ordre déclaré vis-à-vis du split": [45, 125],
    "Classe confondue avec le site": [9],
    "Réglage et évaluation finale non séparés, ou split non décrit": [2, 45, 16],
}

XAI: dict[str, list[int]] = {
    "Entrées interprétables par construction": [2, 6, 28, 46, 65, 83, 109],
    "Importance des variables / sélection": [20, 45, 73, 93],
    "Explication post-hoc (SHAP)": [64],
}

HEADLINE: dict[int, str] = {
    2: "SVM OA 79,2 %, κ 0,50 ; LDA OA 59,0 %", 6: "OA 80,9 %, κ 0,61 ; détection pré-visuelle",
    9: "Sensibilité 0,98, précision 0,93", 16: "Délinéation des couronnes 0,89 ; maladie non évaluée",
    20: "OA 0,69 ; 0,77 pour la classe précoce", 28: "Xf : OA 92 %, κ 0,80 ; Vd : OA 98 %, κ 0,70",
    33: "Accuracy 0,971 (image)", 37: "AUROC 0,99 (QDA)", 43: "Résultats uniquement en graphique radar",
    45: "ROC-AUC 0,977 (RF), 0,955 (XGBoost)", 46: "OA 0,63–0,83, κ 0,29–0,68",
    53: "R² 0,78 (split aléatoire) ; transfert inter-annuel sans résultat utile",
    64: "R² 0,75, RMSE 0,078", 65: "OA ≥ 0,74 interne ; rappel 0,10 inter-zones (2/20)",
    73: "R² test 0,73", 74: "OA 95,9 %, κ 0,94", 83: "OA 0,77 ± 0,03, κ 0,55",
    85: "OA 96,0 %, κ 0,92", 93: "R² 0,84 (aléatoire), 0,84 (zones), 0,82 (saison)",
    109: "Aucune métrique (comparaison qualitative)", 125: "Accuracy 0,99 (image)", 137: "OA 0,927",
}

STOP_WORDS_FR_EN: set[str] = {
    "de", "du", "des", "la", "le", "les", "et", "en", "un", "une", "au", "aux", "par", "pour",
    "sur", "dans", "avec", "sans", "non", "pas", "plus", "ou", "a", "à", "ce", "ces", "se", "sa",
    "son", "ses", "leur", "leurs", "qui", "que", "dont", "est", "sont", "the", "and", "of", "to",
    "in", "for", "on", "with", "not", "no", "by", "from", "as", "at", "an", "or", "is", "are",
    "aucun", "aucune",
}

LIMITATIONS: list[tuple[str, Any]] = [
    ("Aucun test de transfert explicite entre sites ou années", lambda r: r["Q8"] < 2),
    ("Indépendance entraînement test non assurée", lambda r: r["Q5"] < 2),
    ("Référence non confirmée par laboratoire ou mesure physiologique", lambda r: r["Q3"] < 2),
    ("Voie potentielle de fuite de données", lambda r: r["n_leak"] > 0),
    ("Représentativité limitée des sites et saisons", lambda r: r["Q2"] < 2),
    ("Infection non dissociée du déficit hydrique", lambda r: r["target"] == "Vascular pathogen"),
    ("Explicabilité absente des prédictions", lambda r: r["Explicabilité"] == "Aucune"),
    ("Reproductibilité incomplète code données non publics", lambda r: r["Q9"] < 2),
    ("Preuve de déploiement opérationnel absente", lambda r: r["Q10"] < 2),
    ("Comparaison baseline absente", lambda r: r["Q7"] == 0),
    ("Métrique unique ou absente", lambda r: r["Q6"] < 2),
]

GAP_MAP: dict[str, tuple[str, str]] = {
    "Aucun test de transfert explicite entre sites ou années": ("Few transfer tests", "Haute"),
    "Indépendance entraînement test non assurée": ("Potential leakage", "Haute"),
    "Voie potentielle de fuite de données": ("Potential leakage", "Haute"),
    "Référence non confirmée par laboratoire ou mesure physiologique": ("Weak reference data", "Haute"),
    "Infection non dissociée du déficit hydrique": ("Infection confounded with water status", "Haute"),
    "Représentativité limitée des sites et saisons": ("Limited representativeness", "Moyenne"),
    "Preuve de déploiement opérationnel absente": ("No link to decisions", "Moyenne"),
    "Métrique unique ou absente": ("Unsuitable metrics for surveillance", "Moyenne"),
    "Explicabilité absente des prédictions": ("Explainability not verified", "Basse"),
    "Reproductibilité incomplète code données non publics": ("Limited reproducibility", "Basse"),
    "Comparaison baseline absente": ("(reporting : sélection du modèle)", "Basse"),
}

ABBREVIATIONS: dict[str, str] = {
    "cnn": "convolutional neural network",
    "svm": "support vector machine",
    "rf": "random forest",
    "lda": "linear discriminant analysis",
    "qda": "quadratic discriminant analysis",
}


def validate_input_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    """Valide qu'un DataFrame d'entrée n'est pas vide et contient les colonnes attendues."""
    if df.empty:
        raise ValueError(f"Le fichier d'entrée '{source_name}' est vide.")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans '{source_name}' : {', '.join(missing)}"
        )


def clean_text(text: Any) -> str:
    """Normalise une chaîne libre (minuscules, espaces, abréviations courantes)."""
    if text is None or pd.isna(text):
        return ""
    cleaned = re.sub(r"\s+", " ", str(text).lower().strip())
    for abbr, full in ABBREVIATIONS.items():
        cleaned = re.sub(rf"\b{abbr}\b", full, cleaned)
    return cleaned


def clean_limitation_text(text: Any) -> str:
    """Étape 9 — Nettoie un libellé de limitation pour l'analyse lexicale.

    Passe en minuscules, remplace tout caractère non alphabétique par un espace, puis retire
    les mots vides français/anglais et les tokens de moins de 3 lettres.
    """
    if text is None or pd.isna(text):
        return ""
    cleaned = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", str(text).lower())
    return " ".join(t for t in cleaned.split() if t not in STOP_WORDS_FR_EN and len(t) > 2)


def get_ngrams(text: str, n: int = 2) -> list[str]:
    """Étape 9 — Renvoie la liste des n-grammes (bigrammes par défaut) d'un texte nettoyé."""
    if n < 1:
        raise ValueError("La taille n des n-grammes doit être >= 1.")
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def load(data_dir: Path = DATA) -> pd.DataFrame:
    """Étapes 2 à 4 — Charge, valide, nettoie, structure et classe les 22 études."""
    consolidated_path = Path(data_dir) / "lab8_consolidated_extraction_qa.csv"
    master_path = Path(data_dir) / "studies_master_v2.csv"

    for path in (consolidated_path, master_path):
        if not path.exists():
            raise FileNotFoundError(f"Fichier d'entrée introuvable : {path}")

    df = pd.read_csv(consolidated_path, encoding="utf-8-sig")
    sm = pd.read_csv(master_path, encoding="utf-8-sig")

    validate_input_dataframe(
        df,
        ["record", "study", "year", "algorithms", "validation_level", "deployment_level", "quality_total_out_of_20"],
        consolidated_path.name,
    )
    keep = [
        "record", "country", "platform", "modality", "target", "ai_family",
        "reference_data", "early_detection", "document_type", "scenario_b",
        "scenario_c", "scenario_d", "reported_performance", "metric",
    ]
    validate_input_dataframe(sm, keep, master_path.name)

    df = df.drop(
        columns=[
            c for c in ["platform", "modality", "target", "ai_family", "reference_data", "early_detection"]
            if c in df.columns
        ]
    )
    df = df.merge(
        sm[keep].rename(columns={"country": "country_norm"}),
        on="record",
        how="left",
        validate="one_to_one",
    )
    if len(df) != 22 or set(df["record"]) != set(TARGET_DETAIL):
        raise ValueError("Le jeu de données doit contenir exactement les 22 enregistrements attendus.")

    df["Cible_détaillée"] = df["record"].map(TARGET_DETAIL)
    df["Algorithmes"] = df["record"].map(lambda r: "; ".join(ALGOS[r]))
    df["Voies_fuite"] = df["record"].map(lambda r: "; ".join(k for k, v in LEAKAGE.items() if r in v))
    df["n_leak"] = df["record"].map(lambda r: sum(r in v for v in LEAKAGE.values()))
    df["Explicabilité"] = df["record"].map(lambda r: next((k for k, v in XAI.items() if r in v), "Aucune"))
    df["Résultat_principal"] = df["record"].map(HEADLINE)
    df["Limitation"] = df.apply(lambda r: " ; ".join(lab for lab, f in LIMITATIONS if f(r)), axis=1)
    df["Limitation_clean"] = df["Limitation"].apply(clean_limitation_text)
    df["Modèle"] = df["algorithms"].apply(clean_text)

    LOGGER.info("Chargement et classification terminés : %d études.", len(df))
    return df.sort_values(["year", "record"]).reset_index(drop=True)


def comparative_table(df: pd.DataFrame, output_dir: Path = OUT) -> pd.DataFrame:
    """Étape 5 — Construit le tableau comparatif (CSV et Markdown) des 22 études."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame({
        "Étude": df["study"], "Année": df["year"], "Pays": df["country"], "Plateforme": df["platform"],
        "Modalité": df["modality"], "Cible": df["Cible_détaillée"], "Approche": df["ai_family"],
        "Algorithmes": df["Algorithmes"], "Référence": df["reference_data"], "Validation": df["validation_level"],
        "DRL": df["deployment_level"], "Score_QA": df["quality_total_out_of_20"].astype(str) + "/20",
        "Résultat_principal": df["Résultat_principal"], "Limitation": df["Limitation"],
    })
    comp.to_csv(out / "tableau_comparatif.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Tableau comparatif des 22 études incluses (PRISMA 2020 — Lab 9)",
        "",
        "Valeurs de performance citées telles que publiées : tâches, unités et métriques diffèrent, "
        "elles ne doivent pas être comparées d'une ligne à l'autre.",
        "",
        "| Étude | Année | Plateforme | Modalité | Cible | Approche | Algorithmes | Val. | DRL | QA | Résultat principal |",
        "| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for _, r in comp.iterrows():
        lines.append(
            f"| {r['Étude']} | {r['Année']} | {r['Plateforme']} | {r['Modalité']} | {r['Cible']} | "
            f"{r['Approche']} | {r['Algorithmes']} | {r['Validation']} | {r['DRL']} | {r['Score_QA']} | "
            f"{r['Résultat_principal']} |"
        )
    (out / "tableau_comparatif.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comp


def performance_stats(df: pd.DataFrame, output_dir: Path = OUT) -> dict[str, Any]:
    """Étape 6 — Statistiques descriptives des performances et des scores qualité."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["Type_métrique"] = d["metric"].replace({"OA": "OA / Accuracy", "Accuracy": "OA / Accuracy"})
    d["Groupe_validation"] = np.where(d["validation_level"].isin(["V0", "V1", "V2"]), "V0–V2", "V3–V5")
    rows: list[dict[str, Any]] = []
    for (grp_name, col) in [("Approche", "ai_family"), ("Validation", "Groupe_validation")]:
        for (g, m), sub in d.dropna(subset=["reported_performance"]).groupby([col, "Type_métrique"]):
            v = sub["reported_performance"]
            rows.append({
                "Regroupement": grp_name, "Groupe": g, "Type_métrique": m, "n_études": len(v),
                "moyenne": round(v.mean(), 3), "écart_type": round(v.std(), 3) if len(v) > 1 else np.nan,
                "médiane": round(v.median(), 3), "min": round(v.min(), 3), "max": round(v.max(), 3),
            })
    for g, sub in d.groupby("ai_family"):
        v = sub["quality_total_out_of_20"]
        rows.append({
            "Regroupement": "Approche", "Groupe": g, "Type_métrique": "Score QA (/20)", "n_études": len(v),
            "moyenne": round(v.mean(), 2), "écart_type": round(v.std(), 2), "médiane": v.median(),
            "min": v.min(), "max": v.max(),
        })
    st = pd.DataFrame(rows)
    st.to_csv(out / "statistiques_performances.csv", index=False, encoding="utf-8-sig")

    acc = d[d["Type_métrique"] == "OA / Accuracy"].dropna(subset=["reported_performance"])
    vnum = acc["validation_level"].str[1].astype(int)
    rho, p = stats.spearmanr(vnum, acc["reported_performance"])
    qa_ml = d.loc[d.ai_family == "Classical ML", "quality_total_out_of_20"]
    qa_dl = d.loc[d.ai_family == "Deep learning", "quality_total_out_of_20"]
    mw = stats.mannwhitneyu(qa_ml, qa_dl, alternative="two-sided")
    return {
        "n_accuracy_studies": int(len(acc)),
        "spearman_rho_accuracy_vs_validation": round(float(rho), 2),
        "spearman_p": round(float(p), 3),
        "median_accuracy_V0_V2": round(float(acc[vnum <= 2]["reported_performance"].median()), 2),
        "median_accuracy_V3_V4": round(float(acc[vnum >= 3]["reported_performance"].median()), 2),
        "median_qa_classical_ml": float(qa_ml.median()),
        "median_qa_deep_learning": float(qa_dl.median()),
        "mannwhitney_p_qa_ml_vs_dl": round(float(mw.pvalue), 2),
    }


def figures(df: pd.DataFrame, output_dir: Path = OUT) -> None:
    """Étapes 7 et 8 — Produit les trois figures haute résolution (300 dpi) dans output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Figure 1 : publications par année et plateforme
    SEARCH_START = 2013
    years = list(range(SEARCH_START, int(df.year.max()) + 1))
    ct = pd.crosstab(df.year, df.platform).reindex(index=years, fill_value=0)
    ct = ct[[c for c in ["Airborne", "UAV", "Satellite"] if c in ct.columns]]
    lab = {"Airborne": "Aéroporté", "UAV": "Drone (UAV)", "Satellite": "Satellite"}
    col = {"Airborne": "#0072B2", "UAV": "#E69F00", "Satellite": "#009E73"}
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(ct))
    for c in ct.columns:
        ax.bar(ct.index.astype(str), ct[c].values, bottom=bottom, color=col[c], label=lab[c], width=0.7)
        bottom += ct[c].values
    for x, h in zip(ct.index.astype(str), bottom):
        if h > 0:
            ax.text(x, h + 0.08, str(int(h)), ha="center", va="bottom", fontweight="bold")
    first = int(df.year.min())
    if first > SEARCH_START:
        ax.text(
            (first - SEARCH_START - 1) / 2,
            0.25,
            f"aucune étude incluse\n{SEARCH_START}–{first - 1}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#666",
            style="italic",
        )
    ax.set_title(
        f"Études incluses par année et par plateforme (recherche {SEARCH_START}–2026 ; n = 22 ; 2026 incomplète)",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_xlabel("Année de publication")
    ax.set_ylabel("Nombre d'études")
    ax.set_ylim(0, bottom.max() + 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tendances_annuelles.png", dpi=300)
    plt.close(fig)

    # Figure 2 : algorithmes évalués (multi-label)
    cnt = Counter(a for r in df.record for a in ALGOS[r])
    s = pd.Series(cnt).sort_values()
    dl = {"CNN", "YOLO"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(s.index, s.values, color=["#D55E00" if k in dl else "#0072B2" for k in s.index])
    for i, v in enumerate(s.values):
        ax.text(v + 0.1, i, str(v), va="center", fontweight="bold")
    ax.set_xlabel("Nombre d'études ayant évalué l'algorithme (comptage multi-label)")
    ax.set_title(
        "Algorithmes évalués dans les 22 études\n(bleu : ML classique ; orange : deep learning)",
        loc="left",
        fontweight="bold",
        fontsize=10.5,
    )
    ax.set_xlim(0, s.max() + 1.5)
    fig.tight_layout()
    fig.savefig(out / "top_modeles.png", dpi=300)
    plt.close(fig)

    # Figure 3 : niveaux de validation par famille et distribution QA
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lv = ["V0", "V1", "V2", "V3", "V4", "V5"]
    vt = pd.crosstab(df.validation_level, df.ai_family).reindex(lv, fill_value=0)
    b = np.zeros(len(lv))
    for fam, c in [("Classical ML", "#0072B2"), ("Deep learning", "#D55E00")]:
        a1.bar(lv, vt[fam].values, bottom=b, color=c, label="ML classique" if fam == "Classical ML" else "Deep learning")
        b += vt[fam].values
    for x, h in zip(lv, b):
        a1.text(x, h + 0.1, str(int(h)), ha="center", fontweight="bold")
    a1.set_title("(A) Niveau de rigueur de validation", loc="left", fontweight="bold")
    a1.set_ylabel("Nombre d'études")
    a1.legend(frameon=False)
    a1.set_ylim(0, b.max() + 1.5)
    q = df.quality_total_out_of_20
    a2.hist(q, bins=np.arange(3.5, 19.6, 1), color="#56B4E9", edgecolor="white")
    a2.axvline(q.median(), color="#333", ls="--", lw=1.3)
    a2.text(
        q.median() + 0.2,
        a2.get_ylim()[1] * 0.92,
        "médiane = " + f"{q.median():.1f}".replace(".", ",") + "/20",
        fontsize=9,
    )
    a2.set_title("(B) Score de qualité Q1–Q10 (/20)", loc="left", fontweight="bold")
    a2.set_xlabel("Score total")
    a2.set_ylabel("Nombre d'études")
    fig.tight_layout()
    fig.savefig(out / "validation_et_qa.png", dpi=300)
    plt.close(fig)


def gaps(df: pd.DataFrame, output_dir: Path = OUT) -> pd.DataFrame:
    """Étapes 9 et 10 — Bigrammes des limitations et limitations récurrentes (≥ 3 études)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    big = Counter(
        b
        for t in df["Limitation"]
        for item in t.split(";")
        for b in get_ngrams(clean_limitation_text(item), 2)
    )
    pd.DataFrame(big.most_common(15), columns=["Bigramme_Limitation", "Fréquence"]).to_csv(
        out / "limitations_frequentes.csv", index=False, encoding="utf-8-sig"
    )
    lim_to: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        for item in [x.strip() for x in r["Limitation"].split(";") if x.strip()]:
            lim_to.setdefault(item, []).append(r["study"])
    rows = [
        {
            "Limitation_Récurrente": k,
            "Nb_études": len(v),
            "Pourcentage_corpus": round(100 * len(v) / len(df), 1),
            "Gap_revue": GAP_MAP[k][0],
            "Priorité_revue": GAP_MAP[k][1],
            "Études_concernées": ", ".join(v),
        }
        for k, v in sorted(lim_to.items(), key=lambda x: -len(x[1]))
        if len(v) >= 3
    ]
    g = pd.DataFrame(rows)
    g.to_csv(out / "potential_gaps.csv", index=False, encoding="utf-8-sig")
    return g


def write_research_gaps_and_pistes(
    df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path = OUT,
) -> Path:
    """Étapes 11 et 12 (Défis 1 et 2) — Génère le rapport structuré des Research Gaps et pistes futures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    high_priority = gaps_df[gaps_df["Priorité_revue"] == "Haute"]

    lines = [
        "# Research Gaps Argumentés et Pistes de Recherche Futures (Étapes 11–12 & Défis 1–2)",
        "",
        f"Synthèse générée automatiquement à partir du corpus de **{summary['total_studies']} études** "
        f"(score QA médian : **{summary['median_qa_score']}/20** ; **{len(gaps_df)} limitations récurrentes** ≥ 3 études, "
        f"dont **{len(high_priority)} de priorité haute**).",
        "",
        "## 1. Limitations récurrentes de priorité haute extraites de `potential_gaps.csv`",
        "",
        "| Limitation récurrente | Études (n) | % corpus | Gap de la revue | Priorité |",
        "| :--- | :---: | :---: | :--- | :---: |",
    ]
    for _, row in high_priority.iterrows():
        lines.append(
            f"| {row['Limitation_Récurrente']} | {row['Nb_études']}/{summary['total_studies']} | "
            f"{row['Pourcentage_corpus']} % | {row['Gap_revue']} | {row['Priorité_revue']} |"
        )

    lines.extend([
        "",
        "## 2. Trois Research Gaps majeurs argumentés (Étape 11 — Défi 1)",
        "",
        "### Gap 1 — Déficit de validation par transfert spatio-temporel et exposition aux fuites de données",
        "- **Description :** 19 études sur 22 (86,4 %) n'évaluent aucun transfert explicite vers un verger, une région ou une année indépendante (`V0`–`V3`), et aucune n'utilise de jeu de test externe indépendant (`V5`). Parallèlement, 14 études sur 22 (63,6 %) présentent au moins une voie potentielle de fuite de données.",
        f"- **Justification empirique :** Sur les {summary['n_accuracy_studies']} études rapportant une précision globale (OA/Accuracy), la performance chute significativement lorsque le niveau de validation augmente (Spearman ρ = {summary['spearman_rho_accuracy_vs_validation']}, p = {summary['spearman_p']} ; médiane {summary['median_accuracy_V0_V2']} en `V0`–`V2` contre {summary['median_accuracy_V3_V4']} en `V3`–`V4`). En `V4`, D'Addabbo et al. (2025) observent un rappel de seulement 0,10 lors du transfert inter-zones, et Garofalo et al. (2024) rapportent l'échec du transfert inter-annuel.",
        "- **Implication :** Les performances mesurées en validation interne surestiment la capacité de généralisation opérationnelle des modèles.",
        "",
        "### Gap 2 — Faiblesse de la vérité terrain biologique et physiologique",
        "- **Description :** 15 études sur 22 (68,2 %) s'appuient sur des annotations visuelles ou des relevés sans confirmation moléculaire (qPCR/PCR) ni mesure physiologique directe (potentiel hydrique de tige).",
        "- **Justification empirique :** Seules 3 études utilisent une confirmation qPCR/PCR et 4 une mesure physiologique ; aucune des 4 études de Deep Learning ne dispose d'une validation biologique en laboratoire. La détection pré-symptomatique n'est démontrée rigoureusement que par Zarco-Tejada et al. (2018).",
        "- **Implication :** Les classifieurs entraînés sur labels visuels modélisent la perception humaine des symptômes tardifs plutôt que l'infection latente.",
        "",
        "### Gap 3 — Confusion physiologique entre infection vasculaire et déficit hydrique",
        "- **Description :** 11 études sur 22 (50,0 % — soit 100 % des études sur pathogènes vasculaires) ne dissocient pas expérimentalement l'infection par *Xylella fastidiosa* ou *Verticillium dahliae* du stress hydrique abiotique.",
        "- **Justification empirique :** Les deux stress induisent une fermeture stomatique et des altérations similaires des indices spectraux et thermiques. Belwalkar et al. (2026) montrent qu'à température de couvert comparable, les indices classiques échouent à séparer les arbres précoces des arbres asymptomatiques (OA = 0,51), nécessitant la fluorescence résolue par raies de Fraunhofer (OA = 0,77).",
        "- **Implication :** En contexte méditerranéen semi-aride, les alertes issues de capteurs thermiques ou multispectraux présentent un risque élevé de faux positifs liés à la sécheresse.",
        "",
        "## 3. Trois pistes de recherche futures opérationnelles (Étape 12 — Défi 2)",
        "",
        "1. **Piste 1 — Benchmark spatio-temporel strict (`leave-orchard-out`, `leave-region-out`, `leave-season-out`) :**",
        "   - *Hypothèse :* L'écart de performance entre split aléatoire et split par blocs spatio-temporels est statistiquement significatif et dépend de la plateforme capteur.",
        "   - *Protocole :* Comparer systématiquement split aléatoire, *leave-region-out* et *leave-season-out* sur un jeu multi-sites ouvert en appliquant prétraitements et rééquilibrage strictement après séparation.",
        "   - *Métriques :* Sensibilité, précision et F1 de la classe infectée à prévalence réelle, AUPRC et écart de généralisation ΔF1.",
        "2. **Piste 2 — Protocole de référence stratifié confirmé par qPCR et chambre à pression :**",
        "   - *Hypothèse :* L'entraînement sur des étiquettes confirmées par qPCR et potentiel hydrique modifie le classement des modèles et améliore la détection pré-symptomatique.",
        "   - *Protocole :* Échantillonnage stratifié incluant des arbres asymptomatiques testés par qPCR, double notation visuelle (κ inter-observateurs) et mesure du potentiel hydrique de tige.",
        "   - *Métriques :* Taux d'erreur des labels visuels, κ inter-juges, sensibilité pré-symptomatique et taux de faux positifs.",
        "3. **Piste 3 — Plan factoriel croisé Infection × Régime d'irrigation avec covariable hydrique satellitaire :**",
        "   - *Hypothèse :* L'intégration du statut hydrique estimé par série temporelle satellitaire comme covariable réduit les faux positifs chez les arbres sains en déficit hydrique.",
        "   - *Protocole :* Dispositif expérimental à 4 modalités (sain/infecté × irrigué/non irrigué) combinant imagerie hyperspectrale/thermique et indices Sentinel-2.",
        "   - *Métriques :* Matrice de confusion multi-classes séparant « sain en stress hydrique » et « infecté », taux de faux positifs sous déficit hydrique.",
    ])

    target_file = out / "research_gaps_et_pistes.md"
    target_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_file


def run(
    data_dir: Path = DATA,
    output_dir: Path = OUT,
    quiet: bool = False,
) -> dict[str, Any]:
    """Exécute le pipeline complet (étapes 2 à 12) et écrit output/synthese_lab9.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(data_dir=data_dir)
    comparative_table(df, output_dir=out)
    perf = performance_stats(df, output_dir=out)
    figures(df, output_dir=out)
    g = gaps(df, output_dir=out)

    q = df.quality_total_out_of_20
    vc = lambda c: {str(k): int(v) for k, v in df[c].value_counts().items()}
    summary: dict[str, Any] = {
        "total_studies": int(len(df)),
        "total_reports": 23,
        "document_type": vc("document_type"),
        "by_approach": vc("ai_family"),
        "by_platform": vc("platform"),
        "by_modality": vc("modality"),
        "by_target": vc("target"),
        "by_target_detail": vc("Cible_détaillée"),
        "by_reference_data": vc("reference_data"),
        "by_early_detection": vc("early_detection"),
        "by_validation": vc("validation_level"),
        "by_deployment": vc("deployment_level"),
        "by_country": vc("country_norm"),
        "by_year": {str(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
        "algorithm_counts_multilabel": dict(Counter(a for r in df.record for a in ALGOS[r]).most_common()),
        "explainability": vc("Explicabilité"),
        "studies_with_leakage_route": int((df.n_leak > 0).sum()),
        "mean_qa_score": round(float(q.mean()), 2),
        "median_qa_score": float(q.median()),
        "qa_iqr": [float(q.quantile(0.25)), float(q.quantile(0.75))],
        "qa_range": [int(q.min()), int(q.max())],
        "qa_ge_10": int((q >= 10).sum()),
        "qa_items_fully_met": {f"Q{i}": int((df[f"Q{i}"] == 2).sum()) for i in range(1, 11)},
        **perf,
        "potential_gaps_count": int(len(g)),
    }
    write_research_gaps_and_pistes(df, g, summary, output_dir=out)
    (out / "synthese_lab9.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse comparative et d'identification des Research Gaps (PRISMA 2020 — Lab 9)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Répertoire contenant les fichiers CSV d'entrée (défaut : data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Répertoire de destination des livrables générés (défaut : output/).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Désactive l'affichage JSON sur la sortie standard.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée principal en ligne de commande."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run(data_dir=args.data_dir, output_dir=args.output_dir, quiet=args.quiet)


if __name__ == "__main__":
    main()
