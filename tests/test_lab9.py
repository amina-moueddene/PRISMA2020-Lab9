"""Tests du Lab 9.

Trois groupes :
  * tests unitaires des fonctions de nettoyage et de n-grammes (étapes 3 et 9) ;
  * tests des livrables (étapes 5 à 13) : fichiers présents, structure, seuils ;
  * tests de concordance avec la revue systématique (effectifs, qualité, statistiques).

Lancer : python3 -m unittest discover -s tests -v
"""
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"

_spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "analyze_and_gaps.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


class TestCleaningFunctions(unittest.TestCase):
    """Étapes 3 et 9 : fonctions de nettoyage."""

    def test_clean_text_expands_abbreviations(self):
        self.assertEqual(pipeline.clean_text("  SVM;   RF "), "support vector machine; random forest")
        self.assertEqual(pipeline.clean_text(float("nan")), "")

    def test_clean_limitation_text_removes_stopwords_and_short_tokens(self):
        self.assertEqual(pipeline.clean_limitation_text("Aucun test de transfert entre sites"),
                         "test transfert entre sites")

    def test_get_ngrams(self):
        self.assertEqual(pipeline.get_ngrams("a b c", 2), ["a b", "b c"])
        self.assertEqual(pipeline.get_ngrams("a", 2), [])


class TestInputData(unittest.TestCase):
    """Étape 2 : fichiers d'entrée."""

    def test_input_files_have_22_studies(self):
        for f in ["fulltext_extraction_22_studies.csv", "quality_assessment_Q1-Q10.csv",
                  "studies_master_v2.csv", "lab8_consolidated_extraction_qa.csv"]:
            self.assertEqual(len(pd.read_csv(DATA / f, encoding="utf-8-sig")), 22, f)

    def test_loaded_frame_has_derived_columns(self):
        df = pipeline.load()
        self.assertEqual(len(df), 22)
        for col in ["Cible_détaillée", "Algorithmes", "Explicabilité", "Limitation", "n_leak"]:
            self.assertIn(col, df.columns)
        self.assertFalse(df["Limitation"].eq("").any(), "Chaque étude doit avoir au moins une limitation")


class TestDeliverables(unittest.TestCase):
    """Étapes 5 à 13 : livrables."""

    def test_required_files_exist(self):
        for fname in ["tableau_comparatif.csv", "tableau_comparatif.md", "statistiques_performances.csv",
                      "tendances_annuelles.png", "top_modeles.png", "validation_et_qa.png",
                      "limitations_frequentes.csv", "potential_gaps.csv", "synthese_lab9.json", "verification.md"]:
            fpath = OUT / fname
            self.assertTrue(fpath.exists(), f"Fichier manquant : {fname}")
            self.assertGreater(fpath.stat().st_size, 0, f"Fichier vide : {fname}")

    def test_figures_are_valid_png(self):
        for fname in ["tendances_annuelles.png", "top_modeles.png", "validation_et_qa.png"]:
            head = (OUT / fname).read_bytes()[:8]
            self.assertEqual(head, b"\x89PNG\r\n\x1a\n", fname)
            self.assertGreater((OUT / fname).stat().st_size, 50_000, fname)

    def test_comparative_table(self):
        df = pd.read_csv(OUT / "tableau_comparatif.csv")
        self.assertEqual(len(df), 22)
        self.assertEqual(df["Étude"].nunique(), 22)
        for col in ["Plateforme", "Modalité", "Cible", "Approche", "Validation", "Score_QA", "Limitation"]:
            self.assertIn(col, df.columns)

    def test_performance_stats_are_split_by_metric(self):
        st = pd.read_csv(OUT / "statistiques_performances.csv")
        self.assertTrue({"OA / Accuracy", "R²", "Score QA (/20)"} <= set(st["Type_métrique"]))
        for col in ["moyenne", "écart_type", "médiane"]:
            self.assertIn(col, st.columns)

    def test_bigrams_do_not_cross_limitations(self):
        lim = pd.read_csv(OUT / "limitations_frequentes.csv")
        self.assertEqual(len(lim), 15)
        self.assertNotIn("publics preuve", set(lim["Bigramme_Limitation"]))

    def test_potential_gaps_threshold_and_mapping(self):
        gaps = pd.read_csv(OUT / "potential_gaps.csv")
        self.assertGreaterEqual(len(gaps), 3)
        self.assertTrue((gaps["Nb_études"] >= 3).all())
        self.assertTrue(gaps["Priorité_revue"].isin(["Haute", "Moyenne", "Basse"]).all())
        self.assertGreaterEqual((gaps["Priorité_revue"] == "Haute").sum(), 3)

    def test_verification_answers_all_questions(self):
        txt = (OUT / "verification.md").read_text(encoding="utf-8")
        for part in ["### Q1.", "### Q2.", "### Q3.", "Défi 1", "Défi 2"]:
            self.assertIn(part, txt)

    def test_readme_documents_every_step(self):
        txt = (ROOT / "README.md").read_text(encoding="utf-8")
        for k in range(1, 14):
            self.assertIn(f"| {k} |", txt, f"Étape {k} absente du tableau des étapes du README")
        self.assertIn("Prérequis", txt)


class TestConsistencyWithReview(unittest.TestCase):
    """Les chiffres doivent être identiques à ceux de la revue systématique."""

    @classmethod
    def setUpClass(cls):
        cls.s = json.loads((OUT / "synthese_lab9.json").read_text(encoding="utf-8"))

    def test_counts(self):
        s = self.s
        self.assertEqual(s["total_studies"], 22)
        self.assertEqual(s["by_approach"], {"Classical ML": 18, "Deep learning": 4})
        self.assertEqual(s["by_platform"], {"Satellite": 10, "UAV": 7, "Airborne": 5})
        self.assertEqual(s["by_modality"], {"Multispectral": 11, "Multimodal": 6, "RGB": 3, "Hyperspectral": 2})
        self.assertEqual(s["by_target"]["Vascular pathogen"], 11)
        self.assertEqual(s["by_target"]["Water/drought stress"], 7)
        self.assertEqual(s["by_target_detail"]["Xylella fastidiosa"], 4)
        self.assertEqual(s["by_target_detail"]["Verticillium dahliae"], 5)
        self.assertEqual(s["by_validation"], {"V3": 8, "V2": 8, "V4": 3, "V1": 2, "V0": 1})
        self.assertEqual(s["by_deployment"], {"DRL2": 11, "DRL3": 8, "DRL4": 3})
        self.assertEqual(s["by_reference_data"]["Laboratory (qPCR/PCR)"], 3)
        self.assertEqual(s["by_early_detection"]["Pre-symptomatic"], 1)
        algo = s["algorithm_counts_multilabel"]
        self.assertEqual((algo["SVM"], algo["Random Forest"], algo["Boosting"],
                          algo["Analyse discriminante"], algo["Processus gaussien"]), (12, 8, 6, 4, 2))
        self.assertEqual(s["studies_with_leakage_route"], 14)
        self.assertEqual(s["explainability"]["Aucune"], 10)

    def test_quality_and_statistics(self):
        s = self.s
        self.assertEqual((s["median_qa_score"], s["qa_iqr"], s["qa_range"]), (10.5, [9.0, 14.0], [4, 18]))
        self.assertEqual(s["qa_ge_10"], 12)
        self.assertEqual(s["n_accuracy_studies"], 12)
        self.assertAlmostEqual(s["spearman_rho_accuracy_vs_validation"], -0.70, places=2)
        self.assertEqual((s["median_accuracy_V0_V2"], s["median_accuracy_V3_V4"]), (0.96, 0.80))
        self.assertEqual((s["median_qa_classical_ml"], s["median_qa_deep_learning"]), (12.0, 8.5))


if __name__ == "__main__":
    unittest.main()
