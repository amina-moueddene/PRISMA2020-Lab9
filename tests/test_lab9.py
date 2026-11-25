import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"


class TestLab9Deliverables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = json.loads((OUT / "synthese_lab9.json").read_text(encoding="utf-8"))

    def test_required_files_exist(self):
        for fname in ["tableau_comparatif.csv", "tableau_comparatif.md", "statistiques_performances.csv",
                      "tendances_annuelles.png", "top_modeles.png", "validation_et_qa.png",
                      "limitations_frequentes.csv", "potential_gaps.csv", "synthese_lab9.json", "verification.md"]:
            fpath = OUT / fname
            self.assertTrue(fpath.exists(), f"Fichier manquant : {fname}")
            self.assertGreater(fpath.stat().st_size, 0, f"Fichier vide : {fname}")

    def test_comparative_table_has_22_studies(self):
        df = pd.read_csv(OUT / "tableau_comparatif.csv")
        self.assertEqual(len(df), 22)
        self.assertEqual(df["Étude"].nunique(), 22)

    def test_potential_gaps_threshold(self):
        gaps = pd.read_csv(OUT / "potential_gaps.csv")
        self.assertGreaterEqual(len(gaps), 3)
        self.assertTrue((gaps["Nb_études"] >= 3).all())

    def test_counts_match_systematic_review(self):
        """Effectifs identiques à ceux du manuscrit de la revue (main_v2.tex)."""
        s = self.s
        self.assertEqual(s["total_studies"], 22)
        self.assertEqual(s["by_approach"], {"Classical ML": 18, "Deep learning": 4})
        self.assertEqual(s["by_platform"], {"Satellite": 10, "UAV": 7, "Airborne": 5})
        self.assertEqual(s["by_modality"], {"Multispectral": 11, "Multimodal": 6, "RGB": 3, "Hyperspectral": 2})
        self.assertEqual(s["by_target"]["Vascular pathogen"], 11)
        self.assertEqual(s["by_target"]["Water/drought stress"], 7)
        self.assertEqual(s["by_target_detail"]["Xylella fastidiosa"], 4)
        self.assertEqual(s["by_target_detail"]["Verticillium dahliae"], 5)
        self.assertEqual(s["by_target_detail"]["Xylella + Verticillium"], 2)
        self.assertEqual(s["by_validation"], {"V3": 8, "V2": 8, "V4": 3, "V1": 2, "V0": 1})
        self.assertEqual(s["by_deployment"], {"DRL2": 11, "DRL3": 8, "DRL4": 3})
        self.assertEqual(s["by_reference_data"]["Laboratory (qPCR/PCR)"], 3)
        self.assertEqual(s["by_reference_data"]["Physiological measurement"], 4)
        self.assertEqual(s["by_early_detection"]["Pre-symptomatic"], 1)
        algo = s["algorithm_counts_multilabel"]
        self.assertEqual((algo["SVM"], algo["Random Forest"], algo["Boosting"],
                          algo["Analyse discriminante"], algo["Processus gaussien"]), (12, 8, 6, 4, 2))
        self.assertEqual(s["studies_with_leakage_route"], 14)
        self.assertEqual(s["explainability"]["Aucune"], 10)

    def test_quality_and_performance_match_review(self):
        s = self.s
        self.assertEqual(s["median_qa_score"], 10.5)
        self.assertEqual(s["qa_iqr"], [9.0, 14.0])
        self.assertEqual(s["qa_range"], [4, 18])
        self.assertEqual(s["qa_ge_10"], 12)
        self.assertEqual(s["n_accuracy_studies"], 12)
        self.assertAlmostEqual(s["spearman_rho_accuracy_vs_validation"], -0.70, places=2)
        self.assertEqual((s["median_qa_classical_ml"], s["median_qa_deep_learning"]), (12.0, 8.5))


if __name__ == "__main__":
    unittest.main()
