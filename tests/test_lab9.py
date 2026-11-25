import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


class TestLab9Deliverables(unittest.TestCase):
    def test_required_files_exist(self):
        expected = [
            "tableau_comparatif.csv",
            "tableau_comparatif.md",
            "statistiques_performances.csv",
            "tendances_annuelles.png",
            "top_modeles.png",
            "limitations_frequentes.csv",
            "potential_gaps.csv",
            "synthese_lab9.json",
            "verification.md",
        ]
        for fname in expected:
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

    def test_summary_json_consistency(self):
        summary = json.loads((OUT / "synthese_lab9.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["total_studies"], 22)
        self.assertEqual(sum(summary["by_approach"].values()), 22)


if __name__ == "__main__":
    unittest.main()
