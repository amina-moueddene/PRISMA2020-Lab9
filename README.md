# Lab 9 : Analyser les études et identifier les Research Gaps (PRISMA 2020)

**Sujet de la revue systématique :** *Machine learning and remote sensing for disease and stress detection in olive* (22 études incluses issues du Lab 8).

---

## 1. Structure du dépôt

```text
PRISMA2020-Lab9-main/
├── README.md                              # Documentation complète et synthèse des 13 étapes du Lab 9
├── data/                                  # Grilles d'extraction et d'évaluation du Lab 8 (22 études)
│   ├── fulltext_extraction_22_studies.csv
│   ├── lab8_consolidated_extraction_qa.csv
│   ├── quality_assessment_Q1-Q10.csv
│   └── studies_master_v2.csv
├── scripts/
│   └── analyze_and_gaps.py                # Pipeline complet (nettoyage, classification, stats, graphiques, n-grammes, gaps)
├── output/
│   ├── tableau_comparatif.csv             # Tableau synthétique automatisé des 22 études (Étape 5)
│   ├── tableau_comparatif.md              # Version Markdown du tableau comparatif (Étape 5)
│   ├── statistiques_performances.csv      # Moyenne, écart-type et médiane par approche (Étape 6)
│   ├── tendances_annuelles.png            # Évolution temporelle des publications 2015–2026 (Étape 7)
│   ├── top_modeles.png                    # Répartition des familles de modèles utilisés (Étape 8)
│   ├── limitations_frequentes.csv         # Top 15 des bigrammes extraits des limitations (Étape 9)
│   ├── potential_gaps.csv                 # Limitations récurrentes apparaissant dans ≥ 3 études (Étape 10)
│   ├── synthese_lab9.json                 # Résumé statistique structuré pour audit automatisé
│   └── verification.md                    # Réponses aux questions de compréhension et défis pratiques (Étape 13)
└── tests/
    └── test_lab9.py                       # Suite de tests unitaires vérifiant l'intégrité des livrables
```

---

## 2. Exécution du pipeline et des tests

```bash
python3 scripts/analyze_and_gaps.py
python3 -m unittest discover -s tests -v
```

---

## 3. Synthèse Quantitative du Corpus (Étapes 2 à 8)

* **Taille du corpus :** 22 études incluses (2015–2026) ; score de qualité QA moyen de **11,14 / 20** (médiane : **10,5 / 20**).
* **Répartition par approche :**
  * **Machine Learning Classique :** 18 études (81,8 %) — dominé par **SVM** (9 études), **Gradient Boosting / XGBoost / GBM** (5 études), **Analyse Discriminante LDA/QDA** (4 études) et **Random Forest** (4 études).
  * **Deep Learning :** 4 études (18,2 %) — **CNN / ResNet / MobileNet** (3 études) et **YOLOv5** (1 étude).
* **Répartition par modalité capteur :** Multimodal / Fusion (9 études), Multispectral (8 études), RGB (3 études), Hyperspectral (2 études).
* **Répartition par domaine cible :** Stress hydrique & abiotique (7 études), *Verticillium dahliae* (6 études), *Xylella fastidiosa* (4 études), Multi-pathogènes *Xylella + Verticillium* (2 études), Maladies foliaires (2 études), Stress mixte (1 étude).
* **Niveaux de validation (`V0`–`V4`) :** `V3` (8 études), `V2` (8 études), `V4` (3 études), `V1` (2 études), `V0` (1 étude).

---

## 4. Research Gaps Argumentés (Étape 11)

### Gap 1 — Déficit critique de généralisation spatiale et temporelle (`V4`) et chute des performances hors domaine
* **Description :** 86,4 % des études (19/22) n'évaluent aucun transfert spatial inter-régions ou temporel inter-années indépendant (`V0`–`V3`), et les rares études testant un transfert réel (`V4`) observent un effondrement des performances hors domaine.
* **Justification :** Calderón et al. (2015), Zarco-Tejada et al. (2018), Di Nisio et al. (2020), Poblete et al. (2021, 2023) et Ksibi et al. (2022) restent limités à des validations internes ou groupées (`V2`–`V3`). Parmi les études `V4`, **Garofalo et al. (2024)** rapportent l'échec du transfert inter-annuel et **D'Addabbo et al. (2025)** montrent une chute du rappel à **10 %** lors du transfert inter-zones (Gorgognolo $\rightarrow$ Polignano).
* **Implication :** Les modèles entraînés sur un verger, un cultivar ou une campagne spécifique sur-apprennent les conditions locales et ne peuvent pas être déployés directement dans d'autres bassins oléicoles.

### Gap 2 — Faible maturité opérationnelle (`DRL2`–`DRL3`) et quasi-absence de reproductibilité ouverte
* **Description :** La quasi-totalité des travaux reste au stade de prototypes expérimentaux hors-ligne sans partage de code ni explicabilité agronomique.
* **Justification :** **20 études sur 22 (90,9 %)** ne publient pas leur code source complet (seuls Zarco-Tejada et al. 2018 fournissent des scripts R sur GitHub et Martínez-Ruedas et al. 2026 leurs données sur GitLab) ; **19 études sur 22 (86,4 %)** plafonnent aux niveaux `DRL2`–`DRL3` et **9 études** n'offrent aucun mécanisme d'explicabilité des prédictions.
* **Implication :** L'impossibilité de reproduire les prétraitements spectraux et les protocoles de séparation empêche tout benchmark équitable et freine l'adoption par les services phytosanitaires.

### Gap 3 — Sous-exploitation du Deep Learning multimodal et rareté de la détection pré-symptomatique validée en laboratoire
* **Description :** Les architectures Deep Learning sont cantonnées à l'imagerie RGB/multispectrale UAV au stade symptomatique sur labels visuels, tandis que la détection pré-symptomatique validée par qPCR reste quasi absente.
* **Justification :** Les 4 études Deep Learning (**Blekos et al. 2021**, **Ksibi et al. 2022**, **Mamalis et al. 2023**, **Raouhi et al. 2024**) reposent uniquement sur des capteurs RGB/multispectraux UAV, une vérité terrain visuelle (0 PCR/qPCR) et des symptômes déjà visibles. À l'inverse, seules **Zarco-Tejada et al. (2018)** et **Poblete et al. (2021)** démontrent une détection pré-symptomatique/précoce ancrée biologiquement.
* **Implication :** Détecter *Xylella fastidiosa* ou *Verticillium dahliae* uniquement après l'apparition de symptômes foliaires ne permet pas d'endiguer la dissémination épidémique dans les vergers.

---

## 5. Pistes de Recherche Futures (Étape 12)

1. **Piste 1 — Adaptation de domaine guidée par la physique et benchmark spatio-temporel strict (`V4`) :**
   * *Gap ciblé :* Gap 1 (échec de généralisation inter-zones et inter-années).
   * *Hypothèse :* Le couplage d'une inversion de transfert radiatif (PROSAIL/DART) avec une adaptation de domaine non supervisée stabilisera le rappel au-dessus de 75 % lors du transfert vers des vergers et années inédits.
   * *Méthodologie :* Construire un jeu de données multi-régions (Espagne, Italie, Maroc) avec protocole *leave-one-region-out* et *leave-one-year-out*, comparant XGBoost sur traits biophysiques et un Transformer spatio-temporel.
   * *Métriques :* Recall sur classe infectée, F1-score macro, Kappa de Cohen et écart de généralisation $\Delta\text{Recall}_{\text{inter-zones}}$.
2. **Piste 2 — Pipeline open-source reproductible couplant Sentinel-2/UAV et explicabilité SHAP (`DRL4`–`DRL5`) :**
   * *Gap ciblé :* Gap 2 (faible reproductibilité et manque d'interprétabilité).
   * *Hypothèse :* Un pipeline conteneurisé open-source associant séries temporelles Sentinel-2 (Google Earth Engine) et explications locales SHAP augmentera la confiance des agronomes et permettra une reproductibilité intégrale.
   * *Méthodologie :* Publier sous licence libre le code Python, les splits normalisés et un modèle hybride (indices red-edge/thermiques + XGBoost/CNN léger) accompagné de cartes d'attribution SHAP par arbre.
   * *Métriques :* Taux de reproduction par des tiers, latence d'inférence par hectare, F1-score et cohérence biophysique des scores SHAP.
3. **Piste 3 — Fusion multimodale Deep Learning (optique + thermique + SAR) pour la détection pré-symptomatique ancrée par qPCR :**
   * *Gap ciblé :* Gap 3 (absence de DL multimodal pré-symptomatique validé biologiquement).
   * *Hypothèse :* Une architecture Deep Learning à attention croisée fusionnant imagerie multispectrale/thermique UAV et séries temporelles Sentinel-1/2 détectera les infections vasculaires au stade asymptomatique.
   * *Méthodologie :* Mettre en place un suivi longitudinal avec tests qPCR répétés sur arbres asymptomatiques, entraîner un réseau multimodal (attention croisée optique-thermique-SAR) et le comparer à une baseline SVM/RF.
   * *Métriques :* Sensibilité pré-symptomatique ($t - 3$ mois avant symptômes), taux de faux positifs (FP), AUROC et gain moyen d'anticipation en jours.
