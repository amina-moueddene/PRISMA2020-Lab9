# Lab 9 : Analyser les études et identifier les Research Gaps (PRISMA 2020)

**Revue systématique :** *Machine learning and remote sensing for disease and stress detection in olive (Olea europaea L.)*
**Corpus :** 22 études (23 rapports) incluses après évaluation du texte intégral (Scopus et Web of Science Core Collection, recherche du 15 novembre 2026, période 2013–2026).

---

## 1. Introduction

Ce lab transforme la grille d'extraction et l'évaluation de qualité des 22 études incluses (Lab 8) en une analyse comparative : tableau de synthèse, statistiques de performance, tendances temporelles, répartition des algorithmes, limitations récurrentes, puis research gaps argumentés et pistes de recherche.

Toutes les données proviennent des fichiers de la revue (S3 extraction, S4 qualité, S5 données codées). Le script recalcule chaque chiffre, et les tests vérifient qu'ils sont identiques à ceux du manuscrit de la revue.

**Objectifs du lab :**
1. classer les études par approche (ML classique / deep learning), modalité, plateforme et cible ;
2. comparer les performances rapportées sans les pooler (tâches et métriques différentes) ;
3. distinguer une limitation isolée d'un research gap (limitation présente dans au moins 3 études) ;
4. formuler des gaps argumentés et des pistes de recherche testables.

## 2. Prérequis

* **Python ≥ 3.10** (testé avec Python 3.13).
* Bibliothèques listées dans `requirements.txt` : `pandas`, `numpy`, `scipy`, `matplotlib`.
* Aucun accès réseau n'est nécessaire : toutes les données sont dans `data/`.

## 3. Installation et exécution

```bash
git clone https://github.com/amina-moueddene/PRISMA2020-Lab9.git
cd PRISMA2020-Lab9
python3 -m venv .venv && source .venv/bin/activate      # optionnel
pip install -r requirements.txt
python3 scripts/analyze_and_gaps.py                     # génère tous les fichiers de output/ (quelques secondes)
python3 -m unittest discover -s tests -v                # 15 tests
```

Le script est déterministe : deux exécutions produisent les mêmes fichiers.

## 4. Structure du dépôt

```text
PRISMA2020-Lab9/
├── README.md
├── requirements.txt
├── data/                                      # Entrées (identiques aux fichiers de la revue)
│   ├── fulltext_extraction_22_studies.csv     # S3 : extraction du texte intégral (28 variables)
│   ├── quality_assessment_Q1-Q10.csv          # S4 : scores Q1–Q10 (0, 1 ou 2 ; total /20)
│   ├── studies_master_v2.csv                  # S5 : variables codées (plateforme, cible, V, DRL, performance)
│   └── lab8_consolidated_extraction_qa.csv    # S3 + S4 + plateforme normalisée (sortie du Lab 8)
├── scripts/
│   └── analyze_and_gaps.py                    # Pipeline complet, une fonction par groupe d'étapes
├── output/                                    # Sorties générées (voir § 6)
└── tests/
    └── test_lab9.py                           # Tests unitaires, tests des livrables, tests de concordance
```

### Dictionnaire des principales variables d'entrée

| Variable | Fichier | Signification |
| :-- | :-- | :-- |
| `record` | tous | Identifiant de la notice dans le registre de sélection (clé de jointure) |
| `platform`, `modality` | S5 | Plateforme (Satellite, UAV, Airborne) et modalité (Multispectral, Multimodal, RGB, Hyperspectral) |
| `target` | S5 | Cible (Vascular pathogen, Water/drought stress, Foliar disease, Mixed biotic/abiotic stress) |
| `ai_family` | S5 | Famille d'IA (Classical ML, Deep learning) |
| `reference_data` | S5 | Type de vérité terrain (qPCR/PCR, mesure physiologique, relevé visuel…) |
| `validation_level` | S3/S5 | Rigueur de validation V0–V5 (V4 = test de transfert explicite entre sites ou années) |
| `deployment_level` | S3/S5 | Maturité de déploiement DRL1–DRL5 |
| `reported_performance`, `metric` | S5 | Performance principale telle que publiée et sa métrique (OA, R², AUROC…) |
| `Q1`…`Q10`, `quality_total_out_of_20` | S4 | Items de qualité et score total |
| `algorithms`, `transfer_test`, `explainability`, `data_code_availability` | S3 | Champs libres de l'extraction |

## 5. Correspondance entre les étapes du lab et le code

| Étape | Contenu | Où | Sortie |
| :-: | :-- | :-- | :-- |
| 1 | Introduction et prérequis | `README.md` §§ 1–3 | — |
| 2 | Chargement du fichier d'extraction | `load()` | DataFrame de 22 études |
| 3 | Nettoyage et structuration | `load()`, `clean_text()` | colonnes normalisées |
| 4 | Classification par méthode, modalité, plateforme et cible | `load()` (tables `TARGET_DETAIL`, `ALGOS`, `XAI`, `LEAKAGE`) | colonnes `Cible_détaillée`, `Algorithmes`, `Explicabilité` |
| 5 | Tableau comparatif | `comparative_table()` | `tableau_comparatif.csv`, `.md` |
| 6 | Analyse des métriques et performances | `performance_stats()` | `statistiques_performances.csv` |
| 7 | Tendances temporelles | `figures()` | `tendances_annuelles.png` |
| 8 | Répartition des modèles | `figures()` | `top_modeles.png`, `validation_et_qa.png` |
| 9 | Limitations récurrentes (n-grammes) | `clean_limitation_text()`, `get_ngrams()`, `gaps()` | `limitations_frequentes.csv` |
| 10 | Limitation individuelle vs gap global (≥ 3 études) | `gaps()` | `potential_gaps.csv` |
| 11 | Research gaps argumentés | `README.md` § 8, `verification.md` | — |
| 12 | Pistes de recherche futures | `README.md` § 9, `verification.md` (Défi 2) | — |
| 13 | Vérification (questions et défis) | `output/verification.md` | — |

**Choix méthodologiques documentés dans le code :**
* les performances ne sont pas moyennées toutes métriques confondues : les statistiques sont séparées par type de métrique, par famille d'IA et par groupe de validation ;
* les algorithmes sont comptés en multi-label (modèle principal ou comparateur), comme dans la revue ;
* les limitations sont dérivées de critères explicites (ex. `Q8 < 2` → pas de test de transfert ; voie de fuite identifiée dans le texte intégral) ; chaque gap retenu est relié au gap et à la priorité correspondants de la revue.

## 6. Description des sorties

| Fichier | Contenu | Comment le lire |
| :-- | :-- | :-- |
| `tableau_comparatif.csv` / `.md` | Une ligne par étude : plateforme, modalité, cible, approche, algorithmes, référence, V, DRL, QA, résultat principal, limitations | Les valeurs de performance ne se comparent pas d'une ligne à l'autre |
| `statistiques_performances.csv` | n, moyenne, écart-type, médiane, min, max par regroupement et type de métrique ; scores QA par approche | Descriptif ; la revue ne poole pas les performances |
| `tendances_annuelles.png` | Études par année (2013–2026) et plateforme | Aucune étude incluse avant 2015 ; 2026 incomplète |
| `top_modeles.png` | Nombre d'études ayant évalué chaque algorithme | Comptage multi-label |
| `validation_et_qa.png` | (A) niveaux V0–V5 par famille d'IA ; (B) histogramme des scores QA | Médiane 10,5/20 |
| `limitations_frequentes.csv` | 15 bigrammes les plus fréquents | Exploration lexicale |
| `potential_gaps.csv` | Limitations présentes dans ≥ 3 études, avec pourcentage, gap et priorité de la revue, études concernées | Base des gaps de l'étape 11 |
| `synthese_lab9.json` | Tous les effectifs et statistiques clés | Utilisé par les tests |
| `verification.md` | Réponses aux questions Q1–Q3 et défis 1–2 | — |

### Tests (`tests/test_lab9.py`)

* **Fonctions** : `clean_text`, `clean_limitation_text`, `get_ngrams`.
* **Entrées** : 22 études dans chaque fichier ; colonnes dérivées présentes ; chaque étude a au moins une limitation.
* **Livrables** : présence et validité des fichiers et figures, structure du tableau et des statistiques, bigrammes calculés limitation par limitation, seuil ≥ 3 et priorités des gaps, réponses complètes dans `verification.md`, étapes 1–13 documentées dans le README.
* **Concordance avec la revue** : effectifs (plateformes, modalités, cibles, familles, V, DRL, références, algorithmes), scores qualité, ρ de Spearman et médianes de précision.

## 7. Synthèse quantitative du corpus (Étapes 2 à 8)

* **Corpus :** 22 études publiées entre 2015 et 2026 (3 en 2015–2020, 9 en 2021–2023, 10 en 2024–2026, période incomplète) ; 19 articles de revue et 3 communications de conférence. Une étude est décrite dans deux rapports (Kallel et al. 2022 = Makhloufi et al. 2022), comptée une fois.
* **Pays de la zone d'étude :** Italie 7, Grèce 4, Maroc 4, Espagne 2, Tunisie 1, multi-pays 2, Australie 1, Arabie saoudite 1.
* **Plateformes :** satellite 10, drone (UAV) 7, avion piloté 5.
* **Modalités :** multispectral 11, multimodal 6 (dont 4 hyperspectral + thermique), RGB 3, hyperspectral seul 2 ; aucune étude en thermique seul ni LiDAR.
* **Cibles :** pathogènes vasculaires 11 (*Xylella fastidiosa* 4, *Verticillium dahliae* 5, les deux 2), stress hydrique ou sécheresse 7, maladies foliaires 2, stress mixte biotique/abiotique 2.
* **Familles d'IA :** ML classique 18, deep learning 4 (CNN ×3, YOLOv5 ×1), tous sur imagerie UAV ; aucun modèle de fondation ni transformer. Algorithmes évalués (multi-label) : SVM/SVR 12, Random Forest 8, boosting 6, analyse discriminante 4, processus gaussien 2.
* **Données de référence :** qPCR/PCR 3, mesure physiologique 4, relevé visuel ou terrain 9, annotation d'images 3, ROI non décrites 2, données synthétiques 1.
* **Validation (V0–V5) :** V0 1, V1 2, V2 8, V3 8, V4 3, V5 0. **Déploiement (DRL1–5) :** DRL2 11, DRL3 8, DRL4 3, DRL5 0.
* **Qualité Q1–Q10 :** médiane 10,5/20 (IQR 9–14, étendue 4–18, moyenne 11,14) ; 12 études ≥ 10/20. Médiane 12 pour le ML classique contre 8,5 pour le deep learning (Mann–Whitney p = 0,09).
* **Performance et validation :** sur les 12 études rapportant une OA/accuracy, la valeur diminue quand la validation devient plus exigeante (Spearman ρ = −0,70, p = 0,011 ; médiane 0,96 en V0–V2 contre 0,80 en V3–V4). Les performances ne sont pas poolées dans la revue : les statistiques de l'Étape 6 sont descriptives et séparées par type de métrique.

---

## 8. Research gaps argumentés (Étape 11)

Les trois gaps retenus correspondent aux gaps de **priorité haute** de la revue (Section 6, Tableau des gaps).

### Gap 1 — Peu de tests de transfert et validation exposée aux fuites de données
* **Constat :** seulement 3 études sur 22 testent explicitement le transfert vers d'autres sites ou années (V4) et aucune n'utilise de jeu de données externe indépendant (V5). Au moins une voie potentielle de fuite est identifiée dans 14 études (split aléatoire de pixels ou de mesures répétées, augmentation avant le split, classe confondue avec le site) ; son effet n'est quantifié que dans une étude.
* **Justification :** les trois tests de transfert donnent des résultats opposés : perte négligeable pour Martínez-Ruedas et al. (2026) (R² 0,84 → 0,84 en hold-out de zones, 0,82 en hold-out de saison), absence de prédiction utile d'une année à l'autre pour Garofalo et al. (2024), et 2 arbres infectés détectés sur 20 (rappel 0,10) dans une seconde zone pour D'Addabbo et al. (2025), alors que l'OA restait à 0,92 du fait du déséquilibre des classes.
* **Implication :** les précisions internes ne disent pas comment un modèle se comporte dans un autre verger, une autre région ou une autre saison.

### Gap 2 — Données de référence faibles
* **Constat :** une confirmation de laboratoire (qPCR/PCR) ou une mesure physiologique n'existe que dans 7 études sur 22, et dans aucune des 4 études de deep learning.
* **Justification :** les 15 autres études reposent sur des relevés visuels (9), des annotations d'images sans protocole décrit (3), des ROI dont la collecte n'est pas décrite (2) ou des données synthétiques (1). La détection pré-symptomatique n'est démontrée qu'une fois (Zarco-Tejada et al., 2018, confirmation qPCR sur un sous-ensemble).
* **Implication :** les modèles apprennent des symptômes visibles ou le jugement de l'annotateur plutôt que l'infection elle-même.

### Gap 3 — Infection confondue avec le statut hydrique
* **Constat :** aucune étude ne croise statut d'infection et régime d'irrigation dans un plan factoriel ; une seule des 11 études sur pathogènes vasculaires discrimine deux causes (*Xylella* vs *Verticillium*, Poblete et al., 2021) et une seule contrôle le statut hydrique (Belwalkar et al., 2026).
* **Justification :** l'infection vasculaire et le déficit hydrique agissent par la même fermeture stomatique et modifient les mêmes signaux thermiques et de fluorescence. Chez Belwalkar et al. (2026), à température de couvert comparable, la bande O₂-A à résolution standard ne sépare plus les arbres précoces des asymptomatiques (OA 0,51), alors que les raies de Fraunhofer mesurées en résolution subnanométrique le font (OA 0,77).
* **Implication :** un signal positif doit être lu comme un stress de cause indéterminée, à confirmer sur le terrain et au laboratoire.

---

## 9. Pistes de recherche futures (Étape 12)

1. **Piste 1 — Évaluation par blocs spatio-temporels (cible : Gap 1).**
   * *Hypothèse :* l'écart entre performance en split aléatoire et en split par verger/région/saison est significatif et varie selon la plateforme.
   * *Méthodologie :* rejouer sur des données ouvertes les splits aléatoire, *leave-orchard-out*, *leave-region-out* et *leave-season-out* ; augmentation et sur-échantillonnage appliqués après le split ; un arbre dans un seul split.
   * *Métriques :* sensibilité, précision et F1 de la classe infectée à la prévalence du terrain, aire sous la courbe précision-rappel, écart de performance entre splits.
2. **Piste 2 — Référence stratifiée confirmée (cible : Gap 2).**
   * *Hypothèse :* entraîner sur des labels confirmés par qPCR ou par mesure physiologique change le classement des modèles par rapport aux labels visuels.
   * *Méthodologie :* qPCR sur un sous-ensemble stratifié d'arbres (dont asymptomatiques), notation visuelle par plusieurs observateurs avec accord inter-juges, potentiel hydrique de tige à la chambre à pression.
   * *Métriques :* taux d'erreur des labels visuels, κ inter-observateurs, sensibilité pré-symptomatique.
3. **Piste 3 — Plan factoriel infection × irrigation (cible : Gap 3).**
   * *Hypothèse :* des arbres sains en déficit hydrique sont classés « infectés » par les modèles thermiques/fluorescence ; l'ajout du potentiel hydrique estimé par satellite comme covariable réduit ces faux positifs.
   * *Méthodologie :* arbres sains/infectés × irrigués/non irrigués, acquisitions aéroportées hyperspectrales et thermiques, classe « sain en stress hydrique » séparée.
   * *Métriques :* matrice de confusion à trois classes, taux de faux positifs chez les arbres sains stressés.

Ces trois pistes alimentent la proposition de benchmark communautaire (OliveBench) de la revue.
