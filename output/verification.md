# Vérification — Lab 9 : Analyser les études et identifier les Research Gaps (PRISMA 2020)

Tous les chiffres cités proviennent de `output/` (générés par `scripts/analyze_and_gaps.py`) et sont identiques à ceux du manuscrit de la revue.

## 1. Questions de compréhension

### Q1. Pourquoi une limitation qui apparaît dans seulement une étude ne peut-elle pas être considérée comme un research gap ? Illustrez avec un exemple tiré du tableau.
Une limitation isolée décrit un défaut propre à un protocole ou à une équipe, pas une lacune partagée par le domaine. Pour parler de *research gap*, la limitation doit se répéter dans plusieurs études indépendantes : le seuil retenu ici est **au moins trois études** (`potential_gaps.csv`).

* **Exemple tiré du corpus :** Mamalis et al. (2023) ne rapportent leurs métriques YOLOv5 (précision, rappel, mAP) que sous forme de graphique radar, sans valeurs numériques. C'est un défaut de reporting propre à cette étude. Même en le regroupant avec les cas voisins (« Métrique unique ou absente » : Blekos et al. 2021, Kallel et al. 2022, El Atiq et al. 2026), il ne concerne que 3 études, soit 13,6 % du corpus, et la revue le classe en priorité moyenne. À l'inverse, l'absence de test de transfert explicite concerne 19 études sur 22 : c'est un gap de priorité haute.
* **Autre exemple :** une seule étude a appliqué une explication post-hoc (SHAP, Costanza et al. 2025). Ce n'est pas un gap en soi ; le gap est que 10 études n'expliquent aucune prédiction et qu'aucune ne vérifie les attributions contre des mesures physiologiques indépendantes.

---

### Q2. Décrivez la procédure exacte (en code) utilisée pour transformer la colonne `Limitation` en une liste de bigrammes et identifier les limitations récurrentes.
Dans `scripts/analyze_and_gaps.py` :

1. **Construction de `Limitation`.** Chaque étude reçoit une liste de limitations dérivées de l'extraction et des items Q1–Q10 (ex. `Q8 < 2` → « Aucun test de transfert explicite entre sites ou années », `Q3 < 2` → « Référence non confirmée… », voie de fuite identifiée dans le texte intégral → « Voie potentielle de fuite de données »). Les libellés sont joints par ` ; `.
2. **Nettoyage et bigrammes, limitation par limitation** (pour ne pas créer de bigrammes à cheval sur deux limitations) :

```python
def clean_limitation_text(text):
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", str(text).lower())
    return " ".join(t for t in text.split() if t not in STOP_WORDS_FR_EN and len(t) > 2)

def get_ngrams(text, n=2):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

big = Counter(b for t in df["Limitation"] for item in t.split(";")
              for b in get_ngrams(clean_limitation_text(item), 2))
# -> output/limitations_frequentes.csv (top 15)
```

3. **Décompte par étude distincte et seuil ≥ 3** :

```python
lim_to = {}
for _, r in df.iterrows():
    for item in [x.strip() for x in r["Limitation"].split(";") if x.strip()]:
        lim_to.setdefault(item, []).append(r["study"])
rows = [{"Limitation_Récurrente": k, "Nb_études": len(v), ...}
        for k, v in lim_to.items() if len(v) >= 3]
# -> output/potential_gaps.csv, avec la correspondance vers le gap et la priorité de la revue
```

Les bigrammes servent à l'exploration lexicale ; le décompte par étude unique sert à décider ce qui est un gap.

---

### Q3. Comment le graphique des tendances annuelles (`output/tendances_annuelles.png`) peut-il influencer le choix d'une direction de recherche future ? Donnez un exemple concret basé sur vos résultats.
Le graphique (études par année, colorées par plateforme) montre trois périodes : 3 études en 2015–2020, 9 en 2021–2023 et 10 en 2024–2026 (période incomplète). La composition change plus que le volume :

* les 7 études sur le stress hydrique ou la sécheresse sont toutes publiées à partir de 2024 (7 sur 10 contre 0 sur 12 avant ; Fisher p < 0,001, exploratoire) ;
* le satellite représente 7 études sur 10 depuis 2024 contre 3 sur 12 avant (p = 0,08) ;
* le deep learning ne suit pas cette tendance : 3 études en 2021–2023, 1 seule ensuite.

* **Exemple d'orientation :** l'essor récent du satellite pour le stress hydrique ne s'accompagne pas d'une meilleure validation. Le seul modèle qui transfère bien entre zones et saisons (Martínez-Ruedas et al., 2026) utilise 1 628 mesures sur 75 zones et des variables météorologiques, alors qu'un modèle entraîné sur une année n'a rien prédit d'utile l'année suivante (Garofalo et al., 2024). Une direction pertinente est donc l'évaluation *leave-season-out* et *leave-region-out* des modèles satellitaires de potentiel hydrique, et leur usage comme covariable pour séparer stress hydrique et infection, plutôt qu'un nouveau classifieur sur images RGB isolées.

---

## 2. Défis pratiques

### Défi 1 — Transformer une limitation récurrente en research gap argumenté
* **Limitation sélectionnée dans `output/potential_gaps.csv` :** « Aucun test de transfert explicite entre sites ou années » (**19 études sur 22**, 86,4 %), associée à « Voie potentielle de fuite de données » (14 études).
* **Formulation du gap :**
  * **Intitulé :** *Les performances des modèles d'IA pour la santé de l'olivier ne sont presque jamais évaluées sur des vergers, régions ou saisons non vus à l'entraînement.*
  * **Justification :** 8 études sont en V2 (split interne) et 8 en V3 (plusieurs sites ou saisons sans test de transfert), 3 en V0–V1. Les 3 études V4 donnent des résultats opposés : perte négligeable (R² 0,84 → 0,82, Martínez-Ruedas et al., 2026), échec inter-annuel (Garofalo et al., 2024), rappel de 0,10 dans une seconde zone avec une OA trompeuse de 0,92 (D'Addabbo et al., 2025). Sur les 12 études rapportant une OA/accuracy, la valeur baisse quand la validation devient plus exigeante (ρ = −0,70 ; médiane 0,96 en V0–V2 contre 0,80 en V3–V4).
  * **Implication :** les précisions les plus élevées du corpus sont aussi les moins testées ; elles ne permettent pas de décider d'un déploiement.

---

### Défi 2 — Proposer une nouvelle problématique de recherche originale
* **Problématique :** *« Un modèle de détection de* Xylella fastidiosa *fondé sur des traits hyperspectraux et thermiques aéroportés reste-t-il sensible et spécifique lorsqu'il est appliqué à une région, un cultivar et une saison non vus, et lorsque des arbres sains en déficit hydrique sont présents ? »*
* **Méthodologie envisagée :**
  1. **Données et référence :** au moins deux régions et deux saisons ; qPCR sur un sous-ensemble stratifié d'arbres, y compris asymptomatiques ; potentiel hydrique de tige mesuré ; cultivar, irrigation et dates enregistrés pour chaque arbre.
  2. **Plan expérimental :** classes « infecté », « sain irrigué » et « sain en déficit hydrique » ; traits physiologiques (pigments, fluorescence, CWSI) en entrée de modèles classiques (SVM, RF) comparés à un réseau convolutif.
  3. **Validation :** split aléatoire pour référence uniquement, puis *leave-region-out* et *leave-season-out* ; augmentation et rééquilibrage après le split.
* **Métriques :** matrice de confusion ; sensibilité, précision et F1 de la classe infectée à la prévalence du terrain ; taux de faux positifs chez les arbres sains en déficit hydrique ; écart de performance entre split aléatoire et split par région.
