# Vérification — Lab 9 : Analyser les études et identifier les Research Gaps (PRISMA 2020)

## 1. Questions de Compréhension

### Q1. Pourquoi une limitation qui apparaît dans seulement une étude ne peut-elle pas être considérée comme un research gap ? Illustrez avec un exemple tiré du tableau.
Une limitation observée dans une seule étude reflète une **contrainte expérimentale ponctuelle** propre au protocole ou au dispositif d'une équipe particulière, et non un verrou scientifique structurel partagé par la communauté. Pour constituer un *research gap* global, une lacune doit converger à travers **au moins trois études indépendantes** issues de groupes différents.

* **Exemple concret tiré du corpus :** Dans **Mamalis et al. (2023)**, les auteurs présentent les métriques de détection YOLOv5 (Precision, Recall, mAP@0.5) uniquement sous la forme d'un graphique radar sans fournir de valeurs numériques exactes en tableau (`fitness = 0.65`). De même, **Kallel et al. (2022)** rapportent uniquement une évaluation qualitative sans métrique quantitative (`V0`). Ces deux cas relèvent de défauts ponctuels de reporting individuel (`Métriques quantitatives incomplètes` : 2 études seulement) et non d'un gap méthodologique global du domaine.

---

### Q2. Décrivez la procédure exacte (en code) utilisée pour transformer la colonne `Limitation` en une liste de bigrammes et identifier les limitations récurrentes.
Dans `scripts/analyze_and_gaps.py`, la procédure s'articule en quatre étapes déterministes :

```python
def clean_limitation_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Suppression des caractères spéciaux et conservation des lettres accentuées
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", text)
    # Tokenisation et filtrage des mots vides (stopwords FR/EN) et tokens courts
    tokens = [t for t in text.split() if t not in STOP_WORDS_FR_EN and len(t) > 2]
    return " ".join(tokens)

def get_ngrams(text, n=2):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# 1. Extraction de tous les bigrammes du corpus -> output/limitations_frequentes.csv
all_bigrams = []
for txt in df["Limitation_clean"]:
    all_bigrams.extend(get_ngrams(txt, n=2))
bigram_counts = Counter(all_bigrams)

# 2. Agrégation par étude unique et filtrage Nb_études >= 3 -> output/potential_gaps.csv
lim_to_studies = {}
for _, r in df.iterrows():
    for item in [x.strip() for x in r["Limitation"].split(";") if x.strip()]:
        lim_to_studies.setdefault(item, []).append(r["study"])

potential_gaps_df = pd.DataFrame([
    {
        "Limitation_Récurrente": k,
        "Nb_études": len(v),
        "Pourcentage_corpus": round(100.0 * len(v) / len(df), 1),
        "Études_concernées": ", ".join(v),
    }
    for k, v in lim_to_studies.items() if len(v) >= 3
])
```

Cette double approche combine l'exploration lexicale par bigrammes (`output/limitations_frequentes.csv`) et le décompte exact par étude unique (`output/potential_gaps.csv`, seuil $\ge 3$ études).

---

### Q3. Comment le graphique des tendances annuelles (`output/tendances_annuelles.png`) peut-il influencer le choix d'une direction de recherche future ? Donnez un exemple concret basé sur vos résultats.
Le graphique `output/tendances_annuelles.png` met en évidence trois phases distinctes sur les 22 études incluses (2015–2026) :
1. **2015–2020 (3 études) :** Phase pionnière dominée par des campagnes aéroportées hyperspectrales/thermiques ciblant *Verticillium* et *Xylella* (Calderón et al. 2015 ; Zarco-Tejada et al. 2018 ; Di Nisio et al. 2020).
2. **2021–2023 (9 études, soit 3/an) :** Essor des acquisitions par drone (UAV) et apparition des premiers modèles Deep Learning (CNN, YOLOv5).
3. **2024–2026 (10 études, dont 4 en 2025 et 4 en 2026) :** Accélération nette portée par la télédétection satellitaire multi-temporelle (Sentinel-2, PlanetScope, fusion SAR + optique + météo) appliquée au suivi du stress hydrique et de la sécheresse.

* **Exemple concret d'orientation :** L'explosion des études satellitaires en 2025–2026 montre que la disponibilité des séries temporelles spatiales n'est plus le facteur limitant ; en revanche, la transposition de ces modèles vers de nouvelles saisons ou régions échoue encore (Garofalo et al. 2024 ; D'Addabbo et al. 2025). Il est donc plus pertinent d'orienter une thèse vers **l'adaptation de domaine spatio-temporelle et la fusion multi-échelle Satellite + UAV** plutôt que vers un énième classifieur RGB sur images isolées.

---

## 2. Défis Pratiques

### Défi 1 — Transformer une limitation récurrente en Research Gap argumenté
* **Limitation sélectionnée dans `output/potential_gaps.csv` :** *« Absence de transfert spatial ou temporel indépendant »* (**19 études sur 22**, soit **86,4 %** du corpus).
* **Formulation du Research Gap :**
  * **Intitulé :** *Déficit de généralisation spatiale et temporelle hors domaine dans les modèles d'IA appliqués à la télédétection de l'olivier.*
  * **Justification :** Dix-neuf études (dont Calderón et al. 2015 ; Zarco-Tejada et al. 2018 ; Di Nisio et al. 2020 ; Poblete et al. 2021, 2023 ; Ksibi et al. 2022 ; Raouhi et al. 2024 ; Belwalkar et al. 2026 ; El Atiq et al. 2026) évaluent leurs modèles uniquement via validation croisée interne ou découpage aléatoire sur un même site/campagne (`V0`–`V3`). Parmi les trois seules études réalisant un test `V4`, **Garofalo et al. (2024)** constatent l'échec du transfert inter-annuel et **D'Addabbo et al. (2025)** rapportent une chute du rappel de >74 % (OA interne) à **10 %** en transfert inter-zones.
  * **Implication :** Les performances publiées en validation interne surestiment fortement la capacité opérationnelle des modèles face à la variabilité des cultivars, des sols et des conditions climatiques méditerranéennes.

---

### Défi 2 — Proposer une nouvelle problématique de recherche originale
* **Problématique formulée :** *« Comment concevoir un système d'alerte précoce multi-échelle couplant séries temporelles Sentinel-2/Sentinel-1 et imagerie thermique/multispectrale UAV, guidé par un modèle de transfert radiatif, capable de détecter Xylella fastidiosa au stade asymptomatique tout en maintenant un rappel supérieur à 75 % lors du transfert vers de nouveaux bassins oléicoles méditerranéens ? »*
* **Méthodologie envisagée :**
  1. **Acquisition & Vérité terrain moléculaire :** Suivi longitudinal sur deux bassins oléicoles distincts (ex. Pouilles en Italie et Andalousie en Espagne) avec prélèvements qPCR répétés sur des arbres visuellement asymptomatiques.
  2. **Modélisation hybride physique + IA :** Extraction des traits physiologiques (chlorophylle, fluorescence SIF, température de canopée) par inversion PROSAIL/DART, combinée à une architecture d'adaptation de domaine non supervisée (Transformer spatio-temporel + attribution SHAP).
  3. **Protocole de validation strict (`V4`) :** Évaluation croisée *leave-one-region-out* (entraînement sur le bassin A, test aveugle sur le bassin B) et *leave-one-year-out*.
* **Métriques d'évaluation :**
  * **Sensibilité pré-symptomatique (Recall)** sur les arbres positifs en qPCR avant apparition des symptômes foliaires ;
  * **Taux de faux positifs (FP)** et **F1-score macro** (pour contrôler le déséquilibre de classes signalé par D'Addabbo et al. 2025) ;
  * **AUROC** et **délai moyen d'anticipation** (en jours avant le diagnostic visuel).
