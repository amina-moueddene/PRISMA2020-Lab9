# Research Gaps Argumentés et Pistes de Recherche Futures (Étapes 11–12 & Défis 1–2)

Synthèse générée automatiquement à partir du corpus de **22 études** (score QA médian : **10.5/20** ; **11 limitations récurrentes** ≥ 3 études, dont **5 de priorité haute**).

## 1. Limitations récurrentes de priorité haute extraites de `potential_gaps.csv`

| Limitation récurrente | Études (n) | % corpus | Gap de la revue | Priorité |
| :--- | :---: | :---: | :--- | :---: |
| Aucun test de transfert explicite entre sites ou années | 19/22 | 86.4 % | Few transfer tests | Haute |
| Indépendance entraînement test non assurée | 19/22 | 86.4 % | Potential leakage | Haute |
| Référence non confirmée par laboratoire ou mesure physiologique | 15/22 | 68.2 % | Weak reference data | Haute |
| Voie potentielle de fuite de données | 14/22 | 63.6 % | Potential leakage | Haute |
| Infection non dissociée du déficit hydrique | 11/22 | 50.0 % | Infection confounded with water status | Haute |

## 2. Trois Research Gaps majeurs argumentés (Étape 11 — Défi 1)

### Gap 1 — Déficit de validation par transfert spatio-temporel et exposition aux fuites de données
- **Description :** 19 études sur 22 (86,4 %) n'évaluent aucun transfert explicite vers un verger, une région ou une année indépendante (`V0`–`V3`), et aucune n'utilise de jeu de test externe indépendant (`V5`). Parallèlement, 14 études sur 22 (63,6 %) présentent au moins une voie potentielle de fuite de données.
- **Justification empirique :** Sur les 12 études rapportant une précision globale (OA/Accuracy), la performance chute significativement lorsque le niveau de validation augmente (Spearman ρ = -0.7, p = 0.011 ; médiane 0.96 en `V0`–`V2` contre 0.8 en `V3`–`V4`). En `V4`, D'Addabbo et al. (2025) observent un rappel de seulement 0,10 lors du transfert inter-zones, et Garofalo et al. (2024) rapportent l'échec du transfert inter-annuel.
- **Implication :** Les performances mesurées en validation interne surestiment la capacité de généralisation opérationnelle des modèles.

### Gap 2 — Faiblesse de la vérité terrain biologique et physiologique
- **Description :** 15 études sur 22 (68,2 %) s'appuient sur des annotations visuelles ou des relevés sans confirmation moléculaire (qPCR/PCR) ni mesure physiologique directe (potentiel hydrique de tige).
- **Justification empirique :** Seules 3 études utilisent une confirmation qPCR/PCR et 4 une mesure physiologique ; aucune des 4 études de Deep Learning ne dispose d'une validation biologique en laboratoire. La détection pré-symptomatique n'est démontrée rigoureusement que par Zarco-Tejada et al. (2018).
- **Implication :** Les classifieurs entraînés sur labels visuels modélisent la perception humaine des symptômes tardifs plutôt que l'infection latente.

### Gap 3 — Confusion physiologique entre infection vasculaire et déficit hydrique
- **Description :** 11 études sur 22 (50,0 % — soit 100 % des études sur pathogènes vasculaires) ne dissocient pas expérimentalement l'infection par *Xylella fastidiosa* ou *Verticillium dahliae* du stress hydrique abiotique.
- **Justification empirique :** Les deux stress induisent une fermeture stomatique et des altérations similaires des indices spectraux et thermiques. Belwalkar et al. (2026) montrent qu'à température de couvert comparable, les indices classiques échouent à séparer les arbres précoces des arbres asymptomatiques (OA = 0,51), nécessitant la fluorescence résolue par raies de Fraunhofer (OA = 0,77).
- **Implication :** En contexte méditerranéen semi-aride, les alertes issues de capteurs thermiques ou multispectraux présentent un risque élevé de faux positifs liés à la sécheresse.

## 3. Trois pistes de recherche futures opérationnelles (Étape 12 — Défi 2)

1. **Piste 1 — Benchmark spatio-temporel strict (`leave-orchard-out`, `leave-region-out`, `leave-season-out`) :**
   - *Hypothèse :* L'écart de performance entre split aléatoire et split par blocs spatio-temporels est statistiquement significatif et dépend de la plateforme capteur.
   - *Protocole :* Comparer systématiquement split aléatoire, *leave-region-out* et *leave-season-out* sur un jeu multi-sites ouvert en appliquant prétraitements et rééquilibrage strictement après séparation.
   - *Métriques :* Sensibilité, précision et F1 de la classe infectée à prévalence réelle, AUPRC et écart de généralisation ΔF1.
2. **Piste 2 — Protocole de référence stratifié confirmé par qPCR et chambre à pression :**
   - *Hypothèse :* L'entraînement sur des étiquettes confirmées par qPCR et potentiel hydrique modifie le classement des modèles et améliore la détection pré-symptomatique.
   - *Protocole :* Échantillonnage stratifié incluant des arbres asymptomatiques testés par qPCR, double notation visuelle (κ inter-observateurs) et mesure du potentiel hydrique de tige.
   - *Métriques :* Taux d'erreur des labels visuels, κ inter-juges, sensibilité pré-symptomatique et taux de faux positifs.
3. **Piste 3 — Plan factoriel croisé Infection × Régime d'irrigation avec covariable hydrique satellitaire :**
   - *Hypothèse :* L'intégration du statut hydrique estimé par série temporelle satellitaire comme covariable réduit les faux positifs chez les arbres sains en déficit hydrique.
   - *Protocole :* Dispositif expérimental à 4 modalités (sain/infecté × irrigué/non irrigué) combinant imagerie hyperspectrale/thermique et indices Sentinel-2.
   - *Métriques :* Matrice de confusion multi-classes séparant « sain en stress hydrique » et « infecté », taux de faux positifs sous déficit hydrique.
