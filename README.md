# Lab 9 : Synthèse et Research Gaps

---

## 1. Questions de compréhension

**Q1. Expliquez pourquoi une limitation qui apparaît dans seulement une étude ne peut pas être considérée comme un research gap. Illustrez avec un exemple tiré de votre tableau.**
Une limitation individuelle reflète une contrainte matérielle ou méthodologique propre à une expérience spécifique, tandis qu'un *research gap* identifie une lacune systémique dans la littérature scientifique qui bloque l'avancée du domaine. 
*Exemple tiré de ma SLR :* Une étude (Belwalkar et al., 2026) mentionne comme limitation l'utilisation d'une seule date de vol pour capter les bandes d'absorption $O_2-A$. C'est une limitation individuelle. En revanche, le fait que 19 des 22 études du corpus (ex: Ksibi et al., Mamalis et al., Daiz et al.) n'effectuent aucun test de transfert géographique ou temporel (niveaux de validation V0 à V3) constitue un véritable *research gap*.

**Q2. Décrivez la procédure exacte (en code) utilisée pour transformer la colonne Limitation en une liste de bigrammes et identifier les limitations récurrentes.**
1. **Nettoyage textuel (`clean_limitation`) :** Conversion du texte en minuscules, utilisation d'expressions régulières (`re.sub`) pour ne garder que les caractères alphabétiques, et tokenisation (découpage en mots). 
2. **Filtrage :** Suppression des mots vides (stopwords) via la bibliothèque `nltk`.
3. **Génération des N-grammes (`get_ngrams`) :** Parcours de la liste de tokens avec une boucle glissante de taille `n=2` pour concaténer les mots adjacents (ex: `["données", "limitées"]`).
4. **Comptage :** Agrégation de tous les bigrammes générés dans la colonne, puis utilisation de `collections.Counter` pour extraire les fréquences d'apparition (ex: `most_common(15)`).

**Q3. Comment le graphique des tendances annuelles peut-il influencer le choix d'une direction de recherche future ? Donnez un exemple concret basé sur vos résultats.**
L'analyse des tendances temporelles permet d'identifier les technologies émergentes ou les cibles récemment priorisées.
*Exemple concret :* Mon analyse montre que l'utilisation du Deep Learning (4 études) est apparue récemment mais reste strictement cantonnée aux images RGB par drone (UAV) pour les symptômes visibles. Parallèlement, on observe un pic massif d'études sur le stress hydrique depuis 2024, dominé par le ML classique sur images satellites. Une direction de recherche future évidente est donc de combler le croisement manquant : **appliquer des architectures Deep Learning (ex: Vision Transformers) aux séries temporelles satellitaires pour la détection précoce du stress hydrique**.

---

## 2. Défis pratiques

### Défi 1 – Transformer une limitation récurrente en research gap

**Gap identifié : Vulnérabilité aux fuites de données (Data Leakage) et absence de validation externe.**

*   **Justification :** 14 des 22 études incluses présentent au moins une faille méthodologique liée au *data leakage* (ex: split aléatoire ignorant l'autocorrélation spatiale des pixels, ou mélange de dates). Seules 3 études (ex: D'Addabbo et al., 2025 ; Martínez-Ruedas et al., 2026) effectuent un test explicite de transfert de modèle (niveau V4), et aucune n'utilise de jeu de données externe indépendant (V5). Les modèles Deep Learning du corpus sont particulièrement touchés (tous validés au niveau V1 ou V2).
*   **Implication :** Les métriques rapportées dans la littérature (très souvent >95% en Accuracy globale) sont artificiellement gonflées. En conditions réelles d'opération, ces modèles risquent de s'effondrer, comme le prouve l'étude de D'Addabbo et al. où le rappel (recall) chute dramatiquement à 10% lors du transfert du modèle sur une nouvelle zone géographique à faible prévalence.

### Défi 2 – Proposer une nouvelle problématique de recherche

*À partir du gap identifié dans ma SLR concernant la confusion systématique entre l'infection vasculaire (*Xylella*, *Verticillium*) et le déficit hydrique (les deux provoquant la fermeture stomatique).*

*   **Problématique de recherche :** Comment dissocier l'empreinte spectrale induite par le déficit hydrique de celle causée par les infections vasculaires de l'olivier en utilisant une approche d'IA multimodale ?
*   **Hypothèse :** L'utilisation de la température de la canopée comme variable de contrôle (pour isoler les arbres ayant un statut hydrique similaire), couplée à l'analyse de traits hyperspectraux physiologiques (ex: fluorescence), permet à un modèle de Machine Learning d'isoler le signal spécifique à l'infection pathogène.
*   **Méthodologie :**
    1. Mettre en place un plan expérimental factoriel : arbres infectés vs sains, croisés avec irrigation normale vs stress hydrique induit.
    2. Collecter des données multimodales aéroportées (Thermique + Hyperspectral).
    3. Entraîner un modèle de ML (ex: Random Forest) en forçant l'inclusion du CWSI (Crop Water Stress Index) issu du thermique comme covariable de contrôle.
*   **Métriques d'évaluation :** 
    * Matrice de confusion distinguant spécifiquement 3 classes : "Sain irrigué", "Sain stressé hydriquement", "Infecté".
    * Taux de Faux Positifs (FPR) calculé spécifiquement sur le groupe "Sain stressé hydriquement" (pour prouver que la sécheresse n'est plus classée comme maladie).
    * Sensibilité et Précision de la classe "Infecté".
