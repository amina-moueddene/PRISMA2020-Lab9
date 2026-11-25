# Tableau comparatif des 22 études incluses (PRISMA 2020 — Lab 9)

Valeurs de performance citées telles que publiées : tâches, unités et métriques diffèrent, elles ne doivent pas être comparées d'une ligne à l'autre.

| Étude | Année | Plateforme | Modalité | Cible | Approche | Algorithmes | Val. | DRL | QA | Résultat principal |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| Calderón et al. 2015 | 2015 | Airborne | Multimodal | Verticillium dahliae | Classical ML | SVM; Analyse discriminante | V3 | DRL3 | 11/20 | SVM OA 79,2 %, κ 0,50 ; LDA OA 59,0 % |
| Zarco-Tejada et al. 2018 | 2018 | Airborne | Multimodal | Xylella fastidiosa | Classical ML | SVM; Boosting | V3 | DRL3 | 16/20 | OA 80,9 %, κ 0,61 ; détection pré-visuelle |
| Di Nisio et al. 2020 | 2020 | UAV | Multispectral | Xylella fastidiosa | Classical ML | Analyse discriminante | V2 | DRL2 | 9/20 | Sensibilité 0,98, précision 0,93 |
| Blekos et al. 2021 | 2021 | UAV | Multispectral | Verticillium dahliae | Deep learning | CNN | V1 | DRL2 | 5/20 | Délinéation des couronnes 0,89 ; maladie non évaluée |
| Castrignanò et al. 2021 | 2021 | UAV | Multispectral | Xylella fastidiosa | Classical ML | Analyse discriminante | V3 | DRL3 | 10/20 | OA 0,69 ; 0,77 pour la classe précoce |
| Poblete et al. 2021 | 2021 | Airborne | Multimodal | Xylella + Verticillium | Classical ML | Random Forest; SVM | V3 | DRL3 | 14/20 | Xf : OA 92 %, κ 0,80 ; Vd : OA 98 %, κ 0,70 |
| Ksibi et al. 2022 | 2022 | UAV | RGB | Maladies foliaires | Deep learning | CNN | V2 | DRL2 | 9/20 | Accuracy 0,971 (image) |
| Navrozidis et al. 2022 | 2022 | Satellite | Multispectral | Stress mixte biotique/abiotique | Classical ML | SVM; Random Forest; Boosting; Analyse discriminante | V3 | DRL3 | 14/20 | AUROC 0,99 (QDA) |
| Kallel et al. 2022 (= Makhloufi et al. 2022) | 2022 | Satellite | Multispectral | Verticillium dahliae | Classical ML | Réseau de neurones (MLP) | V0 | DRL2 | 4/20 | Aucune métrique (comparaison qualitative) |
| Mamalis et al. 2023 | 2023 | UAV | RGB | Verticillium dahliae | Deep learning | YOLO | V3 | DRL3 | 12/20 | Résultats uniquement en graphique radar |
| Navrozidis et al. 2023 | 2023 | UAV | Hyperspectral | Stress mixte biotique/abiotique | Classical ML | Random Forest; Boosting | V1 | DRL2 | 9/20 | ROC-AUC 0,977 (RF), 0,955 (XGBoost) |
| Poblete et al. 2023 | 2023 | Satellite | Multispectral | Xylella + Verticillium | Classical ML | SVM | V3 | DRL3 | 15/20 | OA 0,63–0,83, κ 0,29–0,68 |
| Garofalo et al. 2024 | 2024 | Satellite | Multispectral | Stress hydrique / sécheresse | Classical ML | Random Forest; SVM; Régression linéaire | V4 | DRL4 | 15/20 | R² 0,78 (split aléatoire) ; transfert inter-annuel sans résultat utile |
| Raouhi et al. 2024 | 2024 | UAV | RGB | Maladies foliaires | Deep learning | CNN | V2 | DRL2 | 8/20 | Accuracy 0,99 (image) |
| Costanza et al. 2025 | 2025 | Satellite | Multispectral | Stress hydrique / sécheresse | Classical ML | Random Forest; SVM; Boosting; kNN; Arbre de décision; Régression linéaire | V3 | DRL3 | 14/20 | R² 0,75, RMSE 0,078 |
| D'Addabbo et al. 2025 | 2025 | Airborne | Multimodal | Xylella fastidiosa | Classical ML | SVM | V4 | DRL4 | 14/20 | OA ≥ 0,74 interne ; rappel 0,10 inter-zones (2/20) |
| Ottaviano et al. 2025 | 2025 | Satellite | Multispectral | Stress hydrique / sécheresse | Classical ML | Random Forest | V2 | DRL2 | 9/20 | R² test 0,73 |
| Oussaoui et al. 2025 | 2025 | Satellite | Multispectral | Stress hydrique / sécheresse | Classical ML | Random Forest; Boosting; SVM; Arbre de décision; Naive Bayes | V2 | DRL2 | 13/20 | OA 95,9 %, κ 0,94 |
| Belwalkar et al. 2026 | 2026 | Airborne | Hyperspectral | Verticillium dahliae | Classical ML | Processus gaussien | V2 | DRL2 | 9/20 | OA 0,77 ± 0,03, κ 0,55 |
| Daiz et al. 2026 | 2026 | Satellite | Multispectral | Stress hydrique / sécheresse | Classical ML | SVM | V2 | DRL2 | 9/20 | OA 96,0 %, κ 0,92 |
| Martínez-Ruedas et al. 2026 | 2026 | Satellite | Multimodal | Stress hydrique / sécheresse | Classical ML | Boosting; SVM; Processus gaussien; Random Forest; Régression linéaire; Réseau de neurones (MLP) | V4 | DRL4 | 18/20 | R² 0,84 (aléatoire), 0,84 (zones), 0,82 (saison) |
| El Atiq et al. 2026 | 2026 | Satellite | Multimodal | Stress hydrique / sécheresse | Classical ML | SVM | V2 | DRL2 | 8/20 | OA 0,927 |
