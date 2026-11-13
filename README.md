# Credit Card Fraud Detection

## Présentation du projet

Ce projet porte sur la détection des transactions bancaires frauduleuses à partir du dataset Credit Card Transactions Fraud Detection (Sparkov).

L'objectif est d'étudier différentes approches de Machine Learning pour identifier les transactions suspectes, en tenant compte d'une difficulté majeure du problème : le fort déséquilibre entre les transactions normales et les transactions frauduleuses.

Pour répondre à cette problématique, j'ai exploré deux approches complémentaires : le Machine Learning non supervisé et le Machine Learning supervisé. Après avoir évalué et comparé les modèles de chaque approche, j'ai développé une approche hybride qui combine les prédictions des deux meilleurs modèles retenus.

---

## 1. Préparation des données

La première étape a consisté à préparer les données afin de construire une base adaptée aux différentes expériences.

J'ai notamment travaillé sur :

- L'exploration du dataset et l'analyse de la distribution des transactions.
- Le nettoyage et la préparation des variables.
- Le Feature Engineering afin de créer des caractéristiques pertinentes pour la détection des comportements suspects.
- La prise en compte du déséquilibre important entre les transactions normales et frauduleuses.
- La séparation des données en ensembles d'entraînement, de validation et de test.

Un point important de cette préparation concerne la séparation temporelle des données. Plutôt que d'effectuer une séparation aléatoire, j'ai conservé l'ordre chronologique des transactions. Cette stratégie permet d'évaluer les modèles dans des conditions plus proches d'une utilisation réelle, où les données futures ne doivent pas influencer l'apprentissage sur les données passées.

Les ensembles de validation et de test conservent la distribution naturelle des classes afin d'obtenir une évaluation représentative des performances.

## 2. Approche non supervisée

J'ai commencé par explorer les méthodes de détection d'anomalies non supervisées. L'intérêt de cette approche est de pouvoir identifier des comportements inhabituels sans utiliser les étiquettes de fraude pendant l'entraînement.

J'ai expérimenté trois modèles :

- Isolation Forest : identifie les observations isolées par rapport au reste des données.
- HBOS (Histogram-Based Outlier Score) : détecte les anomalies en s'appuyant sur la distribution des caractéristiques.
- One-Class SVM : apprend une représentation du comportement normal afin de repérer les observations qui s'en éloignent.

Les modèles ont été entraînés sans utiliser la variable cible `is_fraud`. Les étiquettes ont uniquement servi lors de la validation et de l'évaluation pour mesurer leur capacité à détecter les transactions frauduleuses.

### Comparaison et sélection du meilleur modèle

Après l'entraînement, j'ai comparé les modèles à l'aide de plusieurs métriques :

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Pour cette première comparaison, j'ai accordé une importance particulière au Recall.

Dans un problème de détection de fraude, le Recall mesure la proportion de transactions frauduleuses réellement détectées parmi toutes les transactions frauduleuses existantes.

Ce choix s'explique par le coût potentiel des faux négatifs : une transaction frauduleuse classée comme normale risque de ne pas être détectée. Il est donc important de limiter autant que possible ces cas.

Le Recall a ainsi constitué le critère décisif dans la sélection du meilleur modèle non supervisé, tout en prenant en considération les autres métriques pour mieux comprendre les compromis entre détection et fausses alertes.

## 3. Approche supervisée

Dans un deuxième temps, j'ai exploré les méthodes supervisées, qui apprennent directement à distinguer les transactions normales des transactions frauduleuses à partir des données étiquetées.

Trois modèles ont été étudiés :

- Random Forest
- XGBoost
- Logistic Regression

Une difficulté importante reste le déséquilibre des classes. Pour étudier son influence sur les performances, j'ai mis en place deux expériences.

### Expérience A : données originales et pondération des classes

Dans cette première expérience, j'ai conservé la distribution originale des classes et utilisé une pondération des classes afin de donner davantage d'importance aux transactions frauduleuses pendant l'apprentissage.

### Expérience B : Borderline-SMOTE

Dans cette deuxième expérience, j'ai appliqué **Borderline-SMOTE uniquement sur les données d'entraînement.

Cette méthode génère de nouvelles observations minoritaires à proximité de la frontière de décision, là où les exemples frauduleux peuvent être plus difficiles à distinguer des transactions normales.

Les données de validation et de test n'ont pas été rééquilibrées. Cela permet de conserver une évaluation sur une distribution naturelle et d'éviter toute fuite de données.

### Comparaison et sélection

J'ai ensuite comparé les performances des trois modèles pour les deux expériences, en utilisant les mêmes métriques d'évaluation.

Cette comparaison m'a permis d'étudier l'effet de Borderline-SMOTE par rapport à l'utilisation des poids de classes et d'identifier le modèle supervisé offrant le meilleur compromis pour la détection des fraudes.

## 4. Approche hybride : combinaison des deux approches

Après avoir étudié séparément les modèles supervisés et non supervisés, j'ai choisi d'explorer une approche hybride.

L'idée est de tirer parti des avantages des deux familles de modèles :

- Le modèle supervisé apprend à reconnaître les caractéristiques associées aux fraudes connues.
- Le modèle non supervisé apporte une capacité complémentaire à identifier des comportements atypiques.

J'ai donc combiné les scores du meilleur modèle supervisé et du meilleur modèle non supervisé retenus lors des expériences précédentes.

Cette approche permet de combiner les informations apprises à partir des fraudes connues avec la détection de comportements inhabituels.


## Technologies utilisées

- Langage : Python
- Manipulation des données :Pandas, NumPy
- Machine Learning : Scikit-learn, XGBoost
- Rééquilibrage : Imbalanced-learn (Borderline-SMOTE)
- Visualisation : Matplotlib, Seaborn
- Interface de comparaison : Streamlit
- Sauvegarde des modèles : Joblib
