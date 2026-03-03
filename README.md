# Prédiction de la prime d'assurance

Ce projet a été réalisé dans le cadre du challenge proposé par l'Académie Data Science de **Crédit Agricole Assurances**. 

## 🔗 Plateforme et Données
Les données et le contexte officiel du challenge sont hébergés sur la plateforme ENS Challenge Data :
👉 **[Accéder au Challenge sur challengedata.ens.fr](https://challengedata.ens.fr)**

## 📋 Présentation du Projet
L'objectif principal est de modéliser avec précision la **prime pure incendie** pour les contrats de Multirisque Agricole gérés par Pacifica. 

Le risque incendie est un enjeu stratégique majeur en assurance : les sinistres sont statistiquement rares, mais leur impact financier (la "sévérité") peut être extrêmement élevé. Une modélisation précise permet de garantir la continuité de l'activité des agriculteurs tout en assurant l'équilibre technique de l'assureur.

## Stratégie de Modélisation
Pour répondre à la problématique, j'ai implémenté une **modélisation disjointe Fréquence / Coût Moyen**. Cette approche  permet de traiter séparément la probabilité de survenance d'un sinistre et son intensité financière.

La variable finale (la charge) est reconstruite selon la formule actuarielle :

$$\text{Charge Total} = \text{Fréquence} \times \text{Coût Moyen} \times \text{Exposition}$$

### Détails des Modèles (LightGBM)
Le projet utilise deux modèles basés sur l'algorithme de Gradient Boosting (**LightGBM**) avec des configurations spécifiques :

1.  **Modèle de Fréquence (FREQ)** : 
    * **Distribution** : Poisson (optimisée pour les données de comptage d'événements rares).
    * **Lien** : Log.
2.  **Modèle de Coût Moyen (CM)** : 
    * **Distribution** : Tweedie (efficace pour gérer la forte variance et les montants extrêmes).
    * **Lien** : Log.

## Métriques d'Évaluation
La performance du modèle est mesurée par la **RMSE** (*Root Mean Square Error*) appliquée à la variable **CHARGE**. Cette métrique est particulièrement adaptée car elle pénalise les erreurs importantes de prédiction.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Où :
* $y_i$ : Valeur réelle de la charge.
* $\hat{y}_i$ : Valeur prédite par le modèle.
* $n$ : Nombre d'observations dans le jeu de test.


## Variables Exploitées
Le jeu de données comprend plus de 370 variables explicatives :
* **Données Géographiques & Météo** : Localisation du risque et historique climatique.
* **Données Contrat** : Type d'activité (éleveur, cultivateur, polyculteur) et garanties souscrites.
* **Données Techniques** : Surfaces des bâtiments et capitaux assurés.
* **Prévention** : Présence d'équipements de sécurité et caractéristiques des structures.

## 🛠️ Stack Technique
* **Langage** : Python 
* **Librairies** : `LightGBM`, `Pandas`, `NumPy`, `Scikit-Learn`, `Matplotlib`.
