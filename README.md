# TP ANAD: Analyse Statistique Multivariee du Dataset Butron.Maize

**Auteurs:** Bouguessa Wail & Nemamcha Oussama  
**Date:** Decembre 2025

---

## Description du Projet

Ce projet realise une analyse statistique multivariee complete du dataset **`agridat - butron.maize`** (donnees d'essais de mais). L'analyse suit strictement les methodes enseignees dans le cours ANAD.

### Dataset

- **Source:** Package R `agridat` - `butron.maize`
- **Observations:** 225 (apres nettoyage des valeurs manquantes)
- **Variables:**
    - **Qualitatives:** `gen` (45 modalites), `male` (9), `female` (9), `env` (5)
    - **Quantitative:** `yield` (rendement en t/ha)

---

## Methodes Utilisees (basees sur le cours)

| Methode                   | Reference     | Description                             |
| ------------------------- | ------------- | --------------------------------------- |
| Statistiques descriptives | Cours Rappels | Moyenne, ecart-type, boxplots           |
| ACM                       | Cours ACM1    | Analyse des Correspondances Multiples   |
| AFC                       | Cours AFC     | Analyse Factorielle des Correspondances |

### ACM (Cours ACM1)

- Tableau Disjonctif Complet (TDC) - Section 1.2
- Tableau de Burt - Section 1.2.1
- Calcul de l'inertie totale (I = J/p - 1) - Section 1.5
- Corrections de Benzecri et Greenacre - Section 1.5.2
- Contributions (CTR) - Section 1.7.1
- Qualite de representation (Cos2) - Section 1.7.2

### AFC (Cours AFC)

- Tableau de contingence - Section 1.1.1
- Test du Chi2 (independance) - Section 1.2
- Inertie totale (I = Chi2/K) - Section 1.7.1
- Coordonnees factorielles - Section 1.5
- Contributions (CTR) - Section 1.7.3
- Qualite de representation (Cos2) - Section 1.7.4

---

## Installation

### Avec environnement virtuel (recommande)

```bash
# Creer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/macOS
# OU
venv\Scripts\activate     # Windows

# Installer les dependances
pip install -r requirements.txt
```

### Installation manuelle

```bash
pip install pandas numpy matplotlib seaborn prince scipy scikit-learn
```

---

## Utilisation

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Executer l'analyse
python main.py
```

Les resultats sont sauvegardes dans le dossier `results/`:

- `results/tables/` - Fichiers CSV
- `results/plots/` - Graphiques PNG

---

## Fichiers de Sortie

### Tables CSV

| Fichier                                | Description                            |
| -------------------------------------- | -------------------------------------- |
| `statistiques_yield.csv`               | Statistiques descriptives du rendement |
| `effectifs_variables_qualitatives.csv` | Effectifs par modalite                 |
| `tdc_extrait.csv`                      | Extrait du TDC (20 lignes)             |
| `burt_extrait.csv`                     | Extrait du Tableau de Burt (15x15)     |
| `acm_parametres_inertie.csv`           | Parametres d'inertie ACM               |
| `acm_valeurs_propres.csv`              | Valeurs propres ACM                    |
| `acm_benzecri_greenacre.csv`           | Corrections Benzecri/Greenacre         |
| `acm_coordonnees_modalites.csv`        | Coordonnees des modalites              |
| `acm_contributions.csv`                | Contributions CTR                      |
| `acm_cos2.csv`                         | Qualite de representation Cos2         |
| `contingence_male_female.csv`          | Tableau de contingence Male x Female   |
| `afc_test_chi2.csv`                    | Resultat du test Chi2                  |
| `afc_valeurs_propres.csv`              | Valeurs propres AFC                    |
| `afc_coordonnees_males.csv`            | Coordonnees des males                  |
| `afc_coordonnees_females.csv`          | Coordonnees des females                |
| `afc_contributions_males.csv`          | Contributions des males                |
| `afc_contributions_females.csv`        | Contributions des females              |
| `afc_cos2_males.csv`                   | Cos2 des males                         |
| `afc_cos2_females.csv`                 | Cos2 des females                       |

### Graphiques PNG

| Fichier                          | Description                              |
| -------------------------------- | ---------------------------------------- |
| `distribution_rendement.png`     | Histogramme et boxplot du rendement      |
| `distributions_qualitatives.png` | Distributions des variables qualitatives |
| `acm_eboulis.png`                | Graphe des valeurs propres (eboulis)     |
| `acm_individus.png`              | Carte des individus ACM                  |
| `acm_par_environnement.png`      | Carte ACM par environnement              |
| `afc_biplot.png`                 | Biplot AFC Male x Female                 |

---

## References

- Cours ANAD - Ens. N. BESSAH (2025)
- Benzecri, J.P. (1970) - Correction des valeurs propres
- Greenacre, M.J. (1993) - Correction de l'inertie totale
