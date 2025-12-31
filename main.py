"""
=============================================================================
ANALYSE STATISTIQUE DES DONNEES D'ESSAI MULTILOCAL DE MAIS
=============================================================================

Auteurs: Bouguessa Wail & Nemamcha Oussama
Cours: Analyse des Donnees (ANAD) - Ens. N. BESSAH
Date: Decembre 2025
Dataset: agridat - butron.maize

Description:
Ce script effectue une analyse statistique complete d'un essai de mais
multilocal avec 245 observations. L'analyse suit strictement les methodes
enseignees dans le cours ANAD.

Structure du dataset:
- gen: Genotype (croisement male x femelle)
- male: Ligne parentale male
- female: Ligne parentale femelle
- env: Code environnement
- yield: Rendement en grain (quantitative, t/ha)

Methodes utilisees (basees sur le cours ANAD):
1. Statistiques descriptives (Cours Rappels)
2. ACM - Analyse des Correspondances Multiples (Cours ACM1)
   - Tableau Disjonctif Complet (TDC)
   - Tableau de Burt
   - Valeurs propres et inertie
   - Corrections de Benzecri et Greenacre
   - Contributions (CTR) et Cos2
3. AFC - Analyse Factorielle des Correspondances (Cours AFC)
   - Tableau de contingence
   - Test du Chi2 (test d'independance)
   - Valeurs propres et inertie
   - Coordonnees, Contributions, Cos2
   - Biplot

Prerequis:
    pip install pandas numpy matplotlib seaborn prince scipy scikit-learn

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince
from scipy.stats import chi2_contingency
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Dossier de sortie
DOSSIER_RESULTATS = 'results'
DOSSIER_TABLES = os.path.join(DOSSIER_RESULTATS, 'tables')
DOSSIER_PLOTS = os.path.join(DOSSIER_RESULTATS, 'plots')


def creer_dossiers_sortie():
    """Cree les dossiers de sortie s'ils n'existent pas."""
    os.makedirs(DOSSIER_TABLES, exist_ok=True)
    os.makedirs(DOSSIER_PLOTS, exist_ok=True)
    print(f"Dossiers de sortie crees: {DOSSIER_RESULTATS}/")


# =============================================================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# =============================================================================

def charger_et_preparer_donnees(chemin_fichier):
    """Charge le dataset de mais et le prepare pour l'analyse."""
    print("="*70)
    print("ETAPE 1: CHARGEMENT ET PREPARATION DES DONNEES")
    print("="*70)

    df = pd.read_csv(chemin_fichier)
    colonnes_necessaires = ['gen', 'male', 'female', 'env', 'yield']
    df = df[colonnes_necessaires]

    print(f"\nDimensions du dataset: {df.shape}")
    print(f"Valeurs manquantes:\n{df.isnull().sum()}")

    df = df.dropna()

    print(f"\nDimensions finales: {df.shape}")
    print(f"\nPremieres observations:")
    print(df.head())

    print("\n--- MODALITES DES VARIABLES QUALITATIVES ---")
    for var in ['gen', 'male', 'female', 'env']:
        n_modalites = df[var].nunique()
        print(f"{var}: {n_modalites} modalites")

    return df


# =============================================================================
# 2. STATISTIQUES DESCRIPTIVES (Cours Rappels)
# =============================================================================

def statistiques_descriptives(df):
    """Genere les statistiques descriptives pour toutes les variables."""
    print("\n" + "="*70)
    print("ETAPE 2: STATISTIQUES DESCRIPTIVES (Cours Rappels)")
    print("="*70)

    resultats = {}

    print("\n--- VARIABLE QUANTITATIVE: YIELD (Rendement) ---")
    stats_yield = df['yield'].describe()
    stats_yield['Variance'] = df['yield'].var()
    print(stats_yield)

    resultats['yield_stats'] = stats_yield

    stats_yield_df = pd.DataFrame({
        'Statistique': ['Nombre', 'Moyenne', 'Ecart-type', 'Minimum', 'Q1 (25%)', 
                        'Mediane (50%)', 'Q3 (75%)', 'Maximum', 'Variance'],
        'Valeur': [stats_yield['count'], stats_yield['mean'], stats_yield['std'],
                   stats_yield['min'], stats_yield['25%'], stats_yield['50%'],
                   stats_yield['75%'], stats_yield['max'], stats_yield['Variance']]
    })
    stats_yield_df.to_csv(os.path.join(DOSSIER_TABLES, 'statistiques_yield.csv'), 
                          index=False, encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/statistiques_yield.csv")

    print("\n--- VARIABLES QUALITATIVES ---")
    vars_qualitatives = ['gen', 'male', 'female', 'env']

    effectifs_tous = []
    for var in vars_qualitatives:
        effectifs = df[var].value_counts()
        print(f"\n{var.upper()}: {len(effectifs)} modalites")
        print(effectifs.head())
        resultats[f'{var}_effectifs'] = effectifs
        
        for modalite, eff in effectifs.items():
            effectifs_tous.append({
                'Variable': var,
                'Modalite': modalite,
                'Effectif': eff,
                'Frequence (%)': round(eff / len(df) * 100, 2)
            })

    effectifs_df = pd.DataFrame(effectifs_tous)
    effectifs_df.to_csv(os.path.join(DOSSIER_TABLES, 'effectifs_variables_qualitatives.csv'),
                        index=False, encoding='utf-8-sig')
    print(f"\n  -> Exporte: {DOSSIER_TABLES}/effectifs_variables_qualitatives.csv")

    return resultats


def tracer_distributions(df):
    """Cree les visualisations pour les distributions (boxplots, histogrammes)."""
    print("\n--- Generation des graphiques de distribution ---")

    # Distribution du rendement (histogramme + boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['yield'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['yield'].mean(), color='red', linestyle='--',
                    label=f'Moyenne = {df["yield"].mean():.2f}')
    axes[0].axvline(df['yield'].median(), color='green', linestyle='--',
                    label=f'Mediane = {df["yield"].median():.2f}')
    axes[0].set_xlabel('Rendement (t/ha)')
    axes[0].set_ylabel('Frequence')
    axes[0].set_title('Distribution du Rendement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boxplot (comme dans Cours Rappels Figure 1.1)
    axes[1].boxplot(df['yield'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
    axes[1].set_ylabel('Rendement (t/ha)')
    axes[1].set_title('Boite a moustaches du Rendement')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'distribution_rendement.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")

    # Distribution des variables qualitatives
    vars_qualitatives = ['male', 'female', 'env']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, var in enumerate(vars_qualitatives):
        effectifs = df[var].value_counts()
        axes[idx].bar(range(len(effectifs)), effectifs.values, color='coral', edgecolor='black')
        axes[idx].set_xticks(range(len(effectifs)))
        axes[idx].set_xticklabels(effectifs.index, rotation=45, ha='right')
        axes[idx].set_xlabel(var.capitalize())
        axes[idx].set_ylabel('Effectif')
        axes[idx].set_title(f'Distribution de {var.capitalize()}')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'distributions_qualitatives.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")


# =============================================================================
# 3. ANALYSE DES CORRESPONDANCES MULTIPLES (ACM) - Cours ACM1
# =============================================================================

def construire_tdc_et_burt(df):
    """Construit le Tableau Disjonctif Complet (TDC) et le Tableau de Burt.
    
    Reference: Cours ACM1 Section 1.2
    """
    print("\n" + "="*70)
    print("CONSTRUCTION DU TDC ET TABLEAU DE BURT (Cours ACM1 Section 1.2)")
    print("="*70)

    vars_qual = ['gen', 'male', 'female', 'env']
    tdc = pd.get_dummies(df[vars_qual], prefix=vars_qual, prefix_sep='_')

    print(f"\nDimensions du TDC: {tdc.shape}")
    print(f"Nombre total de modalites J = {tdc.shape[1]}")
    print(f"Nombre de variables p = {len(vars_qual)}")

    # Verification: somme par ligne = nombre de variables
    somme_lignes = tdc.sum(axis=1)
    print(f"\nVerification TDC: somme par ligne = {somme_lignes.iloc[0]} (doit etre = {len(vars_qual)})")

    # Tableau de Burt = TDC' x TDC
    burt = tdc.T @ tdc
    print(f"\nDimensions du Tableau de Burt: {burt.shape}")

    # Export
    tdc_extrait = tdc.head(20)
    tdc_extrait.to_csv(os.path.join(DOSSIER_TABLES, 'tdc_extrait.csv'),
                       encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/tdc_extrait.csv (20 premieres lignes)")

    burt_extrait = burt.iloc[:15, :15]
    burt_extrait.to_csv(os.path.join(DOSSIER_TABLES, 'burt_extrait.csv'),
                        encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/burt_extrait.csv (extrait 15x15)")

    return tdc, burt


def calculer_inertie_acm(tdc, p):
    """Calcule l'inertie totale et moyenne pour l'ACM.
    
    Reference: Cours ACM1 Section 1.5
    Formules:
    - Inertie totale I = J/p - 1
    - Valeur propre moyenne = 1/p
    - Nombre max d'axes r = min(n-1, J-p)
    """
    print("\n--- CALCUL DE L'INERTIE (Cours ACM1 Section 1.5) ---")

    n = tdc.shape[0]  # Nombre d'individus
    J = tdc.shape[1]  # Nombre total de modalites

    inertie_totale = J / p - 1
    lambda_moyen = 1 / p
    r_max = min(n - 1, J - p)

    print(f"\nNombre d'individus n = {n}")
    print(f"Nombre de variables p = {p}")
    print(f"Nombre total de modalites J = {J}")
    print(f"\nInertie totale I = J/p - 1 = {J}/{p} - 1 = {inertie_totale:.4f}")
    print(f"Valeur propre moyenne = 1/p = 1/{p} = {lambda_moyen:.4f}")
    print(f"Nombre maximum d'axes r = min(n-1, J-p) = min({n-1}, {J-p}) = {r_max}")

    # Export
    inertie_df = pd.DataFrame({
        'Parametre': ['Nombre d\'individus (n)', 'Nombre de variables (p)', 
                      'Nombre de modalites (J)', 'Inertie totale (I = J/p - 1)',
                      'Valeur propre moyenne (1/p)', 'Nombre max d\'axes (r)'],
        'Valeur': [n, p, J, inertie_totale, lambda_moyen, r_max]
    })
    inertie_df.to_csv(os.path.join(DOSSIER_TABLES, 'acm_parametres_inertie.csv'),
                      index=False, encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/acm_parametres_inertie.csv")

    return {
        'inertie_totale': inertie_totale,
        'lambda_moyen': lambda_moyen,
        'r_max': r_max,
        'n': n,
        'p': p,
        'J': J
    }


def correction_benzecri_greenacre(valeurs_propres, p, J):
    """Applique les corrections de Benzecri et Greenacre.
    
    Reference: Cours ACM1 Section 1.5.2
    
    Correction de Benzecri (1970):
    - Seuil: on retient les valeurs propres > 1/p
    - Lambda_tilde = ((p/(p-1)) * (lambda - 1/p))^2
    
    Correction de Greenacre (1993):
    - Modifie l'inertie totale pour des taux plus realistes
    """
    print("\n" + "="*70)
    print("CORRECTIONS DE BENZECRI ET GREENACRE (Cours ACM1 Section 1.5.2)")
    print("="*70)

    seuil = 1 / p
    print(f"\nSeuil de retention: 1/p = {seuil:.4f}")

    # Valeurs propres superieures au seuil
    vp_retenues = valeurs_propres[valeurs_propres > seuil]
    s = len(vp_retenues)

    print(f"Nombre de valeurs propres > {seuil:.4f}: {s}")

    if s == 0:
        print("Aucune valeur propre ne depasse le seuil. Corrections non applicables.")
        return None

    # Correction de Benzecri
    lambda_tilde = ((p / (p - 1)) * (vp_retenues - 1/p)) ** 2
    S_B = lambda_tilde.sum()

    # Correction de Greenacre
    somme_lambda_carre = (valeurs_propres ** 2).sum()
    S_G = (p / (p - 1)) * (somme_lambda_carre - (J - p) / (p ** 2))

    print(f"\nSomme Benzecri S_B = {S_B:.4f}")
    print(f"Somme Greenacre S_G = {S_G:.4f}")

    # Tableau des resultats
    resultats = pd.DataFrame({
        'Dimension': range(1, s + 1),
        'Lambda_originale': vp_retenues,
        'Lambda_Benzecri': lambda_tilde,
        'Inertie_Benzecri (%)': (lambda_tilde / S_B * 100) if S_B > 0 else np.zeros(s),
        'Inertie_Greenacre (%)': (lambda_tilde / S_G * 100) if S_G > 0 else np.zeros(s)
    })

    resultats['Cumul_Benzecri (%)'] = resultats['Inertie_Benzecri (%)'].cumsum()
    resultats['Cumul_Greenacre (%)'] = resultats['Inertie_Greenacre (%)'].cumsum()

    print("\n--- TABLEAU DES VALEURS PROPRES CORRIGEES ---")
    print(resultats.to_string(index=False))

    resultats.to_csv(os.path.join(DOSSIER_TABLES, 'acm_benzecri_greenacre.csv'),
                     index=False, encoding='utf-8-sig')
    print(f"\n  -> Exporte: {DOSSIER_TABLES}/acm_benzecri_greenacre.csv")

    return resultats


def effectuer_acm(df):
    """Effectue l'Analyse des Correspondances Multiples.
    
    Reference: Cours ACM1
    """
    print("\n" + "="*70)
    print("ETAPE 3: ANALYSE DES CORRESPONDANCES MULTIPLES (ACM)")
    print("="*70)

    df_qual = df[['gen', 'male', 'female', 'env']].copy()
    p = df_qual.shape[1]

    print(f"\nDataset pour ACM: {df_qual.shape}")
    print(f"Variables: {list(df_qual.columns)}")

    # Construction TDC et Burt
    tdc, burt = construire_tdc_et_burt(df)
    info_inertie = calculer_inertie_acm(tdc, p)

    # Ajustement du modele ACM
    print("\n--- AJUSTEMENT DU MODELE ACM ---")
    n_composantes = min(10, info_inertie['r_max'])
    mca = prince.MCA(
        n_components=n_composantes,
        n_iter=10,
        copy=True,
        random_state=42,
        engine='sklearn'
    )

    mca = mca.fit(df_qual)

    # Valeurs propres et inertie
    print("\n--- VALEURS PROPRES ET INERTIE EXPLIQUEE ---")
    valeurs_propres = mca.eigenvalues_
    inertie_expliquee = mca.percentage_of_variance_

    print("\nDim | Val. Propre | Inertie (%) | Cumulee (%)")
    print("-" * 55)
    cumul = 0
    vp_data = []
    for i, (vp, inertie) in enumerate(zip(valeurs_propres, inertie_expliquee)):
        cumul += inertie
        print(f" {i+1:2d} |   {vp:7.4f}   |   {inertie:6.2f}    |   {cumul:6.2f}")
        vp_data.append({
            'Dimension': i + 1,
            'Valeur_Propre': vp,
            'Inertie (%)': inertie,
            'Inertie_Cumulee (%)': cumul
        })

    vp_df = pd.DataFrame(vp_data)
    vp_df.to_csv(os.path.join(DOSSIER_TABLES, 'acm_valeurs_propres.csv'),
                 index=False, encoding='utf-8-sig')
    print(f"\n  -> Exporte: {DOSSIER_TABLES}/acm_valeurs_propres.csv")

    # Corrections de Benzecri et Greenacre
    corrections = correction_benzecri_greenacre(valeurs_propres, p, info_inertie['J'])

    # Coordonnees des modalites
    coords_modalites = mca.column_coordinates(df_qual)
    coords_modalites.columns = [f'Dim_{i+1}' for i in range(coords_modalites.shape[1])]
    coords_modalites.to_csv(os.path.join(DOSSIER_TABLES, 'acm_coordonnees_modalites.csv'),
                            encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/acm_coordonnees_modalites.csv")

    # Contributions des modalites (CTR) - Cours ACM1 Section 1.7.1
    print("\n--- CONTRIBUTIONS DES MODALITES (CTR) - Section 1.7.1 ---")
    try:
        contributions = mca.column_contributions_
        contributions.columns = [f'CTR_Dim_{i+1}' for i in range(contributions.shape[1])]
        contributions.to_csv(os.path.join(DOSSIER_TABLES, 'acm_contributions.csv'),
                            encoding='utf-8-sig')
        print(f"  -> Exporte: {DOSSIER_TABLES}/acm_contributions.csv")
    except AttributeError:
        print("  Calcul manuel des contributions...")
        freq_modalites = tdc.sum(axis=0) / tdc.sum().sum()
        contributions_manuelles = []
        for i in range(min(5, len(valeurs_propres))):
            ctr = (freq_modalites * coords_modalites[f'Dim_{i+1}']**2) / valeurs_propres[i]
            contributions_manuelles.append(ctr * 100)
        ctr_df = pd.DataFrame(contributions_manuelles).T
        ctr_df.columns = [f'CTR_Dim_{i+1} (%)' for i in range(ctr_df.shape[1])]
        ctr_df.to_csv(os.path.join(DOSSIER_TABLES, 'acm_contributions.csv'),
                      encoding='utf-8-sig')
        print(f"  -> Exporte: {DOSSIER_TABLES}/acm_contributions.csv")

    # Cos2 (qualite de representation) - Cours ACM1 Section 1.7.2
    print("\n--- QUALITE DE REPRESENTATION (COS2) - Section 1.7.2 ---")
    try:
        cos2 = mca.column_cosine_similarities(df_qual)
        cos2.columns = [f'Cos2_Dim_{i+1}' for i in range(cos2.shape[1])]
        cos2.to_csv(os.path.join(DOSSIER_TABLES, 'acm_cos2.csv'),
                    encoding='utf-8-sig')
        print(f"  -> Exporte: {DOSSIER_TABLES}/acm_cos2.csv")
    except Exception:
        print("  Calcul manuel du cos2...")
        cos2_manuel = coords_modalites ** 2
        cos2_total = cos2_manuel.sum(axis=1)
        cos2_normalise = cos2_manuel.div(cos2_total, axis=0)
        cos2_normalise.columns = [f'Cos2_Dim_{i+1}' for i in range(cos2_normalise.shape[1])]
        cos2_normalise.to_csv(os.path.join(DOSSIER_TABLES, 'acm_cos2.csv'),
                              encoding='utf-8-sig')
        print(f"  -> Exporte: {DOSSIER_TABLES}/acm_cos2.csv")

    # Graphique d'eboulis (Scree plot)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(valeurs_propres) + 1), inertie_expliquee,
            color='steelblue', edgecolor='black', alpha=0.7)
    plt.plot(range(1, len(valeurs_propres) + 1), inertie_expliquee,
             'o-', color='red', linewidth=2, markersize=8)
    plt.axhline(y=100/p, color='green', linestyle='--',
                label=f'Seuil 1/p = {100/p:.2f}%')
    plt.xlabel('Dimension')
    plt.ylabel('Inertie Expliquee (%)')
    plt.title('Graphe des Valeurs Propres - ACM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'acm_eboulis.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")

    # Coordonnees des individus
    coords_individus = mca.transform(df_qual)

    # Carte des individus
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords_individus[0], coords_individus[1],
                         c=df['yield'], cmap='viridis',
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Rendement (t/ha)')
    plt.xlabel(f'Dimension 1 ({inertie_expliquee[0]:.2f}%)')
    plt.ylabel(f'Dimension 2 ({inertie_expliquee[1]:.2f}%)')
    plt.title('ACM: Carte des Individus')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.grid(True, alpha=0.3)
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'acm_individus.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")

    # Carte par environnement
    plt.figure(figsize=(12, 8))
    for env in df['env'].unique():
        mask = df['env'] == env
        plt.scatter(coords_individus[0][mask], coords_individus[1][mask],
                   label=env, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'Dimension 1 ({inertie_expliquee[0]:.2f}%)')
    plt.ylabel(f'Dimension 2 ({inertie_expliquee[1]:.2f}%)')
    plt.title('ACM: Carte des Individus par Environnement')
    plt.legend(title='Environnement')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.grid(True, alpha=0.3)
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'acm_par_environnement.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")

    return mca, tdc


# =============================================================================
# 4. ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC) - Cours AFC
# =============================================================================

def effectuer_afc(df):
    """Effectue l'Analyse Factorielle des Correspondances sur male x female.
    
    Reference: Cours AFC
    
    Implementation manuelle basee sur le cours AFC pour eviter les problemes
    de compatibilite avec la bibliotheque prince.
    """
    print("\n" + "="*70)
    print("ETAPE 4: ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC)")
    print("="*70)

    # Tableau de contingence (Cours AFC Section 1.1.1)
    print("\n--- TABLEAU DE CONTINGENCE MALE x FEMALE (Section 1.1.1) ---")
    ct = pd.crosstab(df['male'], df['female'])
    print(f"Dimensions: {ct.shape}")
    
    # Export du tableau de contingence
    ct.to_csv(os.path.join(DOSSIER_TABLES, 'contingence_male_female.csv'),
              encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/contingence_male_female.csv")

    # Test du Chi2 (Cours AFC Section 1.2)
    print("\n--- TEST DU CHI2 (Section 1.2) ---")
    chi2, p_value, ddl, attendus = chi2_contingency(ct)
    print(f"Chi2 = {chi2:.4f}")
    print(f"p-valeur = {p_value:.4e}")
    print(f"Degres de liberte = {ddl}")
    
    # Decision du test
    alpha = 0.05
    if p_value < alpha:
        print(f"\nDecision: p-valeur < {alpha} => On rejette H0")
        print("Les variables Male et Female sont DEPENDANTES")
    else:
        print(f"\nDecision: p-valeur >= {alpha} => On ne rejette pas H0")
        print("Les variables Male et Female sont INDEPENDANTES")

    # Export test Chi2
    chi2_df = pd.DataFrame({
        'Test': ['Chi2 (Male x Female)'],
        'Chi2': [chi2],
        'Degres_liberte': [ddl],
        'p_valeur': [p_value],
        'Significatif (alpha=0.05)': ['Oui' if p_value < 0.05 else 'Non']
    })
    chi2_df.to_csv(os.path.join(DOSSIER_TABLES, 'afc_test_chi2.csv'),
                   index=False, encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_test_chi2.csv")

    # Inertie totale (Cours AFC Section 1.7.1)
    K = ct.sum().sum()
    inertie_totale = chi2 / K
    print(f"\n--- INERTIE TOTALE (Section 1.7.1) ---")
    print(f"Effectif total K = {K}")
    print(f"Inertie totale I = Chi2/K = {chi2:.4f}/{K} = {inertie_totale:.6f}")

    # Implementation manuelle de l'AFC
    print("\n--- CALCUL DE L'AFC ---")
    
    # Matrice des frequences
    P = ct.values / K
    
    # Marges (profils marginaux)
    fi_point = P.sum(axis=1)  # Profils lignes marginaux
    f_point_j = P.sum(axis=0)  # Profils colonnes marginaux
    
    # Matrices diagonales des masses
    Di_sqrt_inv = np.diag(1.0 / np.sqrt(np.where(fi_point > 0, fi_point, 1)))
    Dj_sqrt_inv = np.diag(1.0 / np.sqrt(np.where(f_point_j > 0, f_point_j, 1)))
    
    # Matrice centree (Cours AFC Section 1.4)
    expected = np.outer(fi_point, f_point_j)
    residuals = P - expected
    S = Di_sqrt_inv @ residuals @ Dj_sqrt_inv
    
    # SVD (Decomposition en valeurs singulieres)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    
    # Valeurs propres = carres des valeurs singulieres
    valeurs_propres = singular_values ** 2
    
    # Nombre de composantes
    n_components = min(5, len(valeurs_propres))
    valeurs_propres = valeurs_propres[:n_components]
    
    # Inertie expliquee (pourcentage) - Section 1.7.2
    inertie_expliquee = (valeurs_propres / inertie_totale) * 100

    print("\n--- VALEURS PROPRES ET INERTIE (Section 1.7.2) ---")
    print("\nDim | Val. Propre | Inertie (%) | Cumulee (%)")
    print("-" * 55)
    cumul = 0
    vp_data = []
    for i, (vp, inertie) in enumerate(zip(valeurs_propres, inertie_expliquee)):
        cumul += inertie
        print(f" {i+1:2d} |   {vp:7.4f}   |   {inertie:6.2f}    |   {cumul:6.2f}")
        vp_data.append({
            'Dimension': i + 1,
            'Valeur_Propre': vp,
            'Inertie (%)': inertie,
            'Inertie_Cumulee (%)': cumul
        })

    vp_df = pd.DataFrame(vp_data)
    vp_df.to_csv(os.path.join(DOSSIER_TABLES, 'afc_valeurs_propres.csv'),
                 index=False, encoding='utf-8-sig')
    print(f"\n  -> Exporte: {DOSSIER_TABLES}/afc_valeurs_propres.csv")

    # Coordonnees (Cours AFC Section 1.5)
    # Coordonnees des lignes (males) : F = Di^(-1/2) * U * Sigma
    coords_lignes = Di_sqrt_inv @ U[:, :n_components] * singular_values[:n_components]
    coords_lignes = pd.DataFrame(coords_lignes, 
                                  index=ct.index,
                                  columns=[f'Dim_{i+1}' for i in range(n_components)])
    
    # Coordonnees des colonnes (females) : G = Dj^(-1/2) * V * Sigma
    coords_colonnes = Dj_sqrt_inv @ Vt[:n_components, :].T * singular_values[:n_components]
    coords_colonnes = pd.DataFrame(coords_colonnes, 
                                    index=ct.columns,
                                    columns=[f'Dim_{i+1}' for i in range(n_components)])

    coords_lignes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_coordonnees_males.csv'),
                         encoding='utf-8-sig')
    coords_colonnes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_coordonnees_females.csv'),
                           encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_coordonnees_males.csv")
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_coordonnees_females.csv")

    # Contributions (CTR) - Cours AFC Section 1.7.3
    print("\n--- CONTRIBUTIONS (CTR) - Section 1.7.3 ---")
    ctr_lignes = pd.DataFrame(index=ct.index)
    for k in range(n_components):
        ctr_lignes[f'CTR_Dim_{k+1}'] = (fi_point * coords_lignes[f'Dim_{k+1}'].values**2) / valeurs_propres[k] * 100
    
    ctr_colonnes = pd.DataFrame(index=ct.columns)
    for k in range(n_components):
        ctr_colonnes[f'CTR_Dim_{k+1}'] = (f_point_j * coords_colonnes[f'Dim_{k+1}'].values**2) / valeurs_propres[k] * 100
    
    ctr_lignes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_contributions_males.csv'),
                      encoding='utf-8-sig')
    ctr_colonnes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_contributions_females.csv'),
                        encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_contributions_males.csv")
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_contributions_females.csv")

    # Cos2 (qualite de representation) - Cours AFC Section 1.7.4
    print("\n--- QUALITE DE REPRESENTATION (COS2) - Section 1.7.4 ---")
    cos2_lignes = coords_lignes ** 2
    cos2_lignes = cos2_lignes.div(cos2_lignes.sum(axis=1), axis=0)
    cos2_lignes.columns = [f'Cos2_Dim_{i+1}' for i in range(n_components)]
    
    cos2_colonnes = coords_colonnes ** 2
    cos2_colonnes = cos2_colonnes.div(cos2_colonnes.sum(axis=1), axis=0)
    cos2_colonnes.columns = [f'Cos2_Dim_{i+1}' for i in range(n_components)]
    
    cos2_lignes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_cos2_males.csv'),
                       encoding='utf-8-sig')
    cos2_colonnes.to_csv(os.path.join(DOSSIER_TABLES, 'afc_cos2_females.csv'),
                         encoding='utf-8-sig')
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_cos2_males.csv")
    print(f"  -> Exporte: {DOSSIER_TABLES}/afc_cos2_females.csv")

    # Biplot (representation simultanee)
    plt.figure(figsize=(14, 10))

    plt.scatter(coords_lignes['Dim_1'], coords_lignes['Dim_2'],
               color='blue', s=100, alpha=0.6, edgecolors='black',
               linewidth=1, label='Parents males')
    for idx in coords_lignes.index:
        plt.annotate(idx, (coords_lignes.loc[idx, 'Dim_1'], 
                          coords_lignes.loc[idx, 'Dim_2']),
                    fontsize=9, alpha=0.8)

    plt.scatter(coords_colonnes['Dim_1'], coords_colonnes['Dim_2'],
               color='red', s=100, alpha=0.6, edgecolors='black',
               linewidth=1, marker='s', label='Parents femelles')
    for idx in coords_colonnes.index:
        plt.annotate(idx, (coords_colonnes.loc[idx, 'Dim_1'], 
                          coords_colonnes.loc[idx, 'Dim_2']),
                    fontsize=9, alpha=0.8)

    plt.xlabel(f'Dimension 1 ({inertie_expliquee[0]:.2f}%)')
    plt.ylabel(f'Dimension 2 ({inertie_expliquee[1]:.2f}%)')
    plt.title('AFC: Biplot Male x Female')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    chemin_plot = os.path.join(DOSSIER_PLOTS, 'afc_biplot.png')
    plt.savefig(chemin_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Exporte: {chemin_plot}")

    return {
        'tableau_contingence': ct,
        'chi2': chi2,
        'p_value': p_value,
        'valeurs_propres': valeurs_propres,
        'inertie_expliquee': inertie_expliquee,
        'coords_lignes': coords_lignes,
        'coords_colonnes': coords_colonnes,
        'contributions_lignes': ctr_lignes,
        'contributions_colonnes': ctr_colonnes,
        'cos2_lignes': cos2_lignes,
        'cos2_colonnes': cos2_colonnes,
        'inertie_totale': inertie_totale
    }


# =============================================================================
# 5. FONCTION PRINCIPALE
# =============================================================================

def main(chemin_fichier):
    """Execute le pipeline complet d'analyse statistique."""
    print("\n" + "="*70)
    print("   ANALYSE STATISTIQUE COMPLETE: DATASET BUTRON.MAIZE")
    print("="*70)
    print(f"Date: Decembre 2025")
    print(f"Dataset: {chemin_fichier}")
    print(f"Cours: ANAD - Ens. N. BESSAH")
    print(f"Auteurs: Bouguessa Wail & Nemamcha Oussama")
    print("="*70)

    creer_dossiers_sortie()

    # Etape 1: Chargement des donnees
    df = charger_et_preparer_donnees(chemin_fichier)
    
    # Etape 2: Statistiques descriptives (Cours Rappels)
    stats_desc = statistiques_descriptives(df)
    tracer_distributions(df)

    # Etape 3: ACM (Cours ACM1)
    modele_acm, tdc = effectuer_acm(df)
    
    # Etape 4: AFC (Cours AFC)
    resultats_afc = effectuer_afc(df)

    # Resume final
    print("\n" + "="*70)
    print("ANALYSE TERMINEE")
    print("="*70)
    print(f"\nTous les resultats ont ete sauvegardes dans: {DOSSIER_RESULTATS}/")
    print(f"  - Tables CSV: {DOSSIER_TABLES}/")
    print(f"  - Graphiques: {DOSSIER_PLOTS}/")

    print("\n--- RESUME DES RESULTATS PRINCIPAUX ---")
    print(f"  Observations: {len(df)}")
    print(f"  Variables qualitatives: gen, male, female, env")
    print(f"  Variable quantitative: yield (moyenne = {df['yield'].mean():.2f} t/ha)")
    print(f"\n  ACM:")
    print(f"    - Dimension 1: {modele_acm.percentage_of_variance_[0]:.2f}% d'inertie")
    print(f"    - Dimensions 1+2: {sum(modele_acm.percentage_of_variance_[:2]):.2f}% d'inertie")
    print(f"\n  AFC (Male x Female):")
    print(f"    - Chi2 = {resultats_afc['chi2']:.4f}, p-valeur = {resultats_afc['p_value']:.4e}")
    print(f"    - Dimension 1: {resultats_afc['inertie_expliquee'][0]:.2f}% d'inertie")

    return {
        'donnees': df,
        'descriptif': stats_desc,
        'acm': modele_acm,
        'afc': resultats_afc,
        'tdc': tdc
    }


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    chemin_donnees = "data/butron.maize.csv"
    
    if os.path.exists(chemin_donnees):
        resultats = main(chemin_donnees)
    else:
        print(f"ERREUR: Fichier non trouve: {chemin_donnees}")
        print("Veuillez verifier le chemin du fichier CSV.")
