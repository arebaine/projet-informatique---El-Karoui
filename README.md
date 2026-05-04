# Stochastic control — package complet amélioré

Ce package regroupe :
- le code source complet (`src/`) ;
- des scripts par section du papier (`scripts/`) ;
- un notebook nettoyé exécutable localement (`notebooks/main_reproduction_clean.ipynb`) ;
- un notebook driver pour générer rapidement les figures (`notebooks/paper_figures_driver.ipynb`) ;
- des figures de référence extraites du PDF (`results/reference_figures/`) ;
- les figures générées par les scripts dans `results/figures/`.


## Correctifs importants (2026-04-27)

Les expériences `exp4_gas_storage.py` et `exp5_microgrid.py` ont été corrigées. Le bug principal venait de Qknn : les quantifieurs utilisés étaient ceux de `N(0,1)` alors que les dynamiques du papier utilisent des bruits déjà multipliés par `sigma_p` ou `sigma_R`. Les scripts utilisent maintenant `scale_quantizer()`.

Les sorties `results/` déjà présentes dans l'ancien zip peuvent être anciennes ; pour comparer proprement, régénère les sections 3.4 et 3.5 avec :

```bash
python scripts/exp4_gas_storage.py --fast --out results_corrected
python scripts/exp5_microgrid.py --fast --out results_corrected
```

Voir aussi `CORRECTIONS.md`.

## Installation

```bash
cd stochastic_control_project_full_package
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Génération des figures du papier

Chaque script génère les figures de la section correspondante :

```bash
python scripts/exp1_semilinear_pde.py --fast --out results
python scripts/exp2_linear_quadratic.py --fast --out results
python scripts/exp3_option_hedging.py --fast --out results
python scripts/exp4_gas_storage.py --fast --out results
python scripts/exp5_microgrid.py --fast --out results
```

Ou tout lancer d'un coup :

```bash
python scripts/run_all.py --fast --out results
```

## Correspondance des figures

- Figure 1–2 : `exp1_semilinear_pde.py`
- Figure 3–5 : `exp2_linear_quadratic.py`
- Figure 6–7 : `exp3_option_hedging.py`
- Figure 8–11 : `exp4_gas_storage.py`
- Figure 12–14 : `exp5_microgrid.py`

Les fichiers produits sont écrits dans `results/figures/`.

## Références extraites du PDF

Le dossier `results/reference_figures/` contient des extractions des figures du PDF fourni pour comparaison visuelle. Elles ne sont pas régénérées numériquement : ce sont des références.

Pour les recréer depuis un autre PDF :

```bash
python scripts/extract_reference_figures.py "chemin/vers/article.pdf" --out results/reference_figures
```

## Notes

- Le mode `--fast` donne des graphes de même nature que le papier, avec un budget calcul réduit.
- Le mode complet (`--no-fast`) est plus proche des budgets du papier, mais beaucoup plus lent.
- Le notebook `main_reproduction_clean.ipynb` vient de ton ancien notebook, nettoyé pour un usage local sans Colab.
