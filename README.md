# Projet Informatique — Contrôle stochastique

Implémentation PyTorch des expériences de reproduction de l’article *Deep neural networks algorithms for stochastic control problems on finite horizon: numerical applications*.

Le projet contient cinq expériences : PDE semi-linéaire, contrôle linéaire-quadratique, couverture quadratique d’option, stockage de gaz et micro-réseau.

## Structure

```text
├── requirements.txt
├── scripts/
│   ├── smoke_test.py
│   ├── run_all.py
│   ├── exp1_semilinear_pde.py
│   ├── exp2_linear_quadratic.py
│   ├── exp3_option_hedging.py
│   ├── exp4_gas_storage.py
│   └── exp5_microgrid.py
└── src/
    ├── stochastic_control_core.py
    ├── stochastic_control_extensions.py
    └── paper_plot_utils.py
```

Les résultats sont écrits dans le dossier passé à `--out`, par défaut `results/`.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dépendances principales : `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `tqdm`, `jupyter`, `nbformat`.

## Vérifier l’installation

```bash
python scripts/smoke_test.py
```

Ce test vérifie les imports et la construction des objets principaux. Il ne produit pas de résultats numériques à interpréter.

## Modes de lancement

| Mode | Commande | Usage |
|---|---|---|
| Test minimal | `python scripts/smoke_test.py` | Vérifier l’installation. |
| Rapide | `--fast` | Produire rapidement figures et CSV. C’est le mode recommandé pour tester. |
| Complet | `--no-fast` | Budget plus élevé, plus long, plus proche des expériences du papier. |

`--fast` est activé par défaut dans les scripts. Les deux commandes suivantes sont donc équivalentes :

```bash
python scripts/run_all.py --fast --out results
python scripts/run_all.py --out results
```

## Lancer toutes les expériences

```bash
python scripts/run_all.py --fast --out results
```

Version plus coûteuse :

```bash
python scripts/run_all.py --no-fast --out results_full
```

Pour masquer les barres de progression :

```bash
python scripts/run_all.py --fast --out results --no-progress
```

## Lancer une expérience seule

```bash
python scripts/exp1_semilinear_pde.py --fast --out results
python scripts/exp2_linear_quadratic.py --fast --out results
python scripts/exp3_option_hedging.py --fast --out results
python scripts/exp4_gas_storage.py --fast --out results
python scripts/exp5_microgrid.py --fast --out results
```

Pour une exécution plus lourde, remplacer `--fast` par `--no-fast`.

## Options utiles

Options communes :

```bash
--out PATH       Dossier de sortie
--fast           Mode rapide
--no-fast        Mode plus coûteux
--progress       Afficher tqdm
--no-progress    Masquer tqdm
```

Options supplémentaires pour le stockage de gaz et le micro-réseau :

```bash
--smoke          Test très court, non interprétable numériquement
--qknn-only      Lancer uniquement Qknn et le benchmark associé
--fast-epochs    Modifier le nombre d’époques en mode rapide
--fast-batches   Modifier le nombre de mini-batchs
--fast-batch-size Modifier la taille des mini-batchs
```

Options propres à certaines expériences :

```bash
# Stockage de gaz
python scripts/exp4_gas_storage.py --fast --with-later --out results
python scripts/exp4_gas_storage.py --fast --figure-ain 0.20 --out results

# Micro-réseau
python scripts/exp5_microgrid.py --fast --fig-points 3500 --out results
```

## Sorties principales

Les scripts produisent des fichiers `.csv` et des figures `.png`, principalement dans :

```text
results/
results/figures/
```

Exemples :

```text
results/exp1_table1_gamma.csv
results/exp2_tables.csv
results/exp4_table4.csv
results/exp5_microgrid_stats.csv
results/figures/*.png
```

## Session type

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_all.py --fast --out results
```

## Remarque sur les résultats

Le mode `fast` sert surtout à reproduire les tendances qualitatives. Pour une comparaison quantitative plus fiable, utiliser `--no-fast` ou relancer plusieurs fois les expériences avec un budget plus élevé.
