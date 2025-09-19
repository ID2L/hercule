# Système de Rapports Hercule

Ce document explique comment utiliser le système de génération de rapports pour analyser les expériences d'apprentissage par renforcement avec Hercule.

## 🎯 Vue d'ensemble

Le système de rapports génère automatiquement des analyses complètes des expériences, incluant :
- **Configuration de l'expérience** (environnement, modèle, hyperparamètres)
- **Visualisations d'apprentissage** (évolution des récompenses, étapes, moyennes mobiles)
- **Évaluation finale** (boxplots, statistiques de performance)
- **Analyse de performance** (comparaison apprentissage vs test, généralisation)

## 🚀 Utilisation rapide

### 1. Génération automatique de tous les rapports

```bash
# Exécuter le script de démonstration
poetry run python demo_reports.py
```

### 2. Génération d'un rapport spécifique

```bash
# Générer un rapport pour une expérience spécifique
poetry run python -m hercule.reports.cli generate "chemin/vers/experience"

# Avec sortie personnalisée
poetry run python -m hercule.reports.cli generate "chemin/vers/experience" -o "mon_rapport.py"
```

### 3. Utilisation programmatique

```python
from hercule.reports import generate_report
from pathlib import Path

# Générer un rapport
experiment_path = Path("outputs/simple_games/simple_games/FrozenLake-v1/...")
report_path = generate_report(experiment_path)
print(f"Rapport généré: {report_path}")
```

## 📊 Contenu des rapports

### Structure des fichiers requis

Chaque expérience doit contenir :
- `environment.json` : Configuration de l'environnement
- `model.json` : État du modèle entraîné
- `run_info.json` : Métriques d'apprentissage et d'évaluation

### Sections du rapport

1. **Vue d'ensemble de l'expérience**
   - Configuration de l'environnement
   - Configuration du modèle
   - Informations d'entraînement

2. **Visualisations d'apprentissage**
   - Évolution des récompenses dans le temps
   - Évolution du nombre d'étapes
   - Moyennes mobiles pour l'analyse des tendances
   - Histogrammes de distribution

3. **Évaluation finale**
   - Boxplots des performances de test
   - Statistiques de performance
   - Taux de succès

4. **Analyse de performance**
   - Comparaison apprentissage vs test
   - Analyse de la courbe d'apprentissage
   - Évaluation de la généralisation

## 🛠️ Installation et dépendances

Le système de rapports nécessite les dépendances suivantes :

```bash
# Installer les dépendances
poetry add jinja2 matplotlib pandas

# Ou installer toutes les dépendances du projet
poetry install
```

## 📁 Structure des fichiers

```
src/hercule/reports/
├── __init__.py                 # Fonctions principales de génération
├── cli.py                     # Commandes CLI
├── templates/
│   └── report_template.py.j2  # Template Jinja2 pour les rapports
├── example_usage.py           # Exemple d'utilisation
└── README.md                  # Documentation du module
```

## 🎨 Personnalisation

### Modifier le template de rapport

1. Éditez `src/hercule/reports/templates/report_template.py.j2`
2. Utilisez la syntaxe Jinja2 pour personnaliser le contenu
3. Régénérez les rapports avec votre template

### Ajouter de nouvelles visualisations

```python
# Dans le template Jinja2
{% if learning_rewards %}
# Votre nouvelle visualisation
plt.figure(figsize=(10, 6))
# ... code de visualisation ...
plt.show()
{% endif %}
```

## 🔧 Dépannage

### Problèmes courants

1. **Fichiers JSON manquants**
   ```
   ❌ Error: Failed to load experiment data
   ```
   **Solution** : Vérifiez que le répertoire contient `environment.json`, `model.json`, et `run_info.json`

2. **Dépendances manquantes**
   ```
   ModuleNotFoundError: No module named 'matplotlib'
   ```
   **Solution** : Installez les dépendances avec `poetry install`

3. **Erreurs de template**
   ```
   jinja2.exceptions.TemplateSyntaxError
   ```
   **Solution** : Vérifiez la syntaxe Jinja2 dans le template

### Vérification de l'installation

```bash
# Tester l'installation
poetry run python -c "from hercule.reports import generate_report; print('✅ Installation OK')"
```

## 📈 Exemples d'utilisation

### Analyse comparative de modèles

```python
from hercule.reports import generate_report
from pathlib import Path

# Générer des rapports pour différents modèles
models = ["simple_q_learning", "simple_sarsa"]
for model in models:
    exp_path = Path(f"outputs/experiment/{model}")
    if exp_path.exists():
        report_path = generate_report(exp_path)
        print(f"Rapport {model}: {report_path}")
```

### Intégration dans un pipeline d'analyse

```python
import glob
from hercule.reports import generate_report

# Générer des rapports pour toutes les expériences
experiment_dirs = glob.glob("outputs/**/", recursive=True)
for exp_dir in experiment_dirs:
    exp_path = Path(exp_dir)
    if (exp_path / "run_info.json").exists():
        generate_report(exp_path)
```

## 🎯 Prochaines étapes

- [ ] Support pour d'autres formats de sortie (HTML, PDF)
- [ ] Comparaisons entre expériences
- [ ] Rapports automatisés dans le pipeline CI/CD
- [ ] Intégration avec des outils de visualisation avancés

---

*Système de rapports Hercule - Génération automatique d'analyses d'expériences RL*
