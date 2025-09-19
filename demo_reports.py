#!/usr/bin/env python3
"""
Démonstration du système de génération de rapports Hercule.

Ce script montre comment utiliser le système de rapports pour analyser
les expériences d'apprentissage par renforcement.
"""

import sys
from pathlib import Path


# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hercule.reports import generate_report


def find_experiments(base_path: Path) -> list[Path]:
    """Trouve tous les répertoires d'expériences dans le dossier outputs."""
    experiments = []

    if not base_path.exists():
        print(f"❌ Le dossier {base_path} n'existe pas")
        return experiments

    # Chercher récursivement tous les répertoires contenant les fichiers JSON requis
    for path in base_path.rglob("*"):
        if path.is_dir():
            # Vérifier si ce répertoire contient les fichiers JSON requis
            if (
                (path / "environment.json").exists()
                and (path / "model.json").exists()
                and (path / "run_info.json").exists()
            ):
                experiments.append(path)

    return experiments


def main():
    """Démonstration du système de rapports."""
    print("🔬 Démonstration du système de rapports Hercule")
    print("=" * 50)

    # Chercher les expériences disponibles
    outputs_path = Path("outputs/simple_games/simple_games")
    experiments = find_experiments(outputs_path)

    if not experiments:
        print("❌ Aucune expérience trouvée dans outputs/simple_games/simple_games/")
        print("   Assurez-vous d'avoir exécuté des expériences avec Hercule")
        return

    print(f"📊 {len(experiments)} expérience(s) trouvée(s):")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp.name}")

    print("\n🚀 Génération des rapports...")

    # Générer un rapport pour chaque expérience
    for i, experiment_path in enumerate(experiments, 1):
        print(f"\n📝 Génération du rapport {i}/{len(experiments)}: {experiment_path.name}")

        try:
            report_path = generate_report(experiment_path)
            print(f"   ✅ Rapport généré: {report_path}")

        except Exception as e:
            print(f"   ❌ Erreur: {e}")

    print("\n🎉 Démonstration terminée!")
    print("\n📖 Pour utiliser les rapports:")
    print("   1. Ouvrez un notebook Jupyter")
    print("   2. Exécutez les fichiers report.py générés")
    print("   3. Les rapports afficheront les visualisations et analyses")

    print("\n💡 Commandes utiles:")
    print("   # Générer un rapport spécifique")
    print("   poetry run python -m hercule.reports.cli generate <chemin_experience>")
    print("   ")
    print("   # Générer avec sortie personnalisée")
    print("   poetry run python -m hercule.reports.cli generate <chemin> -o <sortie.py>")


if __name__ == "__main__":
    main()
