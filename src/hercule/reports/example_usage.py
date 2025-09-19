#!/usr/bin/env python3
"""
Example script showing how to use the report generation system.
"""

import sys
from pathlib import Path


# Add the src directory to the path so we can import hercule modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hercule.reports import generate_report


def main():
    """Example usage of the report generation system."""

    # Example experiment path (adjust this to your actual experiment path)
    experiment_path = Path(
        "outputs/simple_games/simple_games/FrozenLake-v1/is_sli_True__map_nam_4x4__max_epi_ste_200/simple_q_learning/dis_fac_0.8__eps_0.77__lea_rat_0.2"
    )

    # Check if the experiment path exists
    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        print("Please adjust the path in this script to point to a valid experiment directory.")
        return

    try:
        # Generate the report
        print(f"Generating report for experiment: {experiment_path}")
        report_path = generate_report(experiment_path)
        print(f"Report generated successfully: {report_path}")

        # Instructions for running the report
        print("\nTo run the generated report:")
        print("1. Open a Jupyter notebook")
        print(f"2. Run the cells in: {report_path}")
        print("3. The report will display visualizations and analysis")

    except Exception as e:
        print(f"Error generating report: {e}")
        return


if __name__ == "__main__":
    main()
