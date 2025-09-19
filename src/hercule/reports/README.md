# Hercule Reports Module

This module provides automatic report generation for reinforcement learning experiments conducted with Hercule.

## Features

- **Automatic Data Loading**: Loads experiment data from JSON files (environment.json, model.json, run_info.json)
- **Rich Visualizations**: Creates learning progress plots, evaluation boxplots, and performance analysis
- **Jupyter Integration**: Generates Python scripts compatible with Jupyter notebooks
- **Template-based**: Uses Jinja2 templates for flexible report customization

## Usage

### Basic Usage

```python
from hercule.reports import generate_report
from pathlib import Path

# Path to your experiment directory containing JSON files
experiment_path = Path("path/to/your/experiment")

# Generate report
report_path = generate_report(experiment_path)
print(f"Report generated: {report_path}")
```

### Running the Generated Report

1. Open a Jupyter notebook
2. Run the generated Python file cell by cell
3. The report will display:
   - Experiment configuration
   - Learning progress visualizations
   - Final model evaluation
   - Performance analysis

## Report Contents

The generated report includes:

### 1. Experiment Overview
- Environment configuration
- Model configuration  
- Training information

### 2. Learning Progress Visualization
- Reward evolution over time
- Steps evolution over time
- Moving averages for trend analysis
- Distribution histograms

### 3. Final Model Evaluation
- Boxplots for testing rewards and steps
- Statistical summaries
- Success rate analysis

### 4. Performance Analysis
- Learning vs testing performance comparison
- Learning curve analysis
- Generalization assessment

## File Structure

```
src/hercule/reports/
├── __init__.py                 # Main report generation functions
├── templates/
│   └── report_template.py.j2  # Jinja2 template for reports
├── example_usage.py          # Example usage script
└── README.md                  # This file
```

## Dependencies

The reports module requires:
- `jinja2` for template rendering
- `matplotlib` for visualizations
- `pandas` for data analysis
- `numpy` for numerical operations

## Example Output

The generated report will create a Python script that, when run in Jupyter, produces:

1. **Text sections** with experiment details and statistics
2. **Interactive plots** showing learning progress and evaluation results
3. **Analysis** comparing learning vs testing performance
4. **Conclusions** about model performance and generalization

## Customization

To customize the report template:

1. Edit `templates/report_template.py.j2`
2. Modify the Jinja2 template syntax as needed
3. Add new visualizations or analysis sections
4. Regenerate reports with your custom template

## Troubleshooting

### Common Issues

1. **Missing JSON files**: Ensure your experiment directory contains:
   - `environment.json`
   - `model.json` 
   - `run_info.json`

2. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install jinja2 matplotlib pandas numpy
   ```

3. **Template errors**: Check that the Jinja2 template syntax is correct

### Getting Help

If you encounter issues:
1. Check that your experiment path contains the required JSON files
2. Verify that the JSON files are properly formatted
3. Ensure all dependencies are installed
4. Check the logs for specific error messages
