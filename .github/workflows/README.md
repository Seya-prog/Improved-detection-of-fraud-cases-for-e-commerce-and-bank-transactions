# GitHub Workflows

This directory contains GitHub Actions workflows for the project.

**Project Status: Complete**

All workflows are up-to-date and support robust CI/CD for the entire fraud detection pipeline, including model explainability.

## Workflows

### 1. Python CI (`python-ci.yml`)

This workflow runs on every push and pull request to the main branch. It:
- Sets up Python environments (3.9 and 3.10)
- Installs dependencies
- Runs linting checks with flake8
- Runs tests with pytest and generates coverage reports

### 2. Data Validation (`data-validation.yml`)

This workflow runs whenever data files or data processing scripts change. It:
- Sets up Python
- Installs dependencies
- Validates data schemas
- Generates and uploads a data validation report

### 3. Dependabot (`dependabot.yml`)

This is a configuration file for Dependabot, which:
- Automatically checks for outdated dependencies weekly
- Creates pull requests to update dependencies
- Labels these PRs as "dependencies" and "security"

## Usage

These workflows run automatically based on their triggers. You can also manually trigger the data validation workflow from the Actions tab in GitHub.

## Artifacts

- All model evaluation results, explainability plots, and reports are available in the `results/` directory after workflow runs. 