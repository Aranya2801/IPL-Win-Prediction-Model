# Contributing to IPL Win Prediction Model

Thank you for contributing! Please read this guide before submitting a PR.

## Development Setup

```bash
git clone https://github.com/Aranya2801/IPL-Win-Prediction-Model.git
cd IPL-Win-Prediction-Model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Code Standards
- **Black** for formatting (`black src/ tests/`)
- **isort** for import ordering (`isort src/ tests/`)
- **flake8** for linting (max line length 100)
- **mypy** for type checking
- **pytest** for tests with ≥ 70% coverage

## Branch Strategy
- `main` — stable, production-ready
- `develop` — integration branch
- `feature/*` — new features
- `fix/*` — bug fixes
- `experiment/*` — ML experiments

## Pull Request Checklist
- [ ] Tests pass (`pytest`)
- [ ] Coverage ≥ 70%
- [ ] Black & isort applied
- [ ] Docstrings on new functions
- [ ] CHANGELOG.md updated

## Adding a New Model
1. Subclass or wrap your model in `src/models/`
2. Add it to the ensemble in `train_model.py`
3. Add corresponding tests in `tests/`
4. Document hyperparameters in `docs/`

## Reporting Issues
Use the GitHub issue templates. Include:
- Python version
- OS
- Minimal reproducible example
- Expected vs actual behaviour

## Code of Conduct
Be respectful, constructive, and inclusive. ❤️
