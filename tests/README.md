
# HTL Pairwise Ranking Graphormer - Unit Tests

This directory contains unit tests for the HTL Pairwise Ranking Graphormer project.

## Test File Descriptions

### test_config.py
Tests configuration classes (ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig):
- Default value settings
- Custom value settings
- Dictionary conversion
- Dictionary creation

### test_features.py
Tests molecular feature extraction functionality:
- One-hot encoding
- Atom feature extraction
- Bond feature extraction
- Molecular featurization
- Distance matrix computation
- MolGraphData class

### test_model.py
Tests model architecture components:
- EdgeEncoding module
- GraphormerLayer module
- GraphormerEncoder module
- HTLRankingModel model
- collate_mol_graphs function

### test_loss_and_dataset.py
Tests loss functions and dataset classes:
- MultiTaskRankingLoss loss function
- PairDataset dataset
- ListDataset dataset
- CachedPairDataset dataset
- DynamicPairDataset dataset

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_config.py
python -m pytest tests/test_features.py
python -m pytest tests/test_model.py
python -m pytest tests/test_loss_and_dataset.py
```

### Run specific test class
```bash
python -m pytest tests/test_config.py::TestModelConfig
python -m pytest tests/test_features.py::TestFeaturizeMol
```

### Run specific test method
```bash
python -m pytest tests/test_config.py::TestModelConfig::test_default_values
```

### Run with unittest
```bash
python -m unittest discover tests
```

### Run specific test file (using unittest)
```bash
python -m unittest tests.test_config
python -m unittest tests.test_features
python -m unittest tests.test_model
python -m unittest tests.test_loss_and_dataset
```

## Test Coverage

To view test coverage, you can use pytest-cov:

```bash
pip install pytest-cov
pytest --cov=htl_ranking_graphormer tests/
```

## Notes

1. Make sure all required dependencies are installed:
   - torch
   - numpy
   - pandas
   - rdkit
   - scikit-learn
   - scipy
   - matplotlib

2. Some tests may take a long time to run, especially those involving model training and inference.

3. Tests use randomly generated data, so test results may vary slightly, but should always pass.

4. Tests run on CPU and do not require GPU support.
