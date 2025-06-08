# API Reference

## Overview

This document provides detailed API documentation for all modules and classes in the XAI Intrusion Detection System.

## Core Modules

### src.data_loader

#### DataLoader Class

**Class:** `DataLoader`

Main class for loading and preprocessing network intrusion data.

**Constructor:**
```python
DataLoader(config_path: str = None, random_state: int = 42)
```

**Parameters:**
- `config_path` (str, optional): Path to data configuration YAML file
- `random_state` (int): Random seed for reproducibility

**Methods:**

##### load_data()
```python
load_data(file_path: str, file_format: str = 'csv') -> pd.DataFrame
```
Load data from various file formats.

**Parameters:**
- `file_path` (str): Path to the data file
- `file_format` (str): Format of the file ('csv', 'parquet', 'json')

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
loader = DataLoader()
data = loader.load_data('data/raw/network_data.csv')
```

##### preprocess()
```python
preprocess(data: pd.DataFrame, fit_preprocessor: bool = True) -> Tuple[np.ndarray, np.ndarray]
```
Preprocess the dataset with scaling, encoding, and feature engineering.

**Parameters:**
- `data` (pd.DataFrame): Raw dataset
- `fit_preprocessor` (bool): Whether to fit preprocessor on this data

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Processed features (X) and target (y)

##### train_test_split()
```python
train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, ...]
```
Split data into training and testing sets.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Target vector
- `test_size` (float): Proportion of test data

**Returns:**
- `Tuple`: X_train, X_test, y_train, y_test

##### load_and_split()
```python
load_and_split(file_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, ...]
```
Convenience method to load, preprocess, and split data in one call.

**Parameters:**
- `file_path` (str): Path to data file
- `test_size` (float): Test set proportion

**Returns:**
- `Tuple`: X_train, X_test, y_train, y_test

---

### src.model_trainer

#### ModelTrainer Class

**Class:** `ModelTrainer`

Handles training and evaluation of machine learning models.

**Constructor:**
```python
ModelTrainer(config_path: str = None, random_state: int = 42)
```

**Methods:**

##### train()
```python
train(X_train: np.ndarray, y_train: np.ndarray, algorithm: str = 'random_forest', 
      hyperparameters: dict = None) -> Any
```
Train a machine learning model.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `algorithm` (str): Algorithm name ('random_forest', 'gradient_boosting', 'svm', 'neural_network', 'xgboost', 'lightgbm')
- `hyperparameters` (dict, optional): Model hyperparameters

**Returns:**
- Trained model object

**Example:**
```python
trainer = ModelTrainer()
model = trainer.train(X_train, y_train, algorithm='random_forest', 
                     hyperparameters={'n_estimators': 200})
```

##### evaluate()
```python
evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]
```
Evaluate model performance on test data.

**Parameters:**
- `model`: Trained model
- `X_test` (np.ndarray): Test features
- `y_test` (np.ndarray): Test labels

**Returns:**
- `Dict[str, float]`: Performance metrics

##### hyperparameter_tuning()
```python
hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, algorithm: str,
                     param_grid: dict, cv_folds: int = 5) -> Tuple[Any, dict]
```
Perform hyperparameter optimization using grid search.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `algorithm` (str): Algorithm name
- `param_grid` (dict): Parameter grid for search
- `cv_folds` (int): Number of cross-validation folds

**Returns:**
- `Tuple[Any, dict]`: Best model and best parameters

##### save_model()
```python
save_model(model: Any, model_path: str) -> None
```
Save trained model to disk.

##### load_model()
```python
load_model(model_path: str) -> Any
```
Load saved model from disk.

---

### src.explainer

#### XAIExplainer Class

**Class:** `XAIExplainer`

Provides explainable AI capabilities using SHAP and LIME.

**Constructor:**
```python
XAIExplainer(model: Any, feature_names: List[str] = None)
```

**Parameters:**
- `model`: Trained model to explain
- `feature_names` (List[str], optional): Names of features

**Methods:**

##### explain_shap()
```python
explain_shap(X: np.ndarray, max_evals: int = 1000) -> np.ndarray
```
Generate SHAP explanations for given instances.

**Parameters:**
- `X` (np.ndarray): Input instances
- `max_evals` (int): Maximum evaluations for SHAP explainer

**Returns:**
- `np.ndarray`: SHAP values

**Example:**
```python
explainer = XAIExplainer(model, feature_names=feature_names)
shap_values = explainer.explain_shap(X_test[:100])
```

##### explain_lime()
```python
explain_lime(X: np.ndarray, num_features: int = 10, num_samples: int = 5000) -> List
```
Generate LIME explanations for given instances.

**Parameters:**
- `X` (np.ndarray): Input instances
- `num_features` (int): Number of features to include in explanation
- `num_samples` (int): Number of samples for LIME

**Returns:**
- `List`: LIME explanation objects

##### plot_summary()
```python
plot_summary(shap_values: np.ndarray, X: np.ndarray, max_display: int = 20, 
            save_path: str = None) -> None
```
Create SHAP summary plot.

**Parameters:**
- `shap_values` (np.ndarray): SHAP values
- `X` (np.ndarray): Input data
- `max_display` (int): Maximum features to display
- `save_path` (str, optional): Path to save plot

##### plot_waterfall()
```python
plot_waterfall(shap_values: np.ndarray, instance: np.ndarray, save_path: str = None) -> None
```
Create SHAP waterfall plot for single instance.

##### plot_force()
```python
plot_force(shap_values: np.ndarray, instance: np.ndarray, save_path: str = None) -> None
```
Create SHAP force plot for single instance.

##### plot_lime_explanation()
```python
plot_lime_explanation(explanation: Any, save_path: str = None) -> None
```
Visualize LIME explanation.

---

### src.real_time_detector

#### RealTimeDetector Class

**Class:** `RealTimeDetector`

Real-time intrusion detection with explanations.

**Constructor:**
```python
RealTimeDetector(model_path: str, preprocessor_path: str = None, threshold: float = 0.5)
```

**Parameters:**
- `model_path` (str): Path to trained model
- `preprocessor_path` (str, optional): Path to fitted preprocessor
- `threshold` (float): Decision threshold

**Methods:**

##### predict()
```python
predict(X: np.ndarray) -> np.ndarray
```
Make predictions on input data.

**Parameters:**
- `X` (np.ndarray): Input features

**Returns:**
- `np.ndarray`: Predictions

##### predict_proba()
```python
predict_proba(X: np.ndarray) -> np.ndarray
```
Get prediction probabilities.

**Parameters:**
- `X` (np.ndarray): Input features

**Returns:**
- `np.ndarray`: Prediction probabilities

##### predict_with_explanation()
```python
predict_with_explanation(X: np.ndarray, explain_method: str = 'shap') -> Tuple[np.ndarray, np.ndarray, Any]
```
Make predictions with explanations.

**Parameters:**
- `X` (np.ndarray): Input features
- `explain_method` (str): Explanation method ('shap' or 'lime')

**Returns:**
- `Tuple`: Predictions, probabilities, explanations

##### predict_batch()
```python
predict_batch(X: np.ndarray, batch_size: int = 1000) -> np.ndarray
```
Process large batches efficiently.

**Example:**
```python
detector = RealTimeDetector('models/trained_models/best_model.pkl')
predictions, probabilities, explanations = detector.predict_with_explanation(X_sample)
```

---

### src.adversarial_analysis

#### AdversarialAnalyzer Class

**Class:** `AdversarialAnalyzer`

Analyzes model robustness against adversarial attacks.

**Constructor:**
```python
AdversarialAnalyzer(model: Any, feature_bounds: dict = None)
```

**Parameters:**
- `model`: Trained model to analyze
- `feature_bounds` (dict, optional): Feature value bounds

**Methods:**

##### generate_adversarial_samples()
```python
generate_adversarial_samples(X: np.ndarray, method: str = 'fgsm', epsilon: float = 0.1) -> np.ndarray
```
Generate adversarial samples using specified attack method.

**Parameters:**
- `X` (np.ndarray): Original samples
- `method` (str): Attack method ('fgsm', 'pgd', 'c&w')
- `epsilon` (float): Perturbation magnitude

**Returns:**
- `np.ndarray`: Adversarial samples

##### evaluate_robustness()
```python
evaluate_robustness(X_original: np.ndarray, X_adversarial: np.ndarray) -> Dict[str, float]
```
Evaluate model robustness metrics.

**Parameters:**
- `X_original` (np.ndarray): Original samples
- `X_adversarial` (np.ndarray): Adversarial samples

**Returns:**
- `Dict[str, float]`: Robustness metrics

##### defense_analysis()
```python
defense_analysis(X: np.ndarray, defense_methods: List[str] = None) -> Dict[str, Any]
```
Analyze effectiveness of defense methods.

---

### src.security_analyzer

#### SecurityAnalyzer Class

**Class:** `SecurityAnalyzer`

Generates security insights and threat intelligence.

**Constructor:**
```python
SecurityAnalyzer(model: Any, explainer: XAIExplainer)
```

**Methods:**

##### analyze_attack_patterns()
```python
analyze_attack_patterns(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]
```
Analyze patterns in attack data.

**Parameters:**
- `X` (np.ndarray): Feature data
- `y` (np.ndarray): Labels

**Returns:**
- `Dict[str, Any]`: Attack pattern analysis

##### generate_threat_report()
```python
generate_threat_report(predictions: np.ndarray, explanations: Any, 
                      save_path: str = None) -> Dict[str, Any]
```
Generate comprehensive threat analysis report.

##### feature_importance_analysis()
```python
feature_importance_analysis(shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]
```
Analyze which features are most important for detection.

---

## Utility Functions

### src.utils

#### Logging Functions

##### setup_logging()
```python
setup_logging(log_level: str = 'INFO', log_file: str = None) -> None
```
Configure logging for the application.

**Parameters:**
- `log_level` (str): Logging level
- `log_file` (str, optional): Log file path

#### Performance Monitoring

##### PerformanceMonitor Class
```python
class PerformanceMonitor:
    def timer(self, name: str) -> ContextManager
    def get_elapsed_time(self, name: str) -> float
    def memory_usage(self) -> Dict[str, float]
```

#### Data Validation

##### validate_data()
```python
validate_data(X: np.ndarray, y: np.ndarray = None) -> bool
```
Validate input data format and consistency.

##### check_data_drift()
```python
check_data_drift(X_reference: np.ndarray, X_current: np.ndarray, 
                threshold: float = 0.05) -> Dict[str, Any]
```
Detect data drift between reference and current datasets.

## Configuration Schema

### Model Configuration
```yaml
model:
  algorithm: str  # Model algorithm
  hyperparameters: dict  # Algorithm-specific parameters
  
training:
  test_size: float
  validation_split: float
  cross_validation_folds: int
  
explainability:
  methods: list  # ['shap', 'lime']
  sample_size: int
  plot_types: list
```

### Data Configuration
```yaml
data:
  input_path: str
  output_path: str
  features:
    categorical: list
    numerical: list
    target: str
    
preprocessing:
  scaling: str  # 'standard', 'minmax', 'robust'
  encoding: str  # 'one_hot', 'label', 'target'
  handle_missing: str  # 'median', 'mean', 'mode', 'drop'
  outlier_detection: bool
```

## Error Handling

All methods include proper error handling and raise appropriate exceptions:

- `ValueError`: Invalid input parameters
- `FileNotFoundError`: Missing files or paths
- `RuntimeError`: Execution errors
- `NotImplementedError`: Unsupported methods or algorithms

## Examples

### Complete Workflow Example
```python
from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer
from src.explainer import XAIExplainer
from src.real_time_detector import RealTimeDetector

# Load and preprocess data
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.load_and_split('data/raw/network_data.csv')

# Train model
trainer = ModelTrainer()
model = trainer.train(X_train, y_train, algorithm='random_forest')

# Generate explanations
explainer = XAIExplainer(model)
shap_values = explainer.explain_shap(X_test[:100])
explainer.plot_summary(shap_values, X_test[:100])

# Setup real-time detection
trainer.save_model(model, 'models/trained_models/rf_model.pkl')
detector = RealTimeDetector('models/trained_models/rf_model.pkl
