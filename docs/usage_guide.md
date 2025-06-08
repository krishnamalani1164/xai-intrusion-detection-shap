# Usage Guide

## Quick Start

### Basic Usage

1. **Prepare your data**
   ```python
   from src.data_loader import DataLoader
   
   loader = DataLoader()
   X_train, X_test, y_train, y_test = loader.load_and_split('data/raw/network_data.csv')
   ```

2. **Train a model**
   ```python
   from src.model_trainer import ModelTrainer
   
   trainer = ModelTrainer()
   model = trainer.train(X_train, y_train, algorithm='random_forest')
   ```

3. **Generate explanations**
   ```python
   from src.explainer import XAIExplainer
   
   explainer = XAIExplainer(model)
   shap_values = explainer.explain_shap(X_test[:100])
   explainer.plot_summary(shap_values, X_test[:100])
   ```

### Command Line Interface

Run the complete pipeline with default settings:
```bash
python src/main.py --config configs/model_config.yaml
```

## Configuration

### YAML Configuration Files

**Model Configuration (`configs/model_config.yaml`):**
```yaml
model:
  algorithm: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
training:
  test_size: 0.2
  validation_split: 0.1
  cross_validation_folds: 5

explainability:
  methods: ["shap", "lime"]
  sample_size: 1000
  plot_types: ["summary", "waterfall", "force"]
```

**Data Configuration (`configs/data_config.yaml`):**
```yaml
data:
  input_path: "data/raw/"
  output_path: "data/processed/"
  features:
    categorical: ["protocol", "service", "flag"]
    numerical: ["duration", "src_bytes", "dst_bytes"]
    target: "label"
  
preprocessing:
  scaling: "standard"
  encoding: "one_hot"
  handle_missing: "median"
  outlier_detection: true
```

## Core Components

### 1. Data Loading and Preprocessing

```python
from src.data_loader import DataLoader

# Initialize data loader
loader = DataLoader(config_path='configs/data_config.yaml')

# Load data
data = loader.load_data('data/raw/kdd_cup.csv')

# Preprocess data
X_processed, y = loader.preprocess(data)

# Split data
X_train, X_test, y_train, y_test = loader.train_test_split(X_processed, y)
```

**Supported Data Formats:**
- CSV files
- Parquet files
- JSON files
- Numpy arrays
- Pandas DataFrames

### 2. Model Training

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Available algorithms
algorithms = [
    'random_forest', 'gradient_boosting', 'svm', 
    'neural_network', 'xgboost', 'lightgbm'
]

# Train model
model = trainer.train(
    X_train, y_train,
    algorithm='random_forest',
    hyperparameters={'n_estimators': 200, 'max_depth': 15}
)

# Evaluate model
metrics = trainer.evaluate(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### 3. Explainable AI

#### SHAP Explanations
```python
from src.explainer import XAIExplainer

explainer = XAIExplainer(model)

# Generate SHAP values
shap_values = explainer.explain_shap(X_test)

# Create visualizations
explainer.plot_summary(shap_values, X_test)
explainer.plot_waterfall(shap_values[0], X_test.iloc[0])
explainer.plot_force(shap_values[0], X_test.iloc[0])
```

#### LIME Explanations
```python
# Generate LIME explanations
lime_explanations = explainer.explain_lime(X_test[:10])

# Visualize explanations
for i, exp in enumerate(lime_explanations):
    explainer.plot_lime_explanation(exp, save_path=f'outputs/figures/lime_{i}.html')
```

### 4. Real-time Detection

```python
from src.real_time_detector import RealTimeDetector

# Initialize detector
detector = RealTimeDetector(model_path='models/trained_models/best_model.pkl')

# Single prediction
sample_data = X_test.iloc[0].values.reshape(1, -1)
prediction, confidence, explanation = detector.predict_with_explanation(sample_data)

print(f"Prediction: {'Intrusion' if prediction[0] == 1 else 'Normal'}")
print(f"Confidence: {confidence[0]:.4f}")

# Batch prediction
batch_predictions = detector.predict_batch(X_test[:100])
```

### 5. Adversarial Analysis

```python
from src.adversarial_analysis import AdversarialAnalyzer

analyzer = AdversarialAnalyzer(model)

# Test adversarial robustness
adversarial_samples = analyzer.generate_adversarial_samples(
    X_test[:50], 
    method='fgsm',
    epsilon=0.1
)

# Evaluate robustness
robustness_metrics = analyzer.evaluate_robustness(
    X_test[:50], 
    adversarial_samples
)

print(f"Adversarial Accuracy: {robustness_metrics['adversarial_accuracy']:.4f}")
```

## Advanced Usage

### Custom Feature Engineering

```python
from src.data_loader import DataLoader

class CustomDataLoader(DataLoader):
    def custom_feature_engineering(self, data):
        # Add custom features
        data['packet_rate'] = data['src_bytes'] / (data['duration'] + 1)
        data['bytes_ratio'] = data['src_bytes'] / (data['dst_bytes'] + 1)
        return data

loader = CustomDataLoader()
```

### Model Ensemble

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Train multiple models
models = {}
algorithms = ['random_forest', 'gradient_boosting', 'xgboost']

for algo in algorithms:
    models[algo] = trainer.train(X_train, y_train, algorithm=algo)

# Create ensemble
ensemble_predictions = trainer.ensemble_predict(models, X_test)
```

### Custom Explainability

```python
from src.explainer import XAIExplainer

class CustomExplainer(XAIExplainer):
    def explain_custom_method(self, X):
        # Implement custom explanation method
        explanations = []
        for instance in X:
            # Custom explanation logic
            explanation = self.custom_explanation_logic(instance)
            explanations.append(explanation)
        return explanations

explainer = CustomExplainer(model)
```

## Workflow Examples

### End-to-End Pipeline

```python
from src.main import XAIIntrusionDetectionPipeline

# Initialize pipeline
pipeline = XAIIntrusionDetectionPipeline(
    config_path='configs/model_config.yaml'
)

# Run complete pipeline
results = pipeline.run(
    data_path='data/raw/network_data.csv',
    output_dir='outputs/'
)

# Access results
print(f"Best model: {results['best_model']}")
print(f"Test accuracy: {results['test_accuracy']:.4f}")
```

### Hyperparameter Tuning

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
best_model, best_params = trainer.hyperparameter_tuning(
    X_train, y_train,
    algorithm='random_forest',
    param_grid=param_grid,
    cv_folds=5
)
```

## Monitoring and Logging

### Enable Logging

```python
import logging
from src.utils import setup_logging

# Setup logging
setup_logging(
    log_level='INFO',
    log_file='outputs/logs/xai_ids.log'
)

logger = logging.getLogger(__name__)
logger.info("Starting XAI-IDS pipeline")
```

### Performance Monitoring

```python
from src.utils import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.timer('model_training'):
    model = trainer.train(X_train, y_train)

print(f"Training time: {monitor.get_elapsed_time('model_training'):.2f} seconds")
```

## Output Files

The system generates various output files:

- **Models**: Saved in `models/trained_models/`
- **Visualizations**: Saved in `outputs/figures/`
- **Reports**: Saved in `outputs/reports/`
- **Logs**: Saved in `outputs/logs/`

## Best Practices

1. **Data Quality**: Ensure data is clean and representative
2. **Feature Selection**: Use domain knowledge for feature engineering
3. **Model Validation**: Always use proper train/validation/test splits
4. **Explanation Validation**: Verify explanations make sense
5. **Regular Updates**: Retrain models with new data periodically

## Performance Tips

1. **Memory Management**: Use data sampling for large datasets
2. **Parallel Processing**: Enable multiprocessing where available
3. **GPU Acceleration**: Use GPU for deep learning models
4. **Caching**: Cache preprocessed data and model predictions

## Troubleshooting

### Common Issues

**1. Memory errors**
- Reduce batch size
- Use data streaming
- Enable garbage collection

**2. Slow training**
- Use feature selection
- Reduce data size
- Enable parallel processing

**3. Poor model performance**
- Check data quality
- Adjust hyperparameters
- Try different algorithms

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Check the [Deployment Guide](deployment_guide.md) for production deployment
- Review example notebooks in the `notebooks/` directory
