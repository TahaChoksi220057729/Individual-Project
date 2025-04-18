# MUN Speech Evaluator - Transfer Learning Implementation

This implementation uses transfer learning to improve the MUN Speech Evaluator model by pretraining on a larger corpus of UN general debates and then fine-tuning on the specialized MUN speech dataset.

## Project Structure

```
project/
│
├── data_processor.py       # Data preprocessing for both datasets
├── base_model.py           # Training script for the base model (UN debates)
├── transfer_model.py       # Fine-tuning script for the transfer model (MUN speeches)
├── evaluate_model.py       # Evaluation script for comparing both models
├── run_transfer_learning.py # Main script for running the entire pipeline
├── requirements.txt        # Dependencies
│
├── un_general_debates.csv  # General UN debates dataset (provided separately)
└── training_data.json      # MUN speech dataset (existing)
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Entire Pipeline

The easiest way to run the entire pipeline is to use the `run_transfer_learning.py` script:

```bash
python run_transfer_learning.py --un_data_path un_general_debates.csv --mun_data_path training_data.json
```

This will:
1. Train the base model on the UN debates dataset
2. Fine-tune the model on the MUN speeches dataset
3. Evaluate and compare both models

### Advanced Options

The run script supports several options:

```bash
python run_transfer_learning.py --help
```

Key options include:
- `--output_dir`: Directory to save all output (default: `output`)
- `--base_epochs`: Number of epochs for base model training (default: 3)
- `--transfer_epochs`: Number of epochs for transfer model training (default: 5)
- `--clean`: Clean output directories before running
- `--skip_base`: Skip base model training (use existing model)
- `--skip_transfer`: Skip transfer model training (use existing model)

### Running Individual Components

You can also run each component separately:

#### 1. Train the Base Model

```bash
python base_model.py --data_path un_general_debates.csv --output_dir models --epochs 3
```

#### 2. Fine-tune with MUN Data

```bash
python transfer_model.py --base_model_path models/un_base_model --mun_data_path training_data.json --output_dir models --epochs 5
```

#### 3. Evaluate and Compare Models

```bash
python evaluate_model.py --base_model_path models/un_base_model --transfer_model_path models/mun_transfer_model --mun_data_path training_data.json --output_dir results
```

## Datasets

### UN General Debates Dataset

This dataset contains speeches from UN General Debates, which are used for pretraining the base model. The dataset should be a CSV file with at least a "text" column containing the speech texts.

### MUN Speech Dataset

This is the existing MUN speech dataset (training_data.json) used for fine-tuning and evaluation.

## Output

The pipeline produces the following outputs:

- `output/models/`: Contains the trained models
  - `un_base_model/`: Base model trained on UN debates
  - `mun_transfer_model/`: Transfer model fine-tuned on MUN speeches
  - `training_history.png`: Plot of training metrics for the base model
  - `transfer_history.png`: Plot of training metrics for the transfer model

- `output/results/`: Contains evaluation results
  - `comparison_results.json`: Detailed metrics comparing both models
  - `predictions_comparison.png`: Plot comparing predictions of both models
  - `error_distribution.png`: Error distributions for both models
  - `metrics_comparison.png`: Bar chart comparing performance metrics
  - `r2_comparison.png`: R² score comparison

## Model Architecture

The implementation uses a transformer-based architecture (BERT) for both the base and transfer models, with the following characteristics:

- **Base Model**: Pretrained BERT fine-tuned on UN general debates, which captures general diplomatic and UN-specific language patterns.
- **Transfer Model**: The base model further fine-tuned on MUN speeches, preserving the general knowledge while adapting to MUN-specific characteristics.

The transfer learning approach applies different learning rates to different parts of the model during fine-tuning:
- Lower learning rate for the base layers
- Higher learning rate for the classifier layers

## Evaluation Metrics

The models are evaluated using various regression metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Performance Improvement

The transfer learning approach typically shows improvement over the base model in the following ways:
1. Lower error metrics (MSE, RMSE, MAE)
2. Higher R² score
3. Better handling of MUN-specific language and topics
4. More accurate predictions for the specialized task 