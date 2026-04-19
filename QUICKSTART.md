# Quick Start Guide

## Setup (First Time Only)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure database (optional - for SQL features)
cp .env.example .env
# Edit .env with your credentials
```

## Run the Complete Pipeline

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Download data from Kaggle
- Clean and preprocess the data
- Train baseline and optimized models
- Save the best XGBoost model to `models/xgboost_model.pkl`

### Step 2: SQL Integration (Optional)

```bash
python sql_integration.py
```

This will:
- Migrate cleaned data to MySQL
- Run analytics queries
- Store model predictions in the database

### Step 3: Test Predictions

```bash
python test_prediction.py
```

This will:
- Load the trained model
- Run sample test scenarios
- Allow interactive testing with custom inputs

## Explore the Notebook

```bash
jupyter notebook notebooks/crime_analysis.ipynb
```

## View Reports

- `reports/Final_Project_Report.md` - Executive summary
- `reports/In_Depth_Analysis_Report.md` - Technical details

## Project Structure

```
├── data/                 # Data storage
│   ├── raw/             # Raw datasets (gitignored)
│   └── processed/       # Processed data (gitignored)
├── notebooks/           # Jupyter notebooks
├── src/                 # Python modules
│   ├── data_preprocessing.py
│   ├── model.py
│   └── sql_pipeline.py
├── models/              # Trained models
├── reports/             # Analysis reports
├── train_model.py       # Training script
├── sql_integration.py   # SQL pipeline script
└── test_prediction.py   # Prediction testing script
```

## Troubleshooting

### Missing Kaggle Credentials
If you get authentication errors with Kaggle:
```bash
kaggle datasets list  # This will prompt you to set up credentials
```

### Database Connection Issues
- Check your `.env` file credentials
- Ensure MySQL server is accessible
- Verify network connectivity

### Module Import Errors
Make sure you're running scripts from the project root directory:
```bash
cd /path/to/chicago-crime-analysis
python train_model.py
```

## Next Steps

1. **Explore the code**: Check `src/` modules to understand the pipeline
2. **Customize features**: Modify `src/data_preprocessing.py` to add new features
3. **Tune models**: Adjust hyperparameters in `src/model.py`
4. **Deploy**: Use the trained model in a web app or API

## Need Help?

- Check the full `README.md` for detailed documentation
- Review the reports in `reports/` directory
- Open an issue on GitHub for bugs or questions
