# Stock Index Trend Prediction using Machine Learning on NYSE Data

## Data Science Institute - Cohort 7 - Team ML 08

**Members**
- [Mahshid Chekini](https://github.com/MahChek)
- [Senthil Arumugam](https://github.com/senthilarumugamsub)
- [Ashwinder Singh](github.com/ashwinder1)
- [Muhammad Faisal](https://github.com/faisalapp)

## Project Overview:

This project applies **supervised machine learning** to forecast **stock price trends** using data from the **New York Stock Exchange (NYSE)**. The objective is to predict the **direction of stock price movement** (increase or decrease), empowering investors, traders, and portfolio managers with **data-driven insights** for market decisions.

By leveraging historical price data (split-adjusted) and technical indicators, this project demonstrates how **modern ML techniques** can enhance traditional **technical analysis**, providing scalable and interpretable solutions for **market trend forecasting**.

### Business Problem

Financial markets move rapidly, and anticipating price direction is essential for traders and investors. Traditional chart-based or indicator-driven approaches often suffer from subjectivity and lag.

Our project bridges this gap by applying machine learning to historical NYSE data, providing an evidence-based framework for predicting stock index trends and identifying patterns not easily visible through manual analysis.

### Stakeholders & Relevance

| Stakeholder | Interest / Why They Care |
| ----- | ----- |
| **Retail Investors** | Want accurate short-term predictions to make informed buy/sell decisions. |
| **Portfolio Managers / Financial Analysts** | Need data-driven insights for portfolio optimization and risk management. |
| **Algorithmic Traders** | Seek predictive signals for automated strategy execution. |
| **Data Scientists / Quant Researchers** | Benefit from benchmark models and feature engineering techniques for stock time-series. |
| **Educational Institutions / ML Learners** | Gain reproducible project examples for time-series financial modeling. |


### Risks and Uncertainties

**Market and Data Risks**

* **Volatility & Non-stationarity:** Stock market dynamics change frequently, making long-term model stability challenging.  
* **External Factors:** Models cannot capture sudden macroeconomic shocks, policy changes, or news events.  
* **Data Lag:** Using only historical data limits reaction to real-time market conditions.

**Modeling Risks**

* **Overfitting:** Deep models like CNN and LSTM may overfit to short-term patterns if not properly regularized.  
* **Feature Leakage:** Derived indicators might indirectly include future information; careful time-order validation is required.  
* **Imbalanced Trends:** Prolonged bullish or bearish phases can skew model performance.  
* **Interpretability:** Deep learning models may be harder to explain to financial decision-makers.

Mitigation includes rolling-window validation, regularization, and model explainability analysis.

### Project Objective

- Developed a reproducible ML pipeline for predicting stock price direction using engineered technical features.

* Predict directional trends (price increase or decrease) for NYSE stocks.  

* Evaluate multiple models (Linear Regression, XGBoost, CNN, LSTM) for trend prediction performance.  

* Assess model interpretability using SHAP or feature importance visualization.

- Evaluated models on accuracy, precision, recall, and ROC-AUC to identify the most effective approach for time-series trend forecasting.

> Ultimately, deliver a prototype for data-driven stock trend forecasting that can serve as a foundation for analysts, researchers, or trading systems.

## Setup and Installation

### Prerequisites
- Python ≥ 3.9
- Git / GitHub
- Jupyter Notebook or VS Code

### Installation Steps

1.   Clone the repository
    
    `git clone https://github.com/ashwinder1/Stock-Trend-Analysis-with-NYSE.git cd Stock-Trend-Analysis-with-NYSE` 
    
2.  Create and activate environment
    
    `conda env create -f requirements.yml
    conda activate stock-trend` 
    
3.  Launch Jupyter Notebook
    
    `jupyter notebook`


## Dataset Overview

-   **Source:** [Kaggle NYSE Dataset](https://www.kaggle.com/datasets/dgawlik/nyse)
    
-   **Content:** Split-adjusted price data for 501 S&P companies
    
-   **Period:** 2010–2016 (varies by company listing date)
    
-   **Features:** `date`, `symbol`, `open`, `close`, `low`, `high`, `volume`, `adj_close`


## Methodology and Workflow

We use the approach that answers the business question by quantifying which indicators best predict the direction of price change, directly supporting investment decision-making.

| Stage                      | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| **1. Data Collection**     | Obtain and preprocess Kaggle NYSE data                                   |
| **2. EDA**                 | Explore distributions, missing values, and correlations                  |
| **3. Feature Engineering** | Compute moving averages, RSI, MACD, Bollinger Bands                      |
| **4. Model Development**   | Train baseline (Logistic, RF, XGBoost) and sequential (LSTM, CNN) models |
| **5. Evaluation**          | Compare models using classification metrics                              |
| **6. Interpretability**    | Use SHAP to visualize feature contributions                              |
| **7. Reporting**           | Summarize results, insights, and visualizations                          |

## Project Structure

```
Stock-Trend-Analysis-with-NYSE/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   └── notebooks/
├── models/
├── images/
├── README.md
└── .gitignore
```

**Folder Descriptions:**

-   `data/` → Raw and processed datasets
    
-   `experiments/` → Jupyter notebooks for experiments and EDA
    
-   `models/` → Trained models and pickled files
    
-   `images/` → Visual outputs for documentation
    
-   `.gitignore` → Excludes large or sensitive files

### Key Dependencies

This project relies on a suite of Python libraries and frameworks that support end-to-end data science workflows, including data preprocessing, machine learning, model evaluation, interpretability, and visualization. 

| Category                  | Libraries                                         |
| ------------------------- | ------------------------------------------------- |
| **Data Processing**       | pandas, NumPy                                     |
| **ML / DL Models**        | scikit-learn, XGBoost, LightGBM, Keras/TensorFlow |
| **EDA / Visualization**   | matplotlib, seaborn, scikit-plot, ydata_profiling |
| **Imbalanced Data**       | imbalanced-learn (SMOTE, ADASYN)                  |
| **Hyperparameter Tuning** | hyperopt                                          |
| **Interpretability**      | shap                                              |
| **Utilities**             | os, pickle, pathlib                               |

## Planned Deliverables

* Cleaned and merged **technical dataset**  
* Comparative model performance analysis (ML vs Deep Learning)  
* Jupyter notebook for trend forecasting 
* Model interpretability / explainability report  
* Documentation for reproducibility (README \+ Jupyter notebook)

## Results

To be filled with results

## Conclusions and Future Directions

### Achievements

### Next Steps


## Team Members Reflection Videos

Each member will provide a short reflection video discussing their role, learning outcomes, and contributions.

