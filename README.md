# Titanic Survival Prediction - Classification Project

## 📋 Project Overview
This project analyzes passenger data from the Titanic disaster to predict survival and uncover factors affecting survival chances. The analysis includes data preprocessing, exploratory data analysis, visualization, and machine learning classification models.

## 🎯 Objectives

### Data Preprocessing
- Handle missing values in Age, Embarked, and Cabin
- Convert categorical features (Sex, Embarked) to numerical form
- Create new features:
  - Family size (SibSp + Parch)
  - Title extraction from passenger names

### Exploratory Data Analysis
- Compare survival rates across:
  - Gender
  - Passenger class
  - Age groups
- Identify relationships between features and survival

### Visualizations
- Bar charts for survival counts
- Count plots for categorical variables
- Age distribution histograms
- Seaborn visualizations:
  - Survival vs Gender
  - Survival vs Passenger Class

### Hidden Pattern Discovery
- Are women and children more likely to survive?
- Does higher class guarantee survival?

### Machine Learning
- Build classification models: Logistic Regression & Decision Tree
- Evaluate using: Accuracy, Confusion Matrix, Classification Report

## 📂 Project Structure
```
titanic-prediction/
├── dashboard_app.py          # Interactive Streamlit dashboard
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks
│   └── titanic_analysis.ipynb
├── scripts/                 # Python scripts
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline (Recommended)
```bash
python scripts/titanic_classification.py
```

This script will:
- load `data/train.csv` if present, otherwise download a compatible Titanic training dataset and save it to `data/train.csv`
- preprocess missing values in **Age**, **Embarked**, **Cabin**
- create engineered features (**Family size**, **Title**, age groups)
- run EDA and hidden pattern checks
- train **Logistic Regression** and **Decision Tree** models
- print **accuracy** and **confusion matrix** for both models
- save plots to `data/outputs/`

### 3. Open the Notebook
```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

### 4. Launch Interactive Dashboard (Web App)
```bash
streamlit run dashboard_app.py
```

Dashboard features:
- interactive filters by gender, class, and age group
- survival charts and EDA visuals
- model comparison with confusion matrix heatmaps
- live passenger survival prediction form

### 5. Dataset
- Primary expected file: `data/train.csv` (Titanic - Machine Learning from Disaster format)
- If missing, the script uses a compatible public Titanic training dataset and stores it locally.

## 📊 Dataset Information
- **Source**: Titanic - Machine Learning from Disaster
- **Records**: ~891 passengers (train set)
- **Features**: Passenger ID, Class, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

## 🛠️ Technologies Used
- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning models and evaluation

## 📈 Expected Outcomes
- Clean, preprocessed dataset ready for modeling
- Comprehensive exploratory data analysis
- Insights into survival patterns
- Trained classification models with performance metrics
- Visualizations showing relationships between features and survival

## ✅ Current Verified Results
- Logistic Regression accuracy: **0.7989**
- Decision Tree accuracy: **0.7877**
- Generated plots:
  - `data/outputs/survival_counts.png`
  - `data/outputs/survival_vs_gender_class.png`
  - `data/outputs/age_distribution.png`

## 📝 Notes
- Missing values are handled appropriately for each feature
- Categorical variables are encoded using appropriate techniques
- Feature engineering creates meaningful new variables
- Models are evaluated using multiple metrics for comprehensive assessment

