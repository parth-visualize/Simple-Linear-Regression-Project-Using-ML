# 📈 Simple Linear Regression — Interactive ML Demo

> An interactive Streamlit web app that teaches and demonstrates Simple Linear Regression with real-time model training, visualization, and predictions.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📌 Project Overview

This project is a beginner-friendly, interactive Machine Learning demo built with Streamlit. It walks you through **Simple Linear Regression** — one of the most foundational ML algorithms — step by step.

You can either explore with built-in sample datasets or **upload your own CSV** and train a regression model on your data instantly. The app visualizes the regression line, displays the equation, and lets you make custom predictions — all from a clean, interactive interface.

---

## 🚀 App Features

| Feature | Description |
|---------|-------------|
| 📂 **Sample Datasets** | 3 built-in datasets to explore right away |
| 📤 **Upload Your CSV** | Bring your own data — select any two columns to model |
| ⚙️ **Configurable Split** | Adjust train/test split (10%–40%) with a slider |
| 📊 **Live Visualization** | Interactive scatter plot + regression line via Plotly |
| 📐 **Regression Equation** | Displays the fitted equation: `y = mx + b` in LaTeX |
| 🔮 **Custom Predictions** | Enter any input value and get an instant prediction |
| 📚 **Learn Mode** | Expandable section explaining the math and concepts |

---

## 📦 Built-in Sample Datasets

| Dataset | Feature (X) | Target (Y) | Real-World Use Case |
|---------|------------|------------|---------------------|
| 🎓 Student Hours vs Scores | Hours Studied | Exam Score | Academic performance prediction |
| 🏠 House Size vs Price | House Size (sqft) | Price (USD) | Real estate valuation |
| 📣 Advertising vs Sales | Ad Spend | Sales | Marketing ROI analysis |

---

## 🧠 What the Model Outputs

| Metric | Description |
|--------|-------------|
| **Slope (m)** | How much the target changes per unit increase in the feature |
| **Intercept (b)** | Predicted target value when the feature equals zero |
| **R² Score** | Model fit quality — ranges from 0 to 1 (higher = better) |
| **Regression Equation** | Full `y = mx + b` formula rendered in LaTeX |

---

## 📁 Project Structure

```
📦 Simple-Linear-Regression-Project-Using-ML
 ┣ 📄 app.py              # Main Streamlit application
 ┗ 📄 requirements.txt   # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/parth-visualize/Simple-Linear-Regression-Project-Using-ML.git
cd Simple-Linear-Regression-Project-Using-ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🔬 How It Works

1. **Choose a data source** — pick a sample dataset or upload your own CSV
2. **Configure the model** — set the train/test split percentage using the sidebar slider
3. **Train the model** — click the "Train" button to fit a Linear Regression model using Scikit-learn
4. **Explore results** — view the fitted regression line, model slope, intercept, and R² score
5. **Make predictions** — enter a custom value and instantly see the predicted output

---

## 📐 The Math Behind It

Simple Linear Regression models the relationship between two variables using a straight line:

$$y = mx + b$$

- **y** — Dependent variable (target / what we're predicting)
- **x** — Independent variable (feature / input)
- **m** — Slope (how much y changes per unit of x)
- **b** — Intercept (value of y when x = 0)

The algorithm finds the best-fit line by minimizing the **Sum of Squared Errors (SSE)** across all data points.

---

## 📦 Tech Stack

- **Python** — Core language
- **Scikit-learn** — Linear Regression model, train-test split, metrics
- **Pandas & NumPy** — Data handling and generation
- **Streamlit** — Interactive web application
- **Plotly** — Interactive scatter plots and regression line charts
- **Matplotlib** — Supporting visualization

---

## 👨‍💻 Author

**Parth** — Data Analyst & ML Enthusiast

---

> ⭐ If this project helped you understand Linear Regression, consider giving it a star!
