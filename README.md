# ad-spend-optimization
# 🧮 Ad Spend Optimization

A data-driven machine learning project that helps businesses **optimize marketing budgets** by predicting the ideal ad spend and expected ROI across campaigns.

---

## 📊 Overview

This project uses regression models to analyze relationships between advertising parameters — such as impressions, clicks, engagement, and total ad spend — to predict the **return on investment (ROI)**.

It aims to help marketers allocate their budget more efficiently and maximize performance across different campaigns.

---

## 🧠 Key Features

- Predicts ROI and optimal ad spend using machine learning  
- Supports data input through a Streamlit-based web app  
- Provides visual analytics for ad performance  
- Handles multiple campaign parameters dynamically  

---

## 🧩 Tech Stack

- **Language:** Python 🐍  
- **Framework:** Streamlit  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Joblib  
- **Version Control:** Git & GitHub  

---

## 📦 Model and Dataset

> 📝 Note: GitHub has a 100 MB upload limit.  
> Therefore, large files are hosted externally.

| File | Description | Download |
|------|--------------|-----------|
| `ad_spend_optimizer.pkl` | Trained regression model | [Download from Google Drive](https://drive.google.com/file/d/155iMBXPjvot1qFD-105OEcMtWzX5HqqG/view?usp=drive_link) |
| `clean_marketing_data.csv` | Preprocessed marketing dataset | [Download from Google Drive](https://drive.google.com/file/d/19TqI_K2rA38TWhwO4dlFAtAQ9bWbmHQ3/view?usp=drive_link) |

After downloading, place the files in this structure:
adspend/
 ┣ app.py
 ┣ model/
 ┃ ┗ ad_spend_optimizer.pkl
 ┣ data/
 ┃ ┗ clean_marketing_data.csv
 ┣ requirements.txt
 ┗ README.md

---

## ⚙️ Setup Instructions

Clone this repository and install the dependencies:

```bash
git clone https://github.com/anshikakatare/ad-spend-optimization.git
cd ad-spend-optimization
pip install -r requirements.txt

