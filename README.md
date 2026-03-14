🏠 House Price Prediction — Machine Learning Project

📌 Overview

This project predicts residential house prices using machine learning techniques applied to the Ames Housing dataset from the Kaggle competition House Prices: Advanced Regression Techniques.

The goal is to build a predictive model that estimates the SalePrice of a house based on various property characteristics such as location, house size, construction year, and overall quality.

---

📊 Dataset

The dataset comes from the Kaggle competition:

House Prices: Advanced Regression Techniques

It contains information on 1460 training houses with 79 explanatory variables describing various aspects of residential homes.

Example features include:

- Lot size
- Neighborhood
- Year built
- Living area
- Basement size
- Garage capacity
- Overall material quality

Target variable:

SalePrice — the final sale price of the house.

---

🧠 Project Workflow

The project follows a typical machine learning pipeline:

1. Data Loading
2. Data Cleaning
3. Missing Value Handling
4. Feature Engineering
5. Categorical Variable Encoding
6. Model Training
7. Cross Validation
8. Prediction & Kaggle Submission

---

⚙️ Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
Matplotlib
Seaborn
Jupyter Notebook

---

🧪 Machine Learning Models

Several regression models were explored:

- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost
- LightGBM

The final solution uses an ensemble approach combining multiple models to improve prediction accuracy.

---

📈 Evaluation Metric

The competition evaluates models using:

Root Mean Squared Log Error (RMSLE)

Lower RMSLE values indicate better prediction accuracy.

---

🏗 Feature Engineering

To improve model performance, additional features were created such as:

- TotalSF — total house square footage
- TotalBath — total number of bathrooms
- HouseAge — age of the property
- Remodeled — whether the house was remodeled
- TotalPorch — total porch area

These features capture important information about property value.

---

📂 Project Structure

house-price-prediction/

notebooks/
    kaggle_house_price_model.ipynb

src/
    preprocessing.py
    feature_engineering.py
    train_model.py

README.md
LICENSE
.gitignore

---

🚀 How to Run the Project

Clone the repository

git clone https://github.com/Nomusa990822/house-price-prediction.git

Install dependencies

pip install -r requirements.txt

Run the model training script

python train_model.py

---

📊 Results

The model successfully predicts house prices using advanced regression techniques and feature engineering.

Further improvements could include:

- Hyperparameter tuning
- Advanced model stacking
- Feature selection optimization

---

📎 Kaggle Competition

Dataset and competition details are available on Kaggle.

House Prices: Advanced Regression Techniques

---

👤 Author

Nomusa Shongwe
Data Analytics & Machine Learning Enthusiast

GitHub: https://github.com/Nomusa990822

---

📄 License

This project is licensed under the MIT License.
