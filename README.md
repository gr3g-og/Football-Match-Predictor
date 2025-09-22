⚽ Football Match Predictor

A machine learning project that predicts the outcomes of football matches. It uses a Random Forest Classifier trained on historical match data to make data-driven predictions. The project includes a full pipeline for data preparation, model training, and prediction generation.

✨ Features

End-to-end pipeline: From raw data input to final, formatted predictions.

Automated model tuning: RandomizedSearchCV finds the best hyperparameters.

Comprehensive feature engineering: 18 features including goal differences, vote counts, and prediction variance.

Two prediction methods:

Machine learning-based predictor

Pattern-matching predictor

User-friendly output: Predictions exported to Excel with color-coding.

📦 Prerequisites

Python 3.x

Install dependencies:

pip install pandas openpyxl scikit-learn joblib matplotlib seaborn tqdm

🔄 Project Workflow
1️⃣ Training the Model (run only when new historical data is added)

Step 1: Prepare the dataset

python prepare_predictions.py


Processes RawPrediction.xlsx → creates enhanced_dataset.csv.

Step 2: Create the scaler

python create_scaler.py


Fits a scaler and saves scaler.pkl.

Step 3: Train the model

python train_model.py


Performs hyperparameter tuning

Saves model → football_model_optimized.pkl

Saves evaluation → model_evaluation.txt

2️⃣ Making Predictions

Step 1: Add new matches → fill in new_matches.xlsx.

Step 2: Run predictor

python predict_matches.py


Exports results → predicted_results.xlsx (with color-coded outcomes).

3️⃣ Alternative Method: Pattern-based Prediction

Step 1: Ensure new_matches.xlsx is filled.

Step 2: Run pattern matcher

python pattern_predict.py


Exports results → pattern_matches.xlsx.

📂 File Descriptions

train_model.py → Trains and evaluates Random Forest with tuning.

prepare_predictions.py → Processes raw Excel data → feature engineering → enhanced_dataset.csv.

create_scaler.py → Fits & saves scaler (scaler.pkl).

predict_matches.py → Predicts outcomes of new_matches.xlsx → saves results.

pattern_predict.py → Pattern-matching prediction method.

RawPrediction.xlsx → Historical raw predictions.

new_matches.xlsx → Input new matches for prediction.

enhanced_dataset.csv → Engineered dataset used for training.

football_model_optimized.pkl → Optimized trained model.

scaler.pkl → Saved data scaler.

🚀 Future Improvements

Automate data scraping (bypass Cloudflare).

Expand dataset across more leagues/seasons.

Try deep learning (e.g. LSTMs for sequence data).

Build a web dashboard for interactive predictions.
