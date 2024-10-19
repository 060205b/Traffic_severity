# Traffic_severity

Road Traffic Accident Severity Prediction
Overview
This project is a Traffic Accident Severity Prediction System that uses machine learning to predict the severity of a traffic accident based on input parameters such as the age of the driver, weather conditions, type of collision, driving experience, and cause of the accident. The model is built using a Random Forest Classifier, tuned with GridSearchCV for optimal hyperparameters. The application is implemented in Flask for a user-friendly web interface where users can input parameters and get predictions.


**Modeling Process**

Data Preprocessing:
The dataset contains features like Age_band_of_driver, Driving_experience, Weather_conditions, Type_of_collision, and Cause_of_accident.
The data was split into training and test sets (X_train, X_test, y_train, y_test) to evaluate the model.

Model Selection:
A Random Forest Classifier was chosen for the prediction task.
The model was trained with class balancing to handle imbalanced data.

Hyperparameter Tuning:
GridSearchCV was used to tune hyperparameters such as the number of estimators, max depth, and minimum samples split, ensuring the model performed optimally.

Model Saving:
After training and tuning, the best model was saved using pickle for future use.

**Features**

Machine Learning Prediction: A tuned Random Forest model is used to predict traffic accident severity (Low, Medium, High).
Web Interface: Built with Flask, allowing users to interact with the model via a simple web form.
Feature Importances: Visualized to show which features most impact accident severity predictions.How to Run the Project

1. Clone the Repository
git clone https://github.com/yourusername/traffic-accident-severity-prediction.git
cd traffic-accident-severity-prediction

2. Set Up a Virtual Environment (optional but recommended)
source venv/bin/activate   # For Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Flask Application
python app.py
The application will run on http://127.0.0.1:5000/. Open this link in your browser.

**Dependencies**
The project uses the following libraries:
Flask: To build the web application
scikit-learn: For machine learning model building and tuning
Pandas: For data manipulation
Matplotlib: For plotting feature importances
Pickle: To save and load machine learning models

**Screenshots**
1. Home Page (Prediction Form)

![Screenshot 2024-10-20 001420](https://github.com/user-attachments/assets/ea2e17bd-0587-4663-a572-386c2fea76f8)

2. Prediction Result

![Screenshot 2024-10-20 001504](https://github.com/user-attachments/assets/0ae34ae4-f373-488c-bdf6-7adb5366f06c)
