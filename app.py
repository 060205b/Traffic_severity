from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained LogisticRegression model
models_dir = 'models_pickle_file'
model_file = os.path.join(models_dir, 'tuned_random_forest_model.pkl')
with open(model_file, 'rb') as f:
    rf_model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Prediction function called")
    try:
        # Get the form values
        age_range = int(request.form['ageRange'])
        weather = request.form['weather'].strip()
        collision = request.form['collision'].strip()
        experience = request.form['experience'].strip()
        cause = request.form['cause'].strip()

        print("Form values:", age_range, weather, collision, experience, cause)

        # Categorical mappings (same as model training)
        weather_mapping = {
            "Normal": 0,
            "Raining": 1,
            "Raining and Windy": 2,
            "Cloudy": 3,
            "Other": 4,
            "Windy": 5,
            "Snow": 6,
            "Unknown": 7,
            "Fog or mist": 8
        }

        collision_mapping = {
            "Vehicle with vehicle collision": 0,
            "Collision with roadside-parked vehicles": 1,
            "Collision with roadside objects": 2,
            "Collision with animals": 3,
            "Other": 4,
            "Rollover": 5,
            "Fall from vehicles": 6,
            "Collision with pedestrians": 7,
            "With Train": 8,
            "Unknown": 9
        }

        experience_mapping = {
            "1-2yr": 0,
            "Above 10yr": 1,
            "5-10yr": 2,
            "2-5yr": 3,
            "No Licence": 4,
            "Below 1yr": 5,
            "Unknown": 6
        }

        cause_mapping = {
            "Moving Backward": 0,
            "Overtaking": 1,
            "Changing lane to the left": 2,
            "Changing lane to the right": 3,
            "Overloading": 4,
            "No priority to vehicle": 5,
            "No priority to pedestrian": 6,
            "Improper parking": 7,
            "Overspeed": 8,
            "Unknown": 9
        }

        age_range_mapping = {
            1: 0,
            2: 1,
            3: 2,
            4: 3
        }

        # Convert categorical inputs to numerical values
        weather_num = weather_mapping.get(weather, -1)
        collision_num = collision_mapping.get(collision, -1)
        experience_num = experience_mapping.get(experience, -1)
        cause_num = cause_mapping.get(cause, -1)
        age_range_num = age_range_mapping.get(age_range, -1)

        print("Numerical values:", weather_num, collision_num, experience_num, cause_num, age_range_num)

        # Prepare the input array for prediction
        input_data = np.array([[age_range_num, weather_num, collision_num, experience_num, cause_num]])
        print("Input data shape:", input_data.shape)

        # Make the prediction
        prediction = rf_model.predict(input_data)[0]
        print("Prediction:", prediction)

        # Map the prediction to a string
        severity_mapping = {0: "Low", 1: "Medium", 2: "High"}
        prediction_str = severity_mapping.get(int(prediction), "Unknown")
        print("Prediction string:", prediction_str)

        # Provide suggestions based on the prediction
        if prediction_str == "High":
            suggestion = "Ensure to take all necessary precautions when driving. Avoid busy roads if possible."
        elif prediction_str == "Medium":
            suggestion = "Be aware of your surroundings and drive cautiously. Consider taking alternative routes."
        else:  # Low
            suggestion = "Safe driving conditions are predicted. Enjoy your drive!"

        # Return the result page
        return render_template('result.html', prediction=prediction_str, suggestion=suggestion)

    except Exception as e:
        print("Error during prediction:", e)
        return render_template('result.html', prediction="Error during prediction", suggestion="")

if __name__ == '__main__':
    app.run(debug=True)
