from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temp'])
        rain = int(request.form['rain'])
        snow = int(request.form['snow'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])
        weather = int(request.form['weather_v2'])
        holiday = int(request.form['holiday_v2'])

        input_data = np.array([[temp, rain, snow, day, month, year, hours,
                                minutes, seconds, weather, holiday]])

        prediction = model.predict(input_data)[0]

        return render_template('result.html',
                               prediction_text=f'Estimated Traffic Volume: {round(prediction, 2)}')

    except Exception as e:
        return render_template('result.html',
                               prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
# This code is a Flask application that serves a web interface for predicting traffic volume based on user inputs.
# It loads a pre-trained model from a pickle file and uses it to make predictions based on the data submitted through a form.
# The application handles errors gracefully and displays the prediction or an error message on the result page.
# The application runs in debug mode, which is useful for development but should be turned off in production.
# The application uses Flask's templating engine to render HTML pages for the home and result views.
# The home route renders the main input form, while the predict route processes the form data and returns the prediction.
