from flask import Flask, render_template, request
import dill
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def diabetes_check():
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bp']
        skin = request.form['skin']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetes_pedigree = request.form['diabetes_pedigree']
        age = request.form['age']

        with open('scaler.joblib', 'rb') as io:
            scaler = dill.load(io)
        with open('classifier.joblib', 'rb') as io:
            classifier = dill.load(io)

        input_data = [pregnancies, glucose, bp, skin, insulin, bmi, diabetes_pedigree, age]
        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        std_data = scaler.transform(input_data_reshaped)
        print(std_data)

        prediction = classifier.predict(std_data)
        print(prediction)

        if str(prediction[0]) == '0':
            return render_template('index.html', status='success', result='The person is not diabetic')
        else:
            return render_template('index.html', status='success', result='The person is diabetic')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)