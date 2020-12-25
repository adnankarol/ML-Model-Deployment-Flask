from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():

    inputs = [int(x) for x in request.form.values()]

    test_input = np.array(inputs).reshape(-1,1)
    prediction=model.predict(test_input)
    output = prediction.reshape(-1)[0]

    return render_template('index.html', pred = "The Predicted Temperature in Farenheit is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
