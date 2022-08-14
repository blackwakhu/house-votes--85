from copyreg import pickle
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('naive.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    initial_features = [int(x) for x in request.form.values()]
    final_features = np.array(initial_features).reshape(1, -1)
    y_pred = model.predict(final_features)
    if y_pred == 0:
        prediction = 'the party is the republican party'
    else:
        prediction = 'the party is the democratic party'
    return render_template('index.html',prediction = prediction)

if __name__ == "__main__":
    app()