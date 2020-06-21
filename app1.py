import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('new22.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction
    if(output==1):
        out_str="Our diagnosis suggests patient does suffer from diabetes.Please get checked soon!"
    else:
        out_str="Diagnosis suggests that patient does not suffers from diabetes."



    return render_template('new22.html', prediction_text=' status:{} '.format(out_str))
    

if __name__ == "__main__":
    app.run(debug=True)
