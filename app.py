from flask import redirect
from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os
from joblib import dump, load
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

model = load('pipe.pkl')
encoding = load('encoding.pkl')

def encode_value(value,column_name):
  labels =  encoding[column_name]
  enc = preprocessing.LabelEncoder().fit(labels)
  return enc.transform(value)[0]

app =Flask(__name__)




@app.route('/')
def index():
    return render_template('2.html')

@app.route('/',methods=['GET','POST'])
def home():

    sl=request.args.get['Gender']
    print(sl)
    
    return render_template('index1.html')#,prediction=prediction[0])

@app.route('/1')
def cool_form():

    return render_template('1.html')

@app.route('/2')
def cool_form1():

    return render_template('2.html')

@app.route('/predict', methods=['POST'])
def predict():
    

    # Create a list to store the form variables
    form_list = []
    form_list.append(request.form['sexe'])
    form_list.append(request.form['cardiopathie'])
    form_list.append(request.form['bilirubine_t_normal'])
    form_list.append(request.form['irm'])
    form_list.append(request.form['chimiotherapie'])
    form_list.append(request.form['clampage_on'])
    form_list.append(request.form['type_resections'])
    form_list.append(request.form['resection_majeure'])
    form_list.append(request.form['cholecystectomie'])
    form_list.append(request.form['saignement_quantite'])
    form_list.append(request.form['ahospitalisation_en_reanimation'])
    form_list.append(request.form['antalgiques'])
    form_list.append(request.form['hemoglobine_postop_norm'])
    form_list.append(request.form['protidemie_postop'])
    form_list.append(request.form['drainage_percutane'])

    new_data = np.array(form_list).reshape(-1, 1)
    columns=['SEXE', 'cardiopathie', 'BilirubineTnormal', 'IRM',
       'Chimiothérapie\xa0', 'clampageON', 'TypedeRésections\xa0',
       'résectionmajeure', 'cholécystectomie', 'Saignementquantité',
       'ahospitalisation\xa0enréanimation', 'Antalgiques',
       'Hémoglobinepostopnorm', 'protidémiepostop', 'Drainagepercutané']


    for i in range(len(columns)): 
        if columns[i] in encoding:
            new_data[i]= encode_value(new_data[i],columns[i])

    #pipeline= model
    prediction = model[1:].predict(new_data.reshape(1,-1))[0]
    prediction_proba = model[1:].predict_proba(new_data.reshape(1,-1))[0]

    # Create a bar chart to visualize the prediction probability
    labels = ['Complication','NComplication']

    plt.bar(labels, prediction_proba, color = ['red','blue'])
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.savefig('static/images/prediction_plot.png') 
    #pred = pipeline.predict(new_data)

    # Return the prediction/result or render a response template
    return render_template('1.html', prediction=prediction, prediction_proba=prediction_proba, image_path='static/images/prediction_plot.png')

@app.route('/predict1', methods=['POST'])
def predict1():
    form_list = []
    form_list.append(request.form['sexe'])
    form_list.append(request.form['cardiopathie'])
    form_list.append(request.form['bilirubine_t_normal'])
    form_list.append(request.form['irm'])
    form_list.append(request.form['chimiotherapie'])
    form_list.append(request.form['clampage_on'])
    form_list.append(int(request.form['type_resections']))
    form_list.append(request.form['resection_majeure'])
    form_list.append(request.form['cholecystectomie'])
    form_list.append(int(request.form['saignement_quantite']))
    form_list.append(request.form['ahospitalisation_en_reanimation'])
    form_list.append(request.form['antalgiques'])
    form_list.append(request.form['hemoglobine_postop_norm'])
    form_list.append(int(request.form['protidemie_postop']))
    form_list.append('oui')

    new_data = np.array(form_list).reshape(-1, 1)
    columns=['SEXE', 'cardiopathie', 'BilirubineTnormal', 'IRM',
       'Chimiothérapie\xa0', 'clampageON', 'TypedeRésections\xa0',
       'résectionmajeure', 'cholécystectomie', 'Saignementquantité',
       'ahospitalisation\xa0enréanimation', 'Antalgiques',
       'Hémoglobinepostopnorm', 'protidémiepostop', 'Drainagepercutané']


    for i in range(len(columns)): 
        if columns[i] in encoding:
            new_data[i]= encode_value(new_data[i],columns[i])

    #pipeline= model
    prediction = model[3:].predict(new_data.reshape(1,-1))[0]
    prediction_proba = model[3:].predict_proba(new_data.reshape(1,-1))[0]

    # Create a bar chart to visualize the prediction probability
    labels = ['Complication','NComplication']

    plt.bar(labels, prediction_proba, color = ['red','blue'])
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.savefig('static/images/prediction_plot.png') 
    #pred = pipeline.predict(new_data)

    # Return the prediction/result or render a response template
    return render_template('2.html', prediction=prediction, prediction_proba=prediction_proba, image_path='static/images/prediction_plot.png')

@app.route('/predict2', methods=['POST'])
def predict2():
    gender = int(request.form.get('Gender'))
    return str(gender)


if __name__ == '__main__':
    app.run(debug=True)