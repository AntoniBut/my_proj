import flask
from flask import render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', "POST"])
@app.route('/index',methods=['GET', "POST"])

def main():
    if flask.request.method=="GET":
        return render_template('main.html')
    if flask.request.method=="POST":
        loaded_model = pickle.load(open('contoso/reg_pkl.pkl', 'rb'))
                
        IW=float(flask.request.form['IW'])
        IF=float(flask.request.form['IF'])
        VW=float(flask.request.form['VW'])
        FP=float(flask.request.form['FP'])
        
        a = np.array([[IW,IF,VW,FP]])
        X_pred=pd.DataFrame(a)
        norm = Normalizer()
        X_pred = norm.fit_transform(X_pred)      
        print(X_pred)  
        y_pred=loaded_model.predict(X_pred)
        str_res='Глубина шва (Depth) = '+str(round(y_pred[0,0],2))+' и ширина шва (Width)='+str(round(y_pred[0,1],2))
        print(X_pred)
        return render_template('main.html',result=str_res)
if __name__=='__main__':
    app.run()
