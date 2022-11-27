import flask
from flask import render_template
import pickle
import sklearn
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', "POST"])
@app.route('/index',methods=['GET', "POST"])

def main():
    if flask.request.method=="GET":
        return render_template('main.html')
    if flask.request.method=="POST":
        with open('C:/Users/Antonida.Butuzova/my_proj/contoso/KNNreg_pkl.pkl','rb') as f:
            loaded_model=pickle.load(f)
        IW=float(flask.request.form['IW'])
        IF=float(flask.request.form['IF'])
        VW=float(flask.request.form['VW'])
        FP=float(flask.request.form['FP'])
        
        a = np.array([[IW,IF,VW,FP]])
        X_pred=pd.DataFrame(a)
        
        y_pred=loaded_model.predict(X_pred)
        str_res='Глубина шва (Depth) = '+str(y_pred[0,0])+' и ширина шва (Width)='+str(y_pred[0,1])
        print(str_res)
        return render_template('main.html',result=str_res)
if __name__=='__main__':
    app.run()
