# -*- coding: utf-8 -*-
from flask import Flask,request,render_template
import pickle

app= Flask(__name__)
@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/predict',methods=['POST'])
def predict():

    model = pickle.load(open('model.pkl','rb'))
    
    text = request.form['Feedback']
    
    cv = pickle.load(open('cv.pkl','rb'))
    
    x = cv.transform([text])
    
    x = x.toarray()
    #return str(x)
    
    op = model.predict(x)[0]
   
    if op==1:
        return 'happy'
        #return ('<h1>' + request.form['Name']+'!'+'<br>'+ "Thank You for liking our restaurant." + '</h1>')
    else:
        return 'not happy'
        #return('<h1>' + request.form['Name']+'!'+'<br>'+"We apologize that our service did not satisfy your expectations." + '</h1>')

 
    

if __name__ =='__main__':
    app.run(debug=True)
