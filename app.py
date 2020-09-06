from flask import Flask, request, render_template
from joblib import load
app = Flask(__name__)
model= load('model-1')
transform1=load('Transform-1')
transform2=load('Transform-2')
transform3=load('Transform-3')
transform4=load('Transform-4')
label1=load('Label-Make')
label2=load('Label-Model')
scal=load('Scaler')
dataset=load('dataset')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    pp=dataset[(dataset['Make']==x_test[0][0])]
    if x_test[0][1] not in pp['Model'].unique():
        return render_template('index.html',prediction_text="Choose the car's model from the same company")
    temp=[x_test[0][0]]
    temp=label1.fit_transform(temp)
    x_test[0][0]=temp[0]
    print(x_test)
    temp=[x_test[0][1]]
    temp=label2.fit_transform(temp)
    x_test[0][1]=temp[0]
    print(x_test)
    x_test=transform1.transform(x_test)
    x_test=x_test[:,1:]
    print(x_test)
    x_test=transform2.transform(x_test)
    x_test=x_test[:,1:]
    print(x_test)
    x_test=transform3.transform(x_test)
    x_test=x_test[:,1:]
    print(x_test)
    x_test=transform4.transform(x_test)
    x_test=x_test[:,1:]
    print(x_test)
    x_test=scal.transform(x_test)
    print(x_test)
    prediction=model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if output<0:
        return render_template('index.html',prediction_text='Data is incorrect',prediction="Sorry not able to predict")
    else:
        return render_template('index.html', prediction_text='Mileage = {:.0f} mpg'.format(output),prediction="Mileage = {:.0f} kmpl".format(output*0.43))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
