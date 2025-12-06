from flask import Flask,request,render_template,redirect,url_for
import os 
import sys

from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return redirect(url_for('predict_datapoint'))

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                age=request.form.get('age'),
                hypertension=request.form.get('hypertension'),
                heart_disease=request.form.get('heart_disease'),
                ever_married=request.form.get('ever_married'),
                work_type=request.form.get('work_type'),
                Residence_type=request.form.get('Residence_type'),
                avg_glucose_level=request.form.get('avg_glucose_level'),
                bmi=request.form.get('bmi'),
                smoking_status=request.form.get('smoking_status'),
            )

            data_df = data.get_data_as_dataFrame()
            print(data_df)
            predict_pipeine = PredictPipeline()
            results = predict_pipeine.predict(data_df)
            return redirect(url_for('result_page', output=int(results[0])))
        except Exception as e:
            raise CustomException(e,sys)

@app.route('/result')
def result_page():
    try:
        output = request.args.get('output', default=None, type=int)
        return render_template('result.html', output=output)
    except Exception as e:
        raise CustomException(e,sys)
        
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
    