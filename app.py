from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score'),
            race_ethnicity=request.form.get('race_ethnicity'),
            lunch=request.form.get('lunch'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            test_preparation_course=request.form.get('test_preparation_course')
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=round(prediction[0],2))
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)