from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import io

app = Flask(__name__)

model = joblib.load("final_salary_model1.joblib")
feature_names = ['age', 'educational-num', 'gender', 'hours-per-week',
    'workclass_Local-gov', 'workclass_Never-worked', 'workclass_NotListed',
    'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
    'workclass_State-gov', 'workclass_Without-pay', 'occupation_Armed-Forces',
    'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_NotListed',
    'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty',
    'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
    'occupation_Transport-moving']

education_map = {
    "Bachelors": 13, "Some-college": 10, "11th": 7, "HS-grad": 9,
    "Assoc-acdm": 12, "Assoc-voc": 11, "Masters": 14, "Doctorate": 16,
    "Prof-school": 15
}
education_levels = [
    "10th", "11th", "12th", "Assoc-acdm", "Assoc-voc", "Bachelors", 
    "Doctorate", "HS-grad", "Masters", "Prof-school", "Some-college"
]


workclass_options = [
    "Local-gov", "Never-worked", "NotListed", "Private", "Self-emp-inc",
    "Self-emp-not-inc", "State-gov", "Without-pay"
]
occupation_options = [
    "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners",
    "Machine-op-inspct", "NotListed", "Other-service", "Priv-house-serv", "Prof-specialty",
    "Protective-serv", "Sales", "Tech-support", "Transport-moving"
]

education_options = list(education_map.keys())

@app.route('/')
def index():
    return render_template("index.html", workclasses=workclass_options,
                           occupations=occupation_options,
                           educations=education_options)

@app.route('/result', methods=['POST'])
def result():
    try:
        age = int(request.form['age'])
        education = request.form['education']
        education_num = education_map[education]
        gender = 1 if request.form['gender'] == 'Male' else 0
        hours = int(request.form['hours-per-week'])
        workclass = request.form['workclass']
        occupation = request.form['occupation']

        input_vector = [0] * len(feature_names)
        input_dict = dict(zip(feature_names, input_vector))
        input_dict['age'] = age
        input_dict['educational-num'] = education_num
        input_dict['gender'] = gender
        input_dict['hours-per-week'] = hours

        if f'workclass_{workclass}' in input_dict:
            input_dict[f'workclass_{workclass}'] = 1
        if f'occupation_{occupation}' in input_dict:
            input_dict[f'occupation_{occupation}'] = 1

        final_vector = [input_dict[feat] for feat in feature_names]

        pred = model.predict([final_vector])[0]
        prob = model.predict_proba([final_vector])[0]
        confidence = round(100 * max(prob), 2)

        if pred == 0:
            salary_range = "Your salary is likely in the range **0 - 50K**."
        else:
            salary_range = "Your salary is likely **above 50K**."

        report_text = f"""Employee Salary Prediction Estimator Report

Input Details:
- Age: {age}
- Education: {education}
- Gender: {"Male" if gender == 1 else "Female"}
- Weekly Work Hours: {hours}
- Workclass: {workclass}
- Occupation: {occupation}

Prediction:
{salary_range}
Model Confidence: {confidence}%
"""

        return render_template("result.html", confidence=confidence,
                               salary_range=salary_range,
                               report_content=report_text,
                               age=age, education=education, gender="Male" if gender == 1 else "Female",
                               hours=hours, workclass=workclass, occupation=occupation)
    except Exception as e:
        return f" Error: {str(e)}"

@app.route('/download', methods=['POST'])
def download():
    report_content = request.form['report_content']
    buffer = io.BytesIO()
    buffer.write(report_content.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Salary_Prediction_Report.txt", mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
