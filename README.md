# Employee Salary Prediction Estimator

This project is a Flask web application that predicts whether an individual's salary is greater than 50K or less than or equal to 50K based on demographic and employment-related inputs.

The model is trained using a cleaned version of the UCI Adult dataset and uses a Gradient Boosting Classifier for prediction.

---

## Features

- Web interface built with Flask, HTML, and CSS (dark theme)
- Predicts salary class based on age, education, gender, workclass, occupation, and hours-per-week
- Shows prediction confidence score
- Option to download a plain text prediction report

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML & CSS (no JavaScript)

---

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/SuprajaSandhya/employee-salary-predictor.git
   cd employee-salary-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

Then open your browser and go to `http://127.0.0.1:5000`

---

## Project Structure

```
employee-salary-predictor/
│
├── app.py
├── model.pkl
├── model_training.ipynb
├── requirements.txt
├── README.md
├── static/
│   └── styles.css
├── templates/
│   ├── index.html
│   └── result.html
└── .gitignore
```

---

## Sample Input

- Age: 35  
- Education: Bachelors  
- Gender: Female  
- Hours per Week: 40  
- Workclass: Private  
- Occupation: Tech-support  

**Prediction:** >50K  
**Confidence Score:** 87%

---

## License

This project is licensed under the MIT License.  
Developed by Supraja Sandhya.
