🏦 **Bank Loan Approval Predictor**

This Streamlit web application predicts whether a bank loan application will be approved or not approved based on user inputs using a machine learning model.

🚀 **Features**

🧠 ML-Powered Prediction: Utilizes a pre-trained classification model to assess loan approval likelihood.

📋 User-Friendly Form: Intuitive input form to collect applicant data such as deposit status, job type, education level, and more.

📊 Confidence Score: Displays the model’s prediction along with the confidence percentage.

❄️ Celebratory UI: Balloons for approval, snow effect for rejection to enhance UX.

🧾 PDF Report Generator: Generates a downloadable loan prediction summary in PDF format.

📈 Feature Importance: Visualizes which features contribute most to the model's decision (if supported by model).

📥 Inputs Collected

Deposit Status

Previous Campaign Outcome

Housing Loan

Account Balance

Credit Default

Education Level

Personal Loan

Job Type

🛠️ **Technologies Used**
Python

Streamlit for UI

scikit-learn for machine learning

reportlab for generating PDF reports

matplotlib for feature importance visualization

pickle for loading the trained model

📄 **How to Run**
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py

📁 **Files**

app.py: Main application script.

loan_approval_model.pkl: Trained machine learning model.

requirements.txt: List of dependencies.
