import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("C:/Users/HP/OneDrive/Documents/loan_data_new.csv")

print("Dataset loaded successfully!")
print(df.head())


# --------------------------------
# 2. Create Text Data
# --------------------------------

df["ApplicationText"] = (
    "gender_" + df["Gender"].astype(str) + " "
    "education_" + df["Education"].astype(str) + " "
    "home_" + df["Home Onwership"].astype(str) + " "
    "intent_" + df["Loan Intent"].astype(str) + " "
    "previous_" + df["Previous Loan"].astype(str)
)


# --------------------------------
# 3. Select Numerical Features
# --------------------------------

numeric_columns = [
    "Age",
    "Person Income",
    "Employee Experience",
    "Loan Amount",
    "Loan interest Rate",
    "Loan percentage",
    "Credit History",
    "Credit Score"
]

X_numeric = df[numeric_columns]

y = df["Loan Status"]

print("\nLoan Status counts:")
print(y.value_counts())

print("\Loan Status percentage:")
print(y.value_counts(normalize=True) * 100)

# --------------------------------
# 4. Convert Text into Numbers
# --------------------------------

vectorizer = CountVectorizer()

X_text = vectorizer.fit_transform(df["ApplicationText"])

print("Text converted into numerical features!")


# --------------------------------
# 5. Combine Text + Numerical Data
# --------------------------------

X = hstack([X_text, X_numeric.values])

print("Features combined successfully!")


# --------------------------------
# 6. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 7. Create Decision Tree
# --------------------------------

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)


# --------------------------------
# 8. Train Model
# --------------------------------

model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")


# --------------------------------
# 9. Prediction
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 10. Evaluation
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 11. Save Model
# --------------------------------

with open("loan_decision_tree.pkl", "wb") as file:
    pickle.dump(model, file)


# --------------------------------
# 12. Save Vectorizer
# --------------------------------

with open("loan_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved successfully!")
print("Vectorizer saved successfully!")

import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import hstack


# --------------------------------
# 1. Load Model
# --------------------------------

with open("loan_decision_tree.pkl", "rb") as file:
    model = pickle.load(file)


# --------------------------------
# 2. Load Vectorizer
# --------------------------------

with open("loan_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# --------------------------------
# 3. Streamlit Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


# --------------------------------
# 4. Application Title
# --------------------------------

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant details to predict the loan status."
)


# --------------------------------
# 5. User Inputs
# --------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

income = st.number_input(
    "Person Income",
    min_value=0,
    value=50000
)

experience = st.number_input(
    "Employee Experience",
    min_value=0,
    max_value=50,
    value=2
)

home = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

interest_rate = st.number_input(
    "Loan Interest Rate",
    min_value=0.0,
    max_value=30.0,
    value=10.0
)

loan_percentage = st.number_input(
    "Loan Percentage",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

credit_history = st.number_input(
    "Credit History",
    min_value=0,
    max_value=10,
    value=3
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

previous_loan = st.selectbox(
    "Previous Loan",
    ["Yes", "No"]
)


# --------------------------------
# 6. Prediction Button
# --------------------------------

if st.button("🔍 Predict Loan Status"):

    # Create text from categorical values

    application_text = (
        "gender_" + gender + " "
        "education_" + education + " "
        "home_" + home + " "
        "intent_" + loan_intent + " "
        "previous_" + previous_loan
    )


    # --------------------------------
    # 7. Convert Text into Numbers
    # --------------------------------

    text_vector = vectorizer.transform(
        [application_text]
    )


    # --------------------------------
    # 8. Create Numerical Data
    # --------------------------------

    numeric_data = pd.DataFrame(
        [[
            age,
            income,
            experience,
            loan_amount,
            interest_rate,
            loan_percentage,
            credit_history,
            credit_score
        ]],
        columns=[
            "Age",
            "Person Income",
            "Employee Experience",
            "Loan Amount",
            "Loan interest Rate",
            "Loan percentage",
            "Credit History",
            "Credit Score"
        ]
    )


    # --------------------------------
    # 9. Combine Features
    # --------------------------------

    final_input = hstack([
        text_vector,
        numeric_data.values
    ])


    # --------------------------------
    # 10. Prediction
    # --------------------------------

    prediction = model.predict(final_input)


    # --------------------------------
    # 11. Display Result
    # --------------------------------

    if prediction[0] == 1:

        st.success(
            "✅ Loan Status: APPROVED"
        )

    else:

        st.error(
            "❌ Loan Status: REJECTED"
        )
