import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix

# LOAD DATASET
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# ENCODE TARGET INTO YES:1 AND NO:0
df['Churn'] = df['Churn'].map({'Yes':1 , 'No':0})

# ENODE CATEGORICAL FEATURES
categorical_columns = df.select_dtypes(include=['object']).columns
for columns in categorical_columns:
    df[columns] = LabelEncoder().fit_transform(df[columns])

# SET FEATURES AND TARGET
features = ['gender', 'tenure', 'MonthlyCharges', 'Contract']
X = df[features]
y = df['Churn']

# Train-Test-Split
# 80% training(X_train,y_train)
# 20% training(X_test,y_test)
X_train,X_text,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# Features Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_text)

# Train Model 
model_rf = LogisticRegression(class_weight='balanced',max_iter=1000)
model_rf.fit(X_train,y_train)

# EVALUATION
prediction = model_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
print('CONFUSION_MATRIX')
print(confusion_matrix(y_test, prediction))

# Save model & scaler
joblib.dump(model_rf, "src/churn_model.pkl")
joblib.dump(scaler, "src/scaler.pkl")
print("\n✅ LogisticRegression model and scaler saved!")

print(df['Churn'].value_counts(normalize=True))

