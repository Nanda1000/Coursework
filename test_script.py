import requests
from sklearn.metrics import accuracy_score

# Download the script dynamically
url = 'https://raw.githubusercontent.com/Nanda1000/Coursework/refs/heads/main/MLchem(3).py?token=GHSAT0AAAAAAC2FRKBHN2ENKDJIXP3KKQM2Z2F6VRA'
response = requests.get(url)
exec(response.text)  # Load the script into the current Python environment

# Import required functions (assuming your script defines these functions)
from team_name import fit_preprocess, load_and_preprocess, fit_model, predict

# Define train path
train_path = r'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'

# Preprocessing
preprocess_params = fit_preprocess(train_path)
X_train, y_train = load_and_preprocess(train_path, preprocess_params)

# Model Training
model = fit_model(X_train)

# Predictions
y_train_pred = predict(X_train, model)

# Output Accuracy
print('Accuracy is: ', accuracy_score(y_train, y_train_pred))

