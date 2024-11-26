import requests
from sklearn.metrics import accuracy_score

# Download the script dynamically from GitHub
url = 'https://raw.githubusercontent.com/Nanda1000/Coursework/main/MLchem(3).py'
response = requests.get(url)

# Check if the download was successful
if response.status_code == 200:
    exec(response.text)  # Dynamically execute the script in the current environment
else:
    raise Exception(f"Failed to download script. Status code: {response.status_code}")

# Define the dataset path
train_path = r'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'

# Call the required functions from the loaded script
preprocess_params = fit_preprocess(train_path)
X_train, y_train = load_and_preprocess(train_path, preprocess_params)
model = fit_model(X_train)
y_train_pred = predict(X_train, model)

# Calculate and display accuracy
print('Accuracy is:', accuracy_score(y_train, y_train_pred))


