import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""

y_true:      True labels for evaluation
y_pred:      Temporary predicted labels
final_y_pred: Final predicted labels 
best_k:      The optimal threshold factor (k)
"""


# Preprocessing Functions
def fit_preprocess(data_path):
    """
    Preprocesses the data to calculate mean, standard deviation, and variance for each feature in X.
    """
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # Assuming the last column is the target variable
    scaler = StandardScaler()
    scaler.fit(X)
    preprocess_params = {'scaler': scaler}
    return preprocess_params


def load_and_preprocess(data_path, preprocess_params):
    """
    Loads the data and preprocesses X using the provided scaler.
    """
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # Assuming the last column is the label
    scaler = preprocess_params['scaler']
    X_scaled = scaler.transform(X)
    return X_scaled, y.to_numpy()


# Probability Density Function (PDF) Model
def probability_density(X, mean, var):
    """
    Computes Mahalanobis distance and the probability density function (PDF)
    using Mahalanobis distance for each point in the data.
    """
    n = len(mean)  # Dimensionality of the data
    X_centered = X - mean  # Center data by subtracting the mean
    var = np.diag(var)  # Convert variance to diagonal covariance matrix if necessary
    p = (2 * np.pi)**(-n / 2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X_centered, np.linalg.pinv(var)) * X_centered, axis=1))
    return p


def fit_model_pdf(X):
    """
    Fit the model by calculating the mean and variance for each feature.
    """
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    return {'mean': mean, 'var': var}


def predict_pdf(X, model, y_true=None):
    """
    Predict anomalies using the probability density function and dynamic thresholding.
    """
    mean = model['mean']
    std = np.sqrt(model['var'])  # Standard deviation from variance

    best_k = None
    best_accuracy = 0
    final_y_pred = None

    # Iterate over different k values
    for k in np.linspace(1, 4, num=10): 
        lower_bound = mean - k * std
        upper_bound = mean + k * std

        # Anomalies: Points outside the range
        anomalies = (np.any(X < lower_bound, axis=1) | np.any(X > upper_bound, axis=1)).astype(int)

        accuracy = accuracy_score(y_true, anomalies) if y_true is not None else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            final_y_pred = anomalies
            
   

    if final_y_pred is None:
        print("Warning: No valid threshold found.")
        final_y_pred = np.zeros(len(X), dtype=int)  # Default to all normal

    print(f"PDF Model - Best threshold k: {best_k}, Accuracy: {best_accuracy}")
    return final_y_pred


# KMeans Clustering Model
def fit_model_kmeans(X, n_clusters=5):
    """
    Fit KMeans clustering model.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans


def predict_kmeans_with_threshold(X, model, y_true=None):
    """
    Predict anomalies using KMeans clustering with dynamic thresholding.
    """
    distances = model.transform(X).min(axis=1)

    best_threshold_factor = None
    best_accuracy = 0
    final_y_pred = None

    for threshold_factor in np.arange(1, 4, 0.2):
        threshold = np.mean(distances) + threshold_factor * np.std(distances)
        y_pred = (distances > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred) if y_true is not None else 0
        
    

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_factor = threshold_factor
            final_y_pred = y_pred

    print(f"KMeans - Best threshold factor: {best_threshold_factor}, Accuracy: {best_accuracy}")
    return final_y_pred


# PCA-Based Model
def fit_model_pca(X, n_components=2):
    """
    Fit PCA model.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def predict_pca(X, model, y_true=None):
    """
    Predict anomalies using PCA reconstruction error.
    """
    X_pca = model.transform(X)
    X_reconstructed = model.inverse_transform(X_pca)
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)

    best_k = None
    best_accuracy = 0
    final_y_pred = None

    for k in np.linspace(1, 6, num=20):
        threshold = np.mean(mse) + k * np.std(mse)
        y_pred = (mse > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred) if y_true is not None else 0
        

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            final_y_pred = y_pred

    print(f"PCA - Best threshold k: {best_k}, Accuracy: {best_accuracy}")
    return final_y_pred



# PCA-Based Model
def predict_pca(X, model, y_true=None):
    """
    Predict anomalies using PCA reconstruction error.
    """
    X_pca = model.transform(X)
    X_reconstructed = model.inverse_transform(X_pca)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)

    best_k = None
    best_accuracy = 0
    final_y_pred = None

    for k in np.linspace(1, 6, num=20):
        threshold = np.mean(reconstruction_error) + k * np.std(reconstruction_error)
        y_pred = (reconstruction_error > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred) if y_true is not None else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            final_y_pred = y_pred

    print(f"PCA - Best threshold k: {best_k}, Accuracy: {best_accuracy}")
    return final_y_pred


def fit_model(X, model_type="PDF"):# fitting all models in one
    if model_type == "PDF":
        return fit_model_pdf(X)
    elif model_type == "KMeans":
        return fit_model_kmeans(X)
    elif model_type == "PCA":
        return fit_model_pca(X)
    else:
        raise ValueError("Unknown model type")
        
def predict(X, model, model_type="PDF", y_true=None):#fitting all models in one
    if model_type == "PDF":
        return predict_pdf(X, model, y_true)
    elif model_type == "KMeans":
        return predict_kmeans_with_threshold(X, model, y_true)
    elif model_type == "PCA":
        return predict_pca(X, model, y_true)
    else:
        raise ValueError("Unknown model type")



def calculate_metrics(y_true, y_pred):
    """
    Manually calculates precision, recall, and F1 score.
    """
    # True Positives, False Positives, False Negatives
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    # Precision
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    # Recall
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the model performance using accuracy, and manually computed precision, recall, and F1 score.
    """
    # Accuracy
    accuracy = sum(y_true == y_pred) / len(y_true)

    # Manually calculate metrics
    precision, recall, f1 = calculate_metrics(y_true, y_pred)

    # Print results
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 30)

    return accuracy, precision, recall, f1


def main():
    """
    Train and evaluate anomaly detection models using only the training dataset.
    """
    start_time = time.time() 
    train_path = "https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv"


    # Preprocessing
    preprocess_params = fit_preprocess(train_path)
    X_train, y_train = load_and_preprocess(train_path, preprocess_params)

    # PDF Model
    print("-" * 20 + "\nPDF Model\n" + "-" * 20)
    model_pdf = fit_model_pdf(X_train)
    y_train_pred_pdf = predict_pdf(X_train, model_pdf, y_train)
    evaluate_model(y_train, y_train_pred_pdf, "PDF")
    anomalies_pdf = (y_train_pred_pdf == 1).sum()
    print(f"Total anomalies detected in PDF model: {anomalies_pdf}")

    # KMeans Clustering
    print("-" * 20 + "\nKMeans Clustering Model\n" + "-" * 20)
    model_kmeans = fit_model_kmeans(X_train)
    y_train_pred_kmeans = predict_kmeans_with_threshold(X_train, model_kmeans, y_train)
    evaluate_model(y_train, y_train_pred_kmeans, "KMeans")
    anomalies_kmeans = (y_train_pred_kmeans == 1).sum()
    print(f"Total anomalies detected in KMeans model: {anomalies_kmeans}")

    # PCA Model
    print("-" * 20 + "\nPCA Model\n" + "-" * 20)
    model_pca = fit_model_pca(X_train)
    y_train_pred_pca = predict_pca(X_train, model_pca, y_train)
    evaluate_model(y_train, y_train_pred_pca, "PCA")
    anomalies_pca = (y_train_pred_pca == 1).sum()
    print(f"Total anomalies detected in PCA model: {anomalies_pca}")

    # Compare models
    accuracies = {
        "PDF": sum(y_train == y_train_pred_pdf) / len(y_train),
        "KMeans": sum(y_train == y_train_pred_kmeans) / len(y_train),
        "PCA": sum(y_train == y_train_pred_pca) / len(y_train)
    }
    best_model = max(accuracies, key=accuracies.get)
    print(f"\nThe best model is: {best_model} with an accuracy of {accuracies[best_model]:.4f}")
    
    end_time = time.time()  # Record the end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
