from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(y_test, y_pred, model_name):
    # Evaluate model performance and print the report
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
