import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_metrics(model, X_test, y_test, label_encoder=None):
    # Predict
    y_pred = model.predict(X_test)

    # Convert predictions to original labels if encoder is given
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
        y_test = label_encoder.inverse_transform(y_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")

    # Save image
    output_path = "static/metrics/confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    return {
        "accuracy": acc,
        "report": report,
        "cm_image": output_path
    }
