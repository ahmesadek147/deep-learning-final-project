from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import torch

def calculate_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, F1-score, and confusion matrix.
    Can accept PyTorch tensors or lists/numpy arrays.
    """
    # Convert PyTorch tensors to CPU numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Calculate overall metrics (using 'macro' average to treat all 200 bird classes equally)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix
    }
    
    return metrics_dict

if __name__ == "__main__":
    # Sanity check with dummy data (e.g., simulating 5 samples across 3 classes)
    sample_true = [0, 1, 2, 1, 0]
    sample_pred = [0, 2, 2, 1, 0]
    
    results = calculate_metrics(sample_true, sample_pred)
    print("Metrics Module Test Results:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"Confusion Matrix Shape: {results['confusion_matrix'].shape}")