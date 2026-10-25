"""
Model Evaluation and Verification Script
Tests accuracy, per-letter precision, recall, and confusion matrix.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from train_model import ASLSignNet

def evaluate_model(model_path='python/models/best_asl_model.pth',
                   data_path='python/data/asl_landmarks.csv',
                   labels_path='python/data/asl_labels.csv'):
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist. Run train_model.py first.")
        return
        
    print(f"Evaluating {model_path} against {data_path}...")
    X = pd.read_csv(data_path, header=None).values.astype(np.float32)
    y_df = pd.read_csv(labels_path)
    y = y_df['label'].values.astype(np.int64)
    
    model = ASLSignNet()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        inputs = torch.from_numpy(X)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    accuracy = np.mean(preds == y) * 100.0
    print(f"\n==========================================")
    print(f"  OVERALL ASL A-Z ACCURACY: {accuracy:.2f}%")
    print(f"==========================================\n")
    
    print("Classification Report:")
    print(classification_report(y, preds, target_names=classes, digits=3))
    
    # Check sample predictions
    print("\nSample Predictions for Each Letter:")
    for letter_idx, letter in enumerate(classes):
        # find first instance of this letter
        idx = np.where(y == letter_idx)[0][0]
        pred_letter = classes[preds[idx]]
        conf = probs[idx][preds[idx]] * 100.0
        status = "PASS" if pred_letter == letter else "FAIL"
        print(f"  Target: {letter} | Predicted: {pred_letter} | Confidence: {conf:5.1f}% | [{status}]")

if __name__ == '__main__':
    evaluate_model()
