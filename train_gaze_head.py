import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from load_mpiigaze import load_mpiigaze_data

def train_gaze_rf():
    base = os.path.join("Dataset", "MPIIGAZE", "MPIIGaze")
    
    print("Loading datasets...")
    # Load up to 1500 samples per participant to keep training fast (15 * 1000 = 15000 samples)
    X, y = load_mpiigaze_data(base, max_samples=1000)
    
    print(f"Total samples loaded: {len(X)}")
    print(f"Feature size: {X.shape[1]}")
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("Training Random Forest...")
    # Random Forest parameters: max_depth=20 prevents massive models, n_estimators=100 is fast
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gaze_rf.joblib")
    joblib.dump(rf, out_path)
    print(f"\nSaved model to {out_path}")

if __name__ == "__main__":
    train_gaze_rf()
