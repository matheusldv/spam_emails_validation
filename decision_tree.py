import time
from sklearn.metrics import accuracy_score, f1_score

def evaluate_decision_tree(model, X_test_vectorized, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test_vectorized)
    end_time = time.time()
    execution_time = end_time - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1, execution_time
