import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def evaluate_knn(k, X_train_vectorized, y_train, X_test_vectorized, y_test):
    start_time = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_vectorized, y_train)
    y_pred_knn = knn_model.predict(X_test_vectorized)
    knn_avg_results = accuracy_score(y_test, y_pred_knn)
    end_time = time.time()
    execution_time = end_time - start_time
    
    f1_knn = f1_score(y_test, y_pred_knn) 
    
    return knn_avg_results, f1_knn, execution_time