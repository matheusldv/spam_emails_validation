import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import time

def evaluate_kfold_naive_bayes(df, df1):
    raw_results = []
    preprocessed_results = []
    raw_f1_scores = []
    preprocessed_f1_scores = []
    raw_execution_times = []
    preprocessed_execution_times = []

    num_repeats = 5

    for repeat in range(num_repeats):
        raw_shuffled_indices = np.random.permutation(len(df))
        preprocessed_shuffled_indices = np.random.permutation(len(df1))
        
        num_folds = 10
        kf = KFold(n_splits=num_folds)
        
        raw_fold_results = []
        preprocessed_fold_results = []
        raw_fold_f1_scores = []
        preprocessed_fold_f1_scores = []
        raw_fold_execution_times = []
        preprocessed_fold_execution_times = []
        
        vectorizer = CountVectorizer()

        for train_index, test_index in kf.split(raw_shuffled_indices):
            raw_train_indices, raw_test_indices = raw_shuffled_indices[train_index], raw_shuffled_indices[test_index]
            preprocessed_train_indices, preprocessed_test_indices = preprocessed_shuffled_indices[train_index], preprocessed_shuffled_indices[test_index]
            
            raw_X_train, raw_X_test = df.iloc[raw_train_indices]['text'], df.iloc[raw_test_indices]['text']
            preprocessed_X_train, preprocessed_X_test = df1.iloc[preprocessed_train_indices]['Email_text'], df1.iloc[preprocessed_test_indices]['Email_text']
            
            y_kfold_train, y_kfold_test = df.iloc[raw_train_indices]['label_num'], df.iloc[raw_test_indices]['label_num']
            y_preprocessed_train, y_preprocessed_test = df1.iloc[preprocessed_train_indices]['Labels'], df1.iloc[preprocessed_test_indices]['Labels']

            X_raw_train_vectorized = vectorizer.fit_transform(raw_X_train)
            X_preprocessed_train_vectorized = vectorizer.transform(preprocessed_X_train)

            X_raw_test_vectorized = vectorizer.transform(raw_X_test)
            X_preprocessed_test_vectorized = vectorizer.transform(preprocessed_X_test)

            start_time = time.time()
            raw_model = MultinomialNB().fit(X_raw_train_vectorized, y_kfold_train)
            raw_predictions = raw_model.predict(X_raw_test_vectorized)
            raw_accuracy = accuracy_score(y_kfold_test, raw_predictions)
            raw_f1 = f1_score(y_kfold_test, raw_predictions)
            end_time = time.time()
            raw_fold_execution_times.append(end_time - start_time)
            raw_fold_results.append(raw_accuracy)
            raw_fold_f1_scores.append(raw_f1)
            
            start_time = time.time()
            preprocessed_model = MultinomialNB().fit(X_preprocessed_train_vectorized, y_preprocessed_train)
            preprocessed_predictions = preprocessed_model.predict(X_preprocessed_test_vectorized)
            preprocessed_accuracy = accuracy_score(y_preprocessed_test, preprocessed_predictions)
            preprocessed_f1 = f1_score(y_preprocessed_test, preprocessed_predictions)
            end_time = time.time()
            preprocessed_fold_execution_times.append(end_time - start_time)
            preprocessed_fold_results.append(preprocessed_accuracy)
            preprocessed_fold_f1_scores.append(preprocessed_f1)
        
        raw_results.append(raw_fold_results)
        preprocessed_results.append(preprocessed_fold_results)
        raw_f1_scores.append(raw_fold_f1_scores)
        preprocessed_f1_scores.append(preprocessed_fold_f1_scores)
        raw_execution_times.append(raw_fold_execution_times)
        preprocessed_execution_times.append(preprocessed_fold_execution_times)

    raw_avg_results = np.mean(raw_results, axis=0)
    preprocessed_avg_results = np.mean(preprocessed_results, axis=0)
    raw_avg_f1_scores = np.mean(raw_f1_scores, axis=0)
    preprocessed_avg_f1_scores = np.mean(preprocessed_f1_scores, axis=0)
    raw_avg_execution_times = np.mean(raw_execution_times, axis=0)
    preprocessed_avg_execution_times = np.mean(preprocessed_execution_times, axis=0)
    
    return raw_avg_results, preprocessed_avg_results, raw_avg_f1_scores, preprocessed_avg_f1_scores, raw_avg_execution_times, preprocessed_avg_execution_times