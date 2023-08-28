from decision_tree import evaluate_decision_tree
from kfold_cross_validation_evaluation import evaluate_kfold_naive_bayes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from knn import evaluate_knn
from preprocessing import preprocess_dataset

def plot_model_comparison(models, accuracies, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(models, accuracies, color='blue', marker='o')
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def load_and_preprocess_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df1 = preprocess_dataset(df)
    return df, df1

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['Email_text'], df['Labels'], test_size=0.3)
    return X_train, X_test, y_train, y_test

def evaluate_nb_kfold(df, df1):
    raw_avg_results, preprocessed_avg_results, raw_avg_f1_scores, preprocessed_avg_f1_scores, raw_avg_execution_times, preprocessed_avg_execution_times = evaluate_kfold_naive_bayes(df, df1)
    return raw_avg_results, preprocessed_avg_results, raw_avg_f1_scores, preprocessed_avg_f1_scores, raw_avg_execution_times, preprocessed_avg_execution_times

def plot_comparison_graphs(raw_results, preprocessed_results, raw_f1_scores, preprocessed_f1_scores, raw_exec_times, preprocessed_exec_times):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Comparação dos Resultados')

    axs[0, 0].plot(raw_results, label='Raw Dataset', marker='o')
    axs[0, 0].plot(preprocessed_results, label='Preprocessed Dataset', marker='o')
    axs[0, 0].set_ylabel('Média de Acurácia')
    axs[0, 0].set_title('Média de Acurácia')

    axs[0, 1].plot(raw_f1_scores, label='Raw Dataset', marker='o')
    axs[0, 1].plot(preprocessed_f1_scores, label='Preprocessed Dataset', marker='o')
    axs[0, 1].set_ylabel('Média de F1-Score')
    axs[0, 1].set_title('Média de F1-Score')

    axs[1, 0].plot(raw_exec_times, label='Raw Dataset', marker='o')
    axs[1, 0].plot(preprocessed_exec_times, label='Preprocessed Dataset', marker='o')
    axs[1, 0].set_xlabel('Repetições')
    axs[1, 0].set_ylabel('Média de Tempo de Execução (segundos)')
    axs[1, 0].set_title('Média de Tempo de Execução')

    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])

    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    print(X_train_vectorized)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized

def evaluate_decision_tree_model(X_train, y_train, X_test, y_test):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    tree_results, decisiontree_f1_score, decisiontree_execution_time = evaluate_decision_tree(tree_model, X_test, y_test)
    return tree_results, decisiontree_f1_score, decisiontree_execution_time

def evaluate_knn_models(k_values, X_train, y_train, X_test, y_test):
    knn_accuracies = []
    knn_f1_scores = []

    for k in k_values:
        knn_results, knn_f1_score, knn_execution_time = evaluate_knn(k, X_train, y_train, X_test, y_test)
        print(f"Tempo de execução do KNN com k={k}: {knn_execution_time}")
        knn_accuracies.append(knn_results)
        knn_f1_scores.append(knn_f1_score)

    knn_avg_f1_score = sum(knn_f1_scores) / len(knn_f1_scores)
    return knn_accuracies, knn_f1_scores, knn_avg_f1_score

csv_file_path = './datasets/spam_ham_dataset.csv'

df, df1 = load_and_preprocess_data(csv_file_path)
X_train, X_test, y_train, y_test = split_data(df1)
raw_avg_results, preprocessed_avg_results, raw_avg_f1_scores, preprocessed_avg_f1_scores, raw_avg_execution_times, preprocessed_avg_execution_times = evaluate_nb_kfold(df, df1)
plot_comparison_graphs(raw_avg_results, preprocessed_avg_results, raw_avg_f1_scores, preprocessed_avg_f1_scores, raw_avg_execution_times, preprocessed_avg_execution_times)
X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)
tree_results, decisiontree_f1_score, decisiontree_execution_time = evaluate_decision_tree_model(X_train_vectorized, y_train, X_test_vectorized, y_test)
k_values = [5, 10, 15]
knn_accuracies, knn_f1_scores, knn_avg_f1_score = evaluate_knn_models(k_values, X_train_vectorized, y_train, X_test_vectorized, y_test)
