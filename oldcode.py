import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
plt.rcParams['figure.figsize'] = [10,6]
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np


# creating word cloud for given corpus  
def make_wordcloud(corpus , stopword):
    word_cloud = WordCloud(
            width=3000,
            height=2000,
            random_state=1,
            background_color="black",
            colormap="Pastel1",
            collocations=False,
            stopwords=stopword,
            ).generate(corpus)
    return word_cloud

# This function swaps 2 columns inside the dataframe
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

#This function removes punctuation from string
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

csv_file_path = './spam_ham_dataset.csv'

df = pd.read_csv(csv_file_path)
value_counts=df["label"].value_counts()
# print(df.head())

# print('\n',value_counts)

# create a bar chart
# value_counts.plot.bar()

# set the title and axis labels
# plt.title("Email Label Counts")
# plt.xlabel("Label")
# plt.ylabel("Count")

# display the chart
# plt.show()

# Dropping columns that are not needed
df1 = df.copy()
df1 = df1.drop('Unnamed: 0', axis=1)
df1 = df1.drop('label', axis=1)
# print(df1.head())

# Creating a new feature, extracting subject of each email
subjects = []
for i in range(len(df1)):
    ln = df1["text"][i]
    line = ""
    for i in ln:
        if(i == '\r'):
            break
        line = line + i
    line = line.replace("Subject" , "")
    subjects.append(line)

df1['Subject'] = subjects
# Renaming the dataframe columns
df1.columns = ["Email_text" , "Labels" , "Email_Subject"]
# print(df.head())

# Converting all strings to lowercase
df1['Email_Subject'] = df1['Email_Subject'].str.lower()
df1['Email_text'] = df1['Email_text'].str.lower()

# Removing Punctuation from the data
df1['Email_Subject'] = df1['Email_Subject'].apply(remove_punctuations)
df1['Email_text'] = df1['Email_text'].apply(remove_punctuations)

# print(df)

# Creting seprate dataset for Spam and Non Spam emails, to perform analysis 
# Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])
# Non_Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])

# Creating Non_Spam email dataset 
# for i in range(len(df)):
#     if(df['Labels'][i] == 0):
#         new_row = {'Email_text':df['Email_text'][i], 'Email_Subject':df['Email_Subject'][i], 'Labels':df['Labels'][i]}
#         Non_Spam = Non_Spam._append(new_row, ignore_index=True)

# Creating Spam email dataset 
# for i in range(len(df)):
#     if(df['Labels'][i] == 1):
#         new_row = {'Email_text':df['Email_text'][i], 'Email_Subject':df['Email_Subject'][i], 'Labels':df['Labels'][i]}
#         Spam = Spam._append(new_row, ignore_index=True)

# Creating stopwords corpus
# more_stopwords = {'re' , 's' , 'subject','hpl','hou','enron'}
# STOPWORDS = STOPWORDS.union(more_stopwords)

# Creating spam subject corpus 
# Subject_corpus_spam = ""
# for i in range(len(Spam)):
#     Subject_corpus_spam = Subject_corpus_spam + Spam['Email_Subject'][i]

# Creating spam text corpus 
# Text_corpus_spam = ""
# for i in range(len(Spam)):
#     Text_corpus_spam = Text_corpus_spam + Spam['Email_text'][i]

# Creating non-spam subject corpus 
# Subject_corpus_non_spam = ""
# for i in range(len(Non_Spam)):
#     Subject_corpus_non_spam = Subject_corpus_non_spam + Non_Spam['Email_Subject'][i]

# Creating non-spam text corpus 
# Text_corpus_non_spam = ""
# for i in range(len(Non_Spam)):
#     Text_corpus_non_spam = Text_corpus_non_spam + Non_Spam['Email_text'][i]

# Plotting word cloud for Spam Subject corpus
#Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_spam , STOPWORDS)

# Plotting word cloud for Spam Text corpus
#Spam_Text_wordcloud = make_wordcloud (Text_corpus_spam , STOPWORDS)

# Plotting word cloud for Non Spam Subject corpus
#Non_Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_non_spam , STOPWORDS)

# Plotting word cloud for Non Spam Text corpus
#Non_Spam_Subject_wordcloud = make_wordcloud (Text_corpus_non_spam , STOPWORDS)

# Split email dataset 
X_train, X_test , y_train, y_test = train_test_split(df1['Email_text'], df1['Labels'] , test_size=0.3)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Initialize variables to store results
raw_results = []
preprocessed_results = []

num_repeats = 5

for repeat in range(num_repeats):
    # Shuffle the dataset randomly for both raw and preprocessed datasets
    raw_shuffled_indices = np.random.permutation(len(df))
    preprocessed_shuffled_indices = np.random.permutation(len(df1))
    
    # Initialize K-Fold cross-validation
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    
    raw_fold_results = []
    preprocessed_fold_results = []
    
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    for train_index, test_index in kf.split(raw_shuffled_indices):
        # Split dataset into training and testing sets for both raw and preprocessed datasets
        raw_train_indices, raw_test_indices = raw_shuffled_indices[train_index], raw_shuffled_indices[test_index]
        preprocessed_train_indices, preprocessed_test_indices = preprocessed_shuffled_indices[train_index], preprocessed_shuffled_indices[test_index]
        
        # Extract the corresponding data based on the shuffled indices
        raw_X_train, raw_X_test = df.iloc[raw_train_indices]['text'], df.iloc[raw_test_indices]['text']
        preprocessed_X_train, preprocessed_X_test = df1.iloc[preprocessed_train_indices]['Email_text'], df1.iloc[preprocessed_test_indices]['Email_text']
        
        # Corrected: Assign labels to y_train and y_test
        y_kfold_train, y_kfold_test = df.iloc[raw_train_indices]['label_num'], df.iloc[raw_test_indices]['label_num']
        y_preprocessed_train, y_preprocessed_test = df1.iloc[preprocessed_train_indices]['Labels'], df1.iloc[preprocessed_test_indices]['Labels']


        # Fit and transform the vectorizer on the training data
        X_raw_train_vectorized = vectorizer.fit_transform(raw_X_train)
        X_preprocessed_train_vectorized = vectorizer.transform(preprocessed_X_train)

        # Transform the test data using the same vectorizer
        X_raw_test_vectorized = vectorizer.transform(raw_X_test)
        X_preprocessed_test_vectorized = vectorizer.transform(preprocessed_X_test)

    
        # Train and evaluate each algorithm on the raw dataset
        raw_model = MultinomialNB().fit(X_raw_train_vectorized, y_kfold_train)
        raw_predictions = raw_model.predict(X_raw_test_vectorized)
        raw_accuracy = accuracy_score(y_kfold_test, raw_predictions)
        raw_fold_results.append(raw_accuracy)
        
        # Train and evaluate each algorithm on the preprocessed dataset
        preprocessed_model = MultinomialNB().fit(X_preprocessed_train_vectorized, y_preprocessed_train)
        preprocessed_predictions = preprocessed_model.predict(X_preprocessed_test_vectorized)
        preprocessed_accuracy = accuracy_score(y_preprocessed_test, preprocessed_predictions)
        preprocessed_fold_results.append(preprocessed_accuracy)
    
    raw_results.append(raw_fold_results)
    preprocessed_results.append(preprocessed_fold_results)

# Print the average results across all repeats
raw_avg_results = np.mean(raw_results, axis=0)
preprocessed_avg_results = np.mean(preprocessed_results, axis=0)
# print('Average Raw Results:', raw_avg_results)
# print('Average Preprocessed Results:', preprocessed_avg_results)
# Gráfico das médias raw_avg_results e preprocessed_avg_results
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_folds+1), raw_avg_results, label='Raw Dataset')
# plt.plot(range(1, num_folds+1), preprocessed_avg_results, label='Preprocessed Dataset')
# plt.xlabel('Fold')
# plt.ylabel('Accuracy')
# plt.title('Average Accuracy Comparison: Raw vs. Preprocessed')
# plt.legend()
# plt.show()



# Convertendo o texto em vetores usando CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Criando e treinando o modelo de Árvore de Decisão
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_vectorized, y_train)

# Fazendo previsões no conjunto de teste
y_pred = tree_model.predict(X_test_vectorized)

# Avaliando a acurácia do modelo
tree_avg_results = accuracy_score(y_test, y_pred)
print("Acurácia do modelo de Árvore de Decisão:", tree_avg_results)

# Criando e treinando o modelo KNN
k = 5  # Número de vizinhos
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_vectorized, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn_model.predict(X_test_vectorized)

# Avaliando a acurácia do modelo
knn_avg_results = accuracy_score(y_test, y_pred)
print("Acurácia do modelo KNN:", knn_avg_results)

# Gráfico de comparação entre accuracy do KNN e accuracy do Decision Tree

models = ['KNN', 'Decision Tree']
accuracies = [knn_avg_results, tree_avg_results]

plt.figure(figsize=(8, 5))
plt.scatter(models, accuracies, color='blue', marker='o')
plt.xlabel('Model')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Comparison: KNN vs. Decision Tree')
plt.ylim(0, 1)  # Definindo o intervalo do eixo y de 0 a 1 (valor máximo da acurácia)
plt.grid(True)
plt.show()