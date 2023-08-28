import string
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt


# Remover pontuação dos emails
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Criando nuvem de palavras
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

def wordcloud(df):
    # Creting seprate dataset for Spam and Non Spam emails, to perform analysis 
    Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])
    Non_Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])

    # Creating Non_Spam email dataset 
    for i in range(len(df)):
        if(df['Labels'][i] == 0):
            new_row = {'Email_text':df['Email_text'][i], 'Email_Subject':df['Email_Subject'][i], 'Labels':df['Labels'][i]}
            Non_Spam = Non_Spam._append(new_row, ignore_index=True)

    # Creating Spam email dataset 
    for i in range(len(df)):
        if(df['Labels'][i] == 1):
            new_row = {'Email_text':df['Email_text'][i], 'Email_Subject':df['Email_Subject'][i], 'Labels':df['Labels'][i]}
            Spam = Spam._append(new_row, ignore_index=True)

    # Creating stopwords corpus
    STOPWORDS = set()
    more_stopwords = {'re' , 's' , 'subject','hpl','hou','enron'}
    STOPWORDS = STOPWORDS.union(more_stopwords)

    # Creating spam subject corpus 
    Subject_corpus_spam = ""
    for i in range(len(Spam)):
        Subject_corpus_spam = Subject_corpus_spam + Spam['Email_Subject'][i]

    # Creating spam text corpus 
    Text_corpus_spam = ""
    for i in range(len(Spam)):
        Text_corpus_spam = Text_corpus_spam + Spam['Email_text'][i]

    # Creating non-spam subject corpus 
    Subject_corpus_non_spam = ""
    for i in range(len(Non_Spam)):
        Subject_corpus_non_spam = Subject_corpus_non_spam + Non_Spam['Email_Subject'][i]

    # Creating non-spam text corpus 
    Text_corpus_non_spam = ""
    for i in range(len(Non_Spam)):
        Text_corpus_non_spam = Text_corpus_non_spam + Non_Spam['Email_text'][i]

    # Plotting word cloud for Spam Subject corpus
    Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_spam , STOPWORDS)

    # Plotting word cloud for Spam Text corpus
    Spam_Text_wordcloud = make_wordcloud (Text_corpus_spam , STOPWORDS)

    # Plotting word cloud for Non Spam Subject corpus
    Non_Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_non_spam , STOPWORDS)

    # Plotting word cloud for Non Spam Text corpus
    Non_Spam_Subject_wordcloud = make_wordcloud (Text_corpus_non_spam , STOPWORDS)

    #plotting word cloud for Spam Subject corpus
    Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_spam , STOPWORDS)
    plt.figure(figsize=(13, 13))
    plt.title("Palavras mais comuns no Assunto de emails Spam", fontdict={'size': 20, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
    plt.imshow(Spam_Subject_wordcloud)
    plt.axis("off")
    plt.show()

    #plotting word cloud for Spam Text corpus
    Spam_Text_wordcloud = make_wordcloud (Text_corpus_spam , STOPWORDS)
    plt.figure(figsize=(13, 13))
    plt.title("Palavras mais comuns em emails Spam", fontdict={'size': 20, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
    plt.imshow(Spam_Text_wordcloud)
    plt.axis("off")
    plt.show()

    #plotting word cloud for Non Spam Subject corpus
    Non_Spam_Subject_wordcloud = make_wordcloud (Subject_corpus_non_spam , STOPWORDS)
    plt.figure(figsize=(13, 13))
    plt.title("Palavras mais comuns no Assunto de emails Nao-Spam", fontdict={'size': 20, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
    plt.imshow(Non_Spam_Subject_wordcloud)
    plt.axis("off")
    plt.show()

    #plotting word cloud for Non Spam Text corpus
    Non_Spam_Subject_wordcloud = make_wordcloud (Text_corpus_non_spam , STOPWORDS)
    plt.figure(figsize=(13, 13))
    plt.title("Palavras mais comuns em emails Nao-Spam", fontdict={'size': 20, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
    plt.imshow(Non_Spam_Subject_wordcloud)
    plt.axis("off")
    plt.show()

def preprocess_dataset(df):
    df1 = df.copy()
    df1 = df1.drop('Unnamed: 0', axis=1)
    df1 = df1.drop('label', axis=1)

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
    df1.columns = ["Email_text" , "Labels" , "Email_Subject"]

    df1['Email_Subject'] = df1['Email_Subject'].str.lower()
    df1['Email_text'] = df1['Email_text'].str.lower()

    df1['Email_Subject'] = df1['Email_Subject'].apply(remove_punctuations)
    df1['Email_text'] = df1['Email_text'].apply(remove_punctuations)
    wordcloud(df1)
    return df1