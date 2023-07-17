import os
import nltk
import re
import string
import multiprocessing
import spacy

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import Phrases, LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from multiprocessing import cpu_count
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
sid = SentimentIntensityAnalyzer()

#####################
# Utility Functions #
#####################

def load_csvs_from_dir(directory):
    """
    Load a directory that contains csvs into one concatenated Pandas Dataframe
    Assumption is that all the csvs have the exact same columns
    """

    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Construct the file path
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file into a DataFrame
            sub_df = pd.read_csv(file_path)
            
            # Append the DataFrame to the list
            dfs.append(sub_df)

    # Concatenate all DataFrames into one
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df


def stratified_sample(df, group_column, sample_fraction, random_state=42):
    """
    Takes a dataframe and returns a random stratified sample
    based on the sample_fraction of the original dataframe
    """
    X = df.drop(columns=[group_column])
    y = df[group_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_fraction, random_state=random_state)

    _, sample_indices = next(stratified_splitter.split(X, y_encoded))

    sample_df = df.iloc[sample_indices]

    return sample_df

def frequency_group_cat_variable(df, cat_col, n):
    """
    Simplifies a categorical variable by taking the most frequent n
    categories and reducing the rest to 1
    df : pandas DataFrame
    cat_col : categorical column name
    n : int, number of top categories to include 
    """
    counts = df[cat_col].value_counts()
    top_n_categories = counts[:n].index.tolist()


    # Apply grouping and calculate entropy of the simplified data
    df['grouped_' + cat_col] = df[cat_col].apply(lambda x: x if x in top_n_categories else 'Other')
    return df

###########################
# Preprocessing Functions #
###########################

def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'URL', text)

def remove_punctuation(text):
    """Remove punctuation from the text."""
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text

def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'  # Dingbats
        u'\U000024C2-\U0001F251' 
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u2640-\u2642'
        u'\u2600-\u2B55'
        u'\u200d'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\ufe0f'  # dingbats
        u'\u3030'
        ']+', flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text)

def normalize_colors(text):
    """Normalize color descriptions."""
    color_mapping = {
        'grey': 'COLOR',
        'gry': 'COLOR',
        'gray': 'COLOR',
        'black': 'COLOR',
        'blck': 'COLOR',
        'blue': 'COLOR',
        'red': 'COLOR',
        'green': 'COLOR',
        'yellow': 'COLOR',
        'purple': 'COLOR',
        'pink': 'COLOR',
        'orange': 'COLOR',
        'white': 'COLOR',
        'brown': 'COLOR',
        'wht': 'COLOR',
        'blk': 'COLOR',
        'blu': 'COLOR',
        'rd': 'COLOR',
        'grn': 'COLOR',
        'ylw': 'COLOR',
        'purp': 'COLOR',
        'pnk': 'COLOR',
        'org': 'COLOR',
    }
    color_pattern = re.compile(r'\b(' + '|'.join(color_mapping.keys()) + r')\b', flags=re.IGNORECASE)
    return color_pattern.sub(lambda x: color_mapping[x.group().lower()], text)

def normalize_size(text):
    """Normalize size descriptions."""
    size_pattern = re.compile(r'\b(large|x+l|small|sml|medium|med|md|s|m|l|x+s)\b', flags=re.IGNORECASE)
    return size_pattern.sub('SIZE', text)

def expand_contractions(text):
    """Expand contractions in text."""
    contractions = {
        "ain't": "are not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    
    contractions_pattern = re.compile('({})'.format('|'.join(contractions.keys())), flags=re.IGNORECASE)
    return contractions_pattern.sub(lambda match: contractions[match.group(0).lower()], text)

def normalize_numbers(text):
    """Normalize numbers in text."""
    number_pattern = re.compile(r'\b\d+\b')
    return number_pattern.sub('NUMBER', text)

def normalize_percentages(text):
    """Normalize percentages in text."""
    percentage_pattern = re.compile(r'\b\d+%|\b%\d+') # Handles cases where % sign is before a number as well as after
    return percentage_pattern.sub('PERCENTAGE', text)

def conditional_lower(text):
    """Conditionally lower the text while keeping the normalizations uppercase"""
    def lowercase_if_not_category(match):
        word = match.group()
        if word.upper() in {'COLOR', 'PERCENTAGE', 'SIZE', 'NUMBER', 'URL', 'EMOJI'}:
            return word
        return word.lower()

    text = re.sub(r'(?i)\b(?<![\w%-])(?!COLOR\b|PERCENTAGE\b|SIZE\b|NUMBER\b|URL\b|EMOJI\b)\w+\b', lowercase_if_not_category, text)

    return text

def normalize_whitespace(text):
    """Normalize the whitespace in the text"""
    return ' '.join(text.split())

def preprocess_text(text):
    """Preprocess text by applying various normalization functions."""
    if type(text) != str:
        return np.nan
    else:
        text = remove_urls(text)
        text = remove_punctuation(text)
        text = remove_emojis(text)
        text = expand_contractions(text)
        text = normalize_numbers(text)
        text = normalize_percentages(text)
        text = normalize_colors(text)
        text = normalize_size(text)
        text = conditional_lower(text)
        text = normalize_whitespace(text)
        return text
    
def tokenize(text, stopwords=False):
    """Tokenizes and strips stopwords from text"""
    if type(text) == str:
        if stopwords:
            return [x for x in nltk.word_tokenize(text) if x not in stopwords]
        else:
            return nltk.word_tokenize(text)
    else:
        return np.nan

def lemmatize(tokenized_text):
    """Lemmatizes a list of tokens"""
    if type(tokenized_text) == list:
        return [lemmatizer.lemmatize(w) for w in tokenized_text]
    else:
        return np.nan

def process_text(text):
    """Tokenizes and lemmatizes an input text"""
    tokens = tokenize(text)
    lemmas = lemmatize(tokens)
    if type(lemmas) == list:
        return ' '.join(lemmas)
    else:
        return lemmas
    
def process_rating(rating):
    """
    Preprocessing function for the additional data rating columns
    """
    if isinstance(rating, str):
        regex_pattern = r'^[\d,.]+$'
        match = re.search(regex_pattern, rating)
        number = float(match.group().replace(',', '')) if match else np.nan
        return number
    else:
        return np.nan

def process_price(price):
    """
    Preprocessing function for the additional data price columns
    """
    if isinstance(price, str):
        regex_pattern = r'[\d,.]+'
        match = re.search(regex_pattern, price)
        number = float(match.group().replace(',', '')) if match else np.nan
        return number
    else:
        return np.nan

    
################################
# Feature Extraction Functions #
################################

def ner(texts):
    """
    Uses Spacy to perform Named Entity Recognition to extract Named Entities
    texts : list of strings
    nlp : spacy nlp object
    """
    docs = list(nlp.pipe([texts], disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]))
    entities = [[(ent.text, ent.label_) for ent in doc.ents] for doc in docs]
    return entities

def sentiment(text):
    """
    Performs Sentiment Analysis using VADER (due to size and speed)
    text : string
    sid : SentimentIntensityAnalyzer 
    """
    if type(text) == str:
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return (sentiment, compound_score)
    else:
        return np.nan

def extract_keywords_tfidf(matrix, tfidf, n_keywords=5):
    """
    Uses a numpy matrix of TFIDF scores to generate keywords

    matrix : numpy matrix of TFIDF scores
    tfidf : sklearn TfidfVectorizer pretrained
    n_keywords : number of keywords to extract
    """
    # Get the feature names (keywords) from the vectorizer
    feature_names = tfidf.get_feature_names_out()

    # Get the TF-IDF scores for each keyword in the texts
    tfidf_scores = matrix.sum(axis=0)
    tfidf_scores = np.asarray(tfidf_scores)[0]

    # Get the top N keywords based on their TF-IDF scores
    top_keywords_indices = tfidf_scores.argsort()[-n_keywords:]
    top_keywords = [feature_names[index] for index in top_keywords_indices]

    return top_keywords

def extract_most_informative_words(sentences, cluster_labels, num_keywords=5):
    """
    Uses TFIDF to extract keywords based on a corpus of sentences and a list of clusters
    those sentences belong to
    sentences : list of strings
    cluster_labels : list of labels (must be same len as sentences)
    num_keywords : how many keywords per cluster
    """

    # Create a TF-IDF vectorizer
    tfidf = TfidfVectorizer()

    # Fit the vectorizer on the sentences to obtain the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(sentences)

    # Get the feature names (words) from the vectorizer
    feature_names = tfidf.get_feature_names_out()

    # Store the most informative words for each cluster
    informative_words_per_cluster = {}

    # Iterate over each unique cluster label
    unique_clusters = set(cluster_labels)
    for cluster in unique_clusters:
        # Get the sentences belonging to the current cluster
        cluster_sentences = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster]

        # Fit the vectorizer on the cluster sentences to obtain the cluster-specific TF-IDF matrix
        cluster_tfidf_matrix = tfidf.transform(cluster_sentences)

        # Get the top N keywords (most informative words) for the cluster
        top_keywords = extract_keywords_tfidf(cluster_tfidf_matrix, tfidf, num_keywords)

        # Store the top keywords for the cluster
        informative_words_per_cluster[cluster] = top_keywords

    return informative_words_per_cluster

def parallelize(func, iterable, num_processes=None):
    """
    Parallelizes a function accross the available cores of a machine
    Greatly speeds up NER and Sentiment
    """
    total = len(iterable)
    with multiprocessing.Pool(num_processes) as pool, \
            tqdm(total=total, ncols=80, unit='item(s)', dynamic_ncols=True) as pbar:
        results = []
        for result in pool.imap(func, iterable):
            results.append(result)
            pbar.update()
    return results

def run_feature_extraction(df, text_feats=True, num_feats=True,
                           ner=True, sentiment=True):
    """
    Performs feature extraction on a dataframe 
    (currently assumes col names in the amazon dataset format)
    text_feats : bool, perform lemmatization on text cols
    num_feats : calculate numerical features on text
    ner : 
    """
    
    if sum([text_feats, num_feats, ner, sentiment]) == 0:
        print("Error: You have to choose at least one feature to extract")
        return
    
    if text_feats and not num_feats:
        print("Error: Numerical feats depend on text feats")
        return
    
    if (ner and not text_feats) or (sentiment and not text_feats):
        print("Error: NER and Sentiment depend on text features")
        return
    
    if text_feats:
        try:
            # Get text lemmas
            df["title_lemmas"] = df.cleaned_title.apply(process_text)
            df["bp_lemmas"] = df.cleaned_bp.apply(process_text)
            df["description_lemmas"] = df.cleaned_description.apply(process_text)

            # Concat text lemmas
            df['all_text'] = df['title_lemmas'].fillna('') + df['bp_lemmas'].fillna('') + df['description_lemmas'].fillna('')

            # Drop lemmas
            df = df.drop(columns=['title_lemmas', 'bp_lemmas', 'description_lemmas'])
        except:
            print("Error with text features")

    if num_feats:
        try:
            # Numerical feature calculation
            df['n_words'] = df['all_text'].apply(lambda x: len(x.split()))
            df['n_unique_words'] = df['all_text'].apply(lambda x: len(set(x.split())))
            df['pct_unique_words'] = df.n_unique_words / df.n_words
            df['char_count'] = df['all_text'].apply(len)
            df['sum_word_len'] = df['all_text'].apply(lambda x: sum([len(w) for w in x.split()]))
            df['avg_word_len'] = df['sum_word_len'] / df['n_words']
        except:
            print("Error with numerical features")
    
    if ner:
        try:
            ner_results = parallelize(ner, df.all_text.values)
            df['ner'] = ner_results
        except:
            print("Error with NER")
    
    if sentiment:
        try:
            sentiment_results = parallelize(sentiment, df.all_text.values)
            df['sentiment'] = sentiment_results
            df['sentiment_score'] = df.sentiment.apply(lambda x: x[1])
            df['sentiment'] = df.sentiment.apply(lambda x: x[0])
        except:
            print("Error with Sentiment")

    return df

#####################
# Doc2Vec Functions #
#####################

def transform_to_doc2vec_vectors(texts, doc2vec_model, batch_size=1000):
    """
    Transforms a list of texts into a list of Doc2Vec vectors
    texts : list of list of strings
    doc2vec_model : gensim Doc2Vec model object
    batch_size : int, size of batches to process text
    """
    vectors = []
    for i in tqdm(range(0, len(texts), 1000)):
        batch_texts = texts[i:i + 1000]
        inferred_vectors = [doc2vec_model.infer_vector(text) for text in batch_texts]
        vectors.append(inferred_vectors)
    vectors = np.concatenate(vectors)
    return vectors

def train_doc2vec_model(sentences, vector_size=100, window=5, min_count=5, epochs=10):
    # Preprocess sentences
    tagged_data = [TaggedDocument(simple_preprocess(sentence), [i]) for i, sentence in enumerate(sentences)]

    # Build vocabulary
    cores = cpu_count()
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=cores)
    model.build_vocab(tagged_data)

    # Train the Doc2Vec model
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    return model

def project_vectors(vecs, n_dimensions, scaled=True):
    """
    Uses randomized PCA to project the vectors to n dimensional space
    """
    pca = PCA(n_components=n_dimensions, svd_solver='randomized')
    embedded_vectors = pca.fit_transform(vecs)

    if scaled:
        # Rescale the embedded vectors for better visualization
        scaler = MinMaxScaler()
        scaled_vectors = scaler.fit_transform(embedded_vectors)
        return scaled_vectors
    else:
        return embedded_vectors

############################
# Topic Modeling Functions #
############################

def train_lda(corpus, dictionary, num_topics):
    """
    Trains a gensim LdaModel on a corpus
    """
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    return lda_model

def display_lda_topics(lda_model, n_words):
    """
    Prints the most informative words for each topic in a Gensim LDA model
    lda_model : Gensim LDAModel object
    n_words : int, number of words to print per topic
    """
    # Get the most informative words for each topic
    topics_words = lda_model.show_topics(num_topics=-1, num_words=n_words, formatted=False)

    # Print the most informative words for each topic
    for topic_id, words in topics_words:
        print(f"Topic {topic_id}:")
        print(", ".join([word for word, _ in words]))
        print()

def predict_lda_topics(lda_model, df, text_col):
    # Assuming you have a list of preprocessed texts called 'preprocessed_texts'
    texts = [x.split() for x in df[text_col].values]

    # Create a dictionary and corpus from preprocessed texts
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    corpus_topic_distribution = [lda_model.get_document_topics(doc) for doc in corpus]
    dominant_topic = [max(doc_topics, key=lambda x: x[1]) for doc_topics in corpus_topic_distribution]
    topics = [x[0] for x in dominant_topic]
    topic_score = [x[1] for x in dominant_topic]
    df['lda_topic'] = topics
    df['lda_score'] = topic_score
    return df


def calculate_coherence(lda_model, dictionary, texts):
    """
    Function that calculates the coherence of a topic model with num_topics
    on a dataset
    """
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score

######################
# Modeling Functions #
######################

def make_model(text_preprocessor=None, text_col=None,
               cat_preprocessor=None, cat_cols=None,
               numerical_preprocessor=None, numerical_cols=None,
               model=None, n_feats=50, k_best=None):
    try:
        transformers = []
        if text_preprocessor and text_col:
            transformers.append(('text', text_preprocessor, text_col))
        if cat_preprocessor and cat_cols:
            transformers.append(('categorical', cat_preprocessor, cat_cols))
        if numerical_preprocessor and numerical_cols:
            transformers.append(('numerical', numerical_preprocessor, numerical_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        if k_best is not None:
            preprocessor = Pipeline([
                ('preprocessor', preprocessor),
                ('k_best', SelectKBest(f_regression, k=k_best))
            ])
        
        if n_feats:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('svd', TruncatedSVD(n_components=n_feats)),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
        
        return pipeline
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None