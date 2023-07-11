import functools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import entropy, boxcox, probplot
from src.process import calculate_coherence, train_lda, extract_keywords_tfidf
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from tqdm import tqdm
from sklearn.cluster import KMeans
from wordcloud import WordCloud

def calculate_entropy_curve(dataframe, category_col, ns):
    """
    Simplifies the catefory column by setting 
    frequency threshold and calculates the entropy
    of that threshold
    Plots those entropies to find the appropriate n
    """
    counts = dataframe[category_col].value_counts()
    original_entropy = entropy(counts)

    entropies = []

    for n in ns:
        top_n_categories = counts[:n].index.tolist()


        # Apply grouping and calculate entropy of the simplified data
        dataframe['grouped_' + 'product_type_id'] = dataframe['product_type_id'].apply(lambda x: x if x in top_n_categories else 'Other')
        simplified_counts = dataframe['grouped_' + 'product_type_id'].value_counts()
        simplified_entropy = entropy(simplified_counts)
        
        entropies.append(simplified_entropy)

    plt.figure(figsize=(10, 6))
    plt.plot(ns, entropies, marker='o')
    plt.axhline(y=original_entropy, color='r', linestyle='--', label='Original Entropy')
    plt.xlabel('n')
    plt.ylabel('Entropy')
    plt.title('Entropy vs. n')
    plt.legend()
    plt.show()

def calculate_topic_coherence(texts, start_topics, end_topics, n=2):
    # Create a dictionary and corpus from preprocessed texts
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]


    # Calculate coherence scores for different numbers of topics
    coherence_scores = []
    for num_topics in tqdm(range(start_topics, end_topics+1, n)):
        lda = train_lda(corpus, dictionary, num_topics)
        coherence_score = calculate_coherence(lda, dictionary, texts)
        coherence_scores.append(coherence_score)
        
    # Plot coherence scores
    plt.plot(range(start_topics, end_topics+1, n), coherence_scores)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Topic Coherence")

    # Find the highest coherence score
    max_index = coherence_scores.index(max(coherence_scores))
    max_score = coherence_scores[max_index]

    # Plot the max coherence point
    plt.plot(max_index * n + start_topics, max_score, 'ro', label='Max Point')
    plt.legend()

    # Show the plot
    plt.show()

    # Use the number of topics with the highest coherence score for topic modeling
    num_topics = max_index * n + start_topics
    return num_topics

def plot_numerical_distribution(df, num_cols):
    """
    Plot the numerical distribution of multiple columns separately.
    """
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        data = df[col]
        unique_values = data.unique()
        
        if len(unique_values) > 10:
            sns.histplot(data, kde=True)
        else:
            sns.countplot(data)
        
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

def plot_categorical_distribution(df, cat_col):
    """
    Plot the basic count distribution of a categorical column
    """
    plt.figure(figsize=(10, 6))
    topic_counts = df[cat_col].value_counts().sort_values(ascending=False)
    sns.barplot(x=topic_counts.index, y=topic_counts.values, order=topic_counts.index, palette='viridis')
    plt.title(f"{cat_col} Distribution")
    plt.xlabel(cat_col)
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability if needed
    plt.show()

def plot_grouped_categorical_distribution(df, group_by_col, cat_col, colors=None):
    """
    Plot the count distribution of a categorical column as grouped by
    a grouping column
    """
    categorical_distribution = df.groupby(group_by_col)[cat_col].value_counts(normalize=True).unstack()

    # Plot stacked bar plots for categorical distributions
    plt.figure(figsize=(10, 6))
    if colors is not None:
        categorical_distribution.plot(kind='bar', stacked=True, color=[colors.get(col, 'gray') for col in categorical_distribution.columns])
    else:
        categorical_distribution.plot(kind='bar', stacked=True)
    
    # Update plot aesthetics
    plt.title(f"{cat_col} by {group_by_col}")
    plt.xlabel(f"{group_by_col}")
    plt.ylabel('Proportion')
    
    # Adjust the legend placement and enable wrapping
    plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, columns):
    """
    Plots a heatmap of the correlation matrix of numerical columns
    """
    corr_matrix = df[columns].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask)
    plt.title("Correlation Matrix")
    plt.show()

def plot_elbow_curve(ks, X):
    """
    Calculates the elbow curve for choosing the appropriate
    k for K Means clustering
    ks : list of options for k
    X : numpy array or matrix of vectors to cluster
    """
    distorsions = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(ks, distorsions)
    plt.grid(True)
    plt.title('Elbow curve')

def plot_word_vectors_with_labels(word_vectors_2d, vec_labels, title='2D Word Vectors with Cluster Colors', label="Cluster"):
    """
    Plots projected 2d word vectors with clusters
    word_vectors_2d : array of 2d vectors
    cluster_names : array of the same length as word_vectors_2d of labels for each vector
    title : plot title
    """
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=vec_labels)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(label=label)
    plt.show()

def plot_word_vectors_with_clusters_and_centroids(word_vectors_2d, vec_labels, cluster_keywords, title='2D Word Vectors with Cluster Colors'):
    """
    Plots 2d projected word vectors with labels and informative words for each label
    Adds centroids to the plot as well
    """
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=vec_labels, alpha=0.8, edgecolor='k')
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Find the unique cluster labels
    unique_clusters = np.unique(vec_labels)

    # Plot centroids and label with cluster keywords
    for cluster in unique_clusters:
        cluster_points = word_vectors_2d[vec_labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        centroid_keywords = cluster_keywords[cluster]
        plt.scatter(centroid[0], centroid[1], color='red', marker='x', s=100)
        plt.text(
            centroid[0], centroid[1], ', '.join(centroid_keywords),
            color='black', fontsize=10, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')
        )

    plt.colorbar(label='Cluster', ticks=unique_clusters)
    plt.tight_layout()
    plt.show()

def plot_word_clouds(informative_words_per_cluster):
    """
    Plots 
    """
    # Configure the number of rows and columns for subplots
    num_clusters = len(informative_words_per_cluster)
    num_rows = (num_clusters + 1) // 2
    num_cols = 2 if num_clusters > 1 else 1

    # Create subplots for each cluster
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    # Generate and plot word clouds for each cluster
    for i, (cluster, words) in enumerate(informative_words_per_cluster.items()):
        # Generate word cloud for the cluster's informative words
        wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(words))

        # Plot the word cloud
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f"Cluster {cluster}")
        axes[i].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, vectorizer=None, num_cols=None):
    """
    Plots the feature importances of a sklearn model
    model : Pretrained Sklearn model
    vectorizer : Pretrained Sklearn vectorizer that has get_feature_names_out() method, optional
    num_cols : list of numerical columns, optional
    """
    if vectorizer is None and num_cols is None:
        raise ValueError("Either 'vectorizer' or 'num_cols' must be provided.")
    
    feature_names = []
    if vectorizer:
        feature_names += list(vectorizer.get_feature_names_out())
    if num_cols:
        feature_names += num_cols
    
    feature_importances = pd.DataFrame({'Feature': feature_names,
                                        'Importance': model.feature_importances_})

    # Sort the feature importances in descending order
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances[:10])
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_qq(df, col, outliers=False):
    if outliers:
        transformed_data, _ = boxcox(df[col])
    # Create a Q-Q plot
    plt.figure(figsize=(8, 6))
    probplot(transformed_data, dist='norm', plot=plt)
    plt.title('Q-Q Plot of Data with Outliers')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()

def plot_numerical_features_by_categorical(df, num_cols, cat_col, mean=False, median=False):
    sns.set(style="whitegrid")
    if mean:
        mean_values = df.groupby(cat_col)[num_cols].mean()
    if median:
        median_values = df.groupby(cat_col)[num_cols].median()

    for numerical_variable in num_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=numerical_variable, data=df)
        plt.title(f"{numerical_variable} by {cat_col}")
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.show()

    for numerical_variable in num_cols:
        if mean:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=mean_values.index, y=mean_values[numerical_variable])
            plt.title(f"Mean {numerical_variable} by {cat_col}")
            plt.xlabel(cat_col)
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(f"Mean {numerical_variable}")
            plt.show()
        if median:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=median_values.index, y=median_values[numerical_variable])
            plt.title(f"Median {numerical_variable} by {cat_col}")
            plt.xlabel(cat_col)
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(f"Median {numerical_variable}")
            plt.show()