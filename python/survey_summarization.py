#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:12:18 2024

@author: elliot
"""

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Load the dataset
# Replace 'your_dataset.csv' with your actual file
data = pd.read_csv('your_dataset.csv')

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords and non-alphabetic
    return tokens

# Summarize each column
def summarize_column(column_data):
    all_tokens = []
    for text in column_data.dropna():  # Drop missing values
        all_tokens.extend(preprocess_text(str(text)))
    word_freq = Counter(all_tokens)  # Word frequency
    most_common = word_freq.most_common(10)  # Top 10 words
    return most_common

# Generate summaries
summaries = {}
for column in data.columns:
    summaries[column] = summarize_column(data[column])

# Print summaries
for question, summary in summaries.items():
    print(f"Question: {question}")
    print(f"Top Words: {summary}")
    print("-" * 40)
    
    
    
###############
# incorporate bigrams or trigrams
from nltk.util import ngrams

def preprocess_text_ngrams(text, n=2):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    n_grams = list(ngrams(tokens, n))  # Generate n-grams
    return n_grams

def summarize_column_ngrams(column_data, n=2):
    all_ngrams = []
    for text in column_data.dropna():
        all_ngrams.extend(preprocess_text_ngrams(str(text), n=n))
    ngram_freq = Counter(all_ngrams)
    most_common = ngram_freq.most_common(10)
    return most_common

# Example usage for bigrams
for column in data.columns:
    print(f"Bigrams for {column}:")
    print(summarize_column_ngrams(data[column], n=2))
    

################
# weight terms using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_with_tfidf(column_data):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Unigrams and bigrams
    tfidf_matrix = vectorizer.fit_transform(column_data.dropna())
    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Sum scores across all rows
    tfidf_ranking = [(feature_array[i], tfidf_scores[i]) for i in range(len(feature_array))]
    tfidf_ranking = sorted(tfidf_ranking, key=lambda x: x[1], reverse=True)[:10]  # Top 10 terms
    return tfidf_ranking

# Example usage
for column in data.columns:
    print(f"TF-IDF Summary for {column}:")
    print(summarize_with_tfidf(data[column]))

###########
# Identify key sentences

from nltk.tokenize import sent_tokenize

def summarize_sentences(column_data, n=3):
    sentence_scores = Counter()
    for text in column_data.dropna():
        sentences = sent_tokenize(text)
        for sentence in sentences:
            tokens = preprocess_text(sentence)
            sentence_scores[sentence] += len(tokens)  # Score based on word count
    most_common_sentences = sentence_scores.most_common(n)
    return [sentence for sentence, _ in most_common_sentences]

# Example usage
for column in data.columns:
    print(f"Key Sentences for {column}:")
    print(summarize_sentences(data[column]))
    

############
# phrase co-occurance analysis

from itertools import combinations

def co_occurrence_matrix(column_data):
    co_occurrence = Counter()
    for text in column_data.dropna():
        tokens = preprocess_text(text)
        for pair in combinations(tokens, 2):  # Pairwise combinations
            co_occurrence[pair] += 1
    return co_occurrence.most_common(10)

# Example usage
for column in data.columns:
    print(f"Co-occurring Pairs for {column}:")
    print(co_occurrence_matrix(data[column]))
    

###########
# domain specific stop words
custom_stop_words = set(stopwords.words('english') + ['yes', 'no', 'maybe'])


###########
# group terms using GloVe embeddings
'''
# note: have to download glove embeddings here: https://nlp.stanford.edu/projects/glove/ and place them in working directory
	•	Use an appropriate dimension (e.g., 50d, 100d) based on your needs and memory constraints.
	2.	Cluster Size:
	•	Adjust n_clusters to control the granularity of grouping.
	3.	Output:
	•	Each cluster will contain semantically similar terms. Review clusters to interpret the themes.
'''

import numpy as np
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Load GloVe Embeddings
def load_glove_embeddings(file_path='glove.6B.50d.txt'):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Preprocess Text
def preprocess_text_glove(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

# Retrieve Word Embeddings
def get_word_embeddings(tokens, embeddings):
    word_vectors = []
    valid_words = []
    for token in tokens:
        if token in embeddings:
            word_vectors.append(embeddings[token])
            valid_words.append(token)
    return np.array(word_vectors), valid_words

# Cluster Words
def cluster_words(word_vectors, valid_words, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(word_vectors)
    clusters = {i: [] for i in range(n_clusters)}
    for word, label in zip(valid_words, labels):
        clusters[label].append(word)
    return clusters

# Main Function
def summarize_with_glove(data_column, glove_path='glove.6B.50d.txt', n_clusters=5):
    embeddings = load_glove_embeddings(glove_path)
    all_tokens = []
    for text in data_column.dropna():
        all_tokens.extend(preprocess_text_glove(str(text)))
    word_vectors, valid_words = get_word_embeddings(all_tokens, embeddings)
    if word_vectors.size > 0:
        clusters = cluster_words(word_vectors, valid_words, n_clusters)
        return clusters
    else:
        return "No valid words found with embeddings."

# Example Usage
data = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
for column in data.columns:
    print(f"Clusters for {column}:")
    clusters = summarize_with_glove(data[column])
    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")    
    
###########
# Advanced Summarization Using Sentence Embeddings
# note: pip install sentence-transformers

'''
Explanation:

	1.	SentenceTransformer Model:
	•	The all-MiniLM-L6-v2 model is fast and effective for sentence embeddings.
	•	It converts each response into a vector, capturing semantic meaning.
	2.	Clustering:
	•	K-Means groups semantically similar responses.
	•	Each cluster represents a potential theme or idea.
	3.	Summarization:
	•	A representative sentence from each cluster is extracted as a summary.
	•	Further refinements can include using cosine similarity to rank sentences by centrality.
'''

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate Sentence Embeddings
def get_sentence_embeddings(data_column):
    sentences = data_column.dropna().tolist()  # Drop missing values
    embeddings = model.encode(sentences)
    return sentences, embeddings

# Cluster Sentences
def cluster_sentences(sentences, embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for sentence, label in zip(sentences, labels):
        clusters[label].append(sentence)
    return clusters

# Summarize Clusters
def summarize_clusters(clusters):
    summaries = {}
    for cluster_id, sentences in clusters.items():
        # Pick a representative sentence (e.g., first sentence)
        summaries[cluster_id] = sentences[0] if sentences else ""
    return summaries

# Main Function
def advanced_summarization(data_column, n_clusters=5):
    sentences, embeddings = get_sentence_embeddings(data_column)
    clusters = cluster_sentences(sentences, embeddings, n_clusters)
    summaries = summarize_clusters(clusters)
    return summaries, clusters

# Example Usage
data = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
for column in data.columns:
    print(f"Summaries for {column}:")
    summaries, clusters = advanced_summarization(data[column])
    for cluster_id, summary in summaries.items():
        print(f"Cluster {cluster_id}: {summary}")
    print("-" * 40)
    
    
####
# variation 1

'''
Adjust Cluster Numbers:
	•	Tune n_clusters to fit the diversity of responses.
	•	Extract Central Sentences:
	•	Use cosine similarity to identify the most representative sentence per cluster.

	Use Hierarchical Clustering:
	•	For a more natural grouping, you can use AgglomerativeClustering instead of K-Means.
'''  
    
from sklearn.metrics.pairwise import cosine_similarity

def find_representative_sentence(cluster_sentences, embeddings):
    cluster_embeddings = model.encode(cluster_sentences)
    similarity_matrix = cosine_similarity(cluster_embeddings)
    central_sentence_idx = similarity_matrix.sum(axis=1).argmax()
    return cluster_sentences[central_sentence_idx]


#########
# add a visualization
# pip install matplotlib scikit-learn umap-learn

'''
Key Points:

	1.	reduce_dimensions Function:
	•	Use t-SNE for local neighborhood preservation.
	•	Use UMAP for better global structure while maintaining local detail.
	2.	plot_clusters Function:
	•	Scatter plot with unique colors for each cluster.
	3.	Usage:
	•	Specify the dimensionality reduction method ('tsne' or 'umap').
	•	Adjust n_clusters based on the dataset.

The plot will show clusters in 2D space, with each cluster assigned a unique color. The separation in the plot indicates how distinct the clusters are.

'''

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

# Function to perform dimensionality reduction
def reduce_dimensions(embeddings, method='tsne'):
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")
    return reduced_embeddings

# Plot Clusters
def plot_clusters(reduced_embeddings, labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            reduced_embeddings[indices, 0], 
            reduced_embeddings[indices, 1], 
            label=f"Cluster {label}", 
            alpha=0.7
        )
    plt.title("Clusters Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()

# Main Function
def visualize_clusters(data_column, method='tsne', n_clusters=5):
    sentences, embeddings = get_sentence_embeddings(data_column)
    clusters = cluster_sentences(sentences, embeddings, n_clusters)
    
    # Map sentences to their cluster labels
    labels = []
    for cluster_id, sentences_in_cluster in clusters.items():
        labels.extend([cluster_id] * len(sentences_in_cluster))
    
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings, method)
    
    # Plot
    plot_clusters(reduced_embeddings, labels)

# Example Usage
data = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
for column in data.columns:
    print(f"Visualizing clusters for {column}:")
    visualize_clusters(data[column], method='umap', n_clusters=5)