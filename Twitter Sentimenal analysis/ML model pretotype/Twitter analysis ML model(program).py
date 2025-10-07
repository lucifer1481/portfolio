import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# Function to preprocess data and generate sentiment analysis
def preprocess_and_analyze():
    try:
        # Read data
        file_path = file_path_entry.get()
        df = pd.read_csv(file_path)

        # Data preprocessing
        df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
        df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
        df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
        tokenised_tweet = df['clean_tweet'].apply(lambda x: x.split())
        tokenised_tweet = tokenised_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
        for i in range(len(tokenised_tweet)):
            tokenised_tweet[i] = " ".join(tokenised_tweet[i])
        df['clean_tweet'] = tokenised_tweet

        # Plot selected type of graph
        selected_plot = plot_type_var.get()
        if selected_plot == "Word Cloud":
            plot_word_cloud(df)
        elif selected_plot == "Bar Plot":
            plot_bar(df)
        elif selected_plot == "Pie Chart":
            plot_pie(df)
        elif selected_plot == "Scatter Plot":
            plot_scatter(df)
        elif selected_plot == "Most Used Word":
            plot_most_used_word(df)

        # Display statistics
        pos_count = df[df['label'] == 0].shape[0]
        neg_count = df[df['label'] == 1].shape[0]
        stat_text = f"Total Tweets: {pos_count+neg_count}\nPositive Tweets: {pos_count}\nNegative Tweets: {neg_count}"
        statistics_label.config(text=stat_text)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to plot bar plot
def plot_bar(df):
    plt.figure(figsize=(8, 6))
    df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Bar Plot of Positive and Negative Tweets')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks([0, 1], ['Positive', 'Negative'], rotation=0)
    plt.show()

# Function to plot pie chart
def plot_pie(df):
    plt.figure(figsize=(8, 6))
    df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Pie Chart of Positive and Negative Tweets')
    plt.ylabel('')
    plt.legend()
    plt.show()

# Function to plot scatter plot
def plot_scatter(df):
    # Generate random data for scatter plot
    np.random.seed(0)
    x = np.random.rand(100)
    y = np.random.rand(100)
    colors = np.where(df['label'] == 0, 'green', 'red')

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.title('Scatter Plot of Positive and Negative Tweets')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Function to plot word cloud
def plot_word_cloud(df):
    # Wordcloud for positive tweets
    positive_tweets = df[df['label'] == 0]['clean_tweet']
    positive_words = " ".join([tweet for tweet in positive_tweets])
    positive_wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(positive_words)
    
    # Wordcloud for negative tweets
    negative_tweets = df[df['label'] == 1]['clean_tweet']
    negative_words = " ".join([tweet for tweet in negative_tweets])
    negative_wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(negative_words)

    # Plot word clouds
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Positive Tweets')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Negative Tweets')
    plt.axis('off')

    plt.show()

# Function to plot the most used word
def plot_most_used_word(df):
    all_words = " ".join([tweet for tweet in df['clean_tweet']])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Used Tweet Word')
    plt.show()

# Function to remove Twitter handles
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, " ", input_txt)
    return input_txt

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
stemmer = nltk.stem.PorterStemmer()

# Set up GUI
root = tk.Tk()
root.title("Twitter Sentiment Analysis")
root.geometry("800x600")

# Header Label
header_label = tk.Label(root, text="Twitter Sentiment Analysis", font=("Helvetica", 20))
header_label.pack(pady=10)

# File Path Entry
file_path_label = tk.Label(root, text="Enter File Path:")
file_path_label.pack()
file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack()

# Plot Type Label
plot_type_label = tk.Label(root, text="Select Plot Type:")
plot_type_label.pack()

# Plot Type Variable
plot_type_var = tk.StringVar(root)
plot_type_var.set("Bar Plot")

# Plot Type Dropdown Menu
plot_type_menu = tk.OptionMenu(root, plot_type_var, "Word Cloud", "Bar Plot", "Pie Chart", "Scatter Plot", "Most Used Word")
plot_type_menu.pack()

# Analyze Button
analyze_button = tk.Button(root, text="Analyze", command=preprocess_and_analyze, bg="#4CAF50", fg="white",
                           font=("Helvetica", 14))
analyze_button.pack(pady=10)

# Instructions Label
instructions_label = tk.Label(root, text="Analysis Results will be displayed below:", font=("Helvetica", 12))
instructions_label.pack(pady=10)

# Statistics Label
statistics_label = tk.Label(root, text="", font=("Helvetica", 12))
statistics_label.pack()

# Separator
separator = ttk.Separator(root, orient='horizontal')
separator.pack(fill='x', padx=20, pady=10)

# Footer Label
footer_label = tk.Label(root, text="Developed by Team 1 (AI&DS)", font=("Helvetica", 10))
footer_label.pack(side='bottom', pady=10)

# Run the GUI 
root.mainloop()


