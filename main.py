# Process abstracts and generate word cloud

# %%
# importing all necessary modules
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import re

# %%
df = pd.read_excel(r"./data/AED50 Accepted Abstracts.xlsx", sheet_name=0)

# update set of common words
comment_words = ''
stopwords = set(STOPWORDS.union(
    {'numplaceholder', 'show', 'based', 'propose', 'proposes', 'proposed', 'used', 'one', 'cost', 'improve'}))

# identify any number as stopword
df['Abstract'] = df['Abstract'].apply(lambda x: re.sub(
    '\b[0-9][0-9.,-]*\b', 'numplaceholder', x))

# %%
# method 1: count word frequencies
# example: https://amueller.github.io/word_cloud/auto_examples/frequency.html?highlight=generate_from_frequencies

all_tfidf_df_nofilter = None
all_tfidf_df = None
all_tfidf_list = []
excluded_phrases_ngrams = {2: ['traffic flow', 'real time', 'real world', 'traffic data', 'trajectory data',
                               'learning models', 'short term', 'spatial temporal', 'ground truth', 'state art', 'traffic state', 'experimental results', 'learning methods', 'data driven', 'large scale', 'covid 19', 'prediction accuracy', 'learning approach', 'results showed', 'learning model', 'case study', 'neural networks', 'term memory', 'long short', 'convolutional neural', 'study aims', 'data collected', 'research gap', 'results indicate', 'results demonstrate', 'paper presents', 'learning framework', 'mean absolute'],
                           3: ['machine learning models', 'long short term', 'short term memory', 'deep learning model',
                               'experimental results show', 'real time traffic',
                               'surrogate safety measures', 'term memory lstm',
                               'machine learning ml', 'real world dataset',
                               'neural network ann', 'shapley additive explanations',
                               'additive explanations shap'],
                           4: ['artificial neural network ann', 'short term memory lstm',
                               'shapley additive explanations shap']}

for ngram in list(range(2, 5)):
    excluded_phrases = excluded_phrases_ngrams[ngram]
    print(
        f'processing ngram number: {ngram} \nphrases excluded:\n{excluded_phrases}')

    # build tf-idf
    vectorizer = TfidfVectorizer(
        # max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
        # max_df = 25 means "ignore terms that appear in more than 25 documents".
        max_df=0.5,
        # min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
        # min_df = 5 means "ignore terms that appear in less than 5 documents".
        min_df=5,
        ngram_range=(ngram, ngram),
        stop_words=stopwords
    )
    t0 = time()
    X_tfidf = vectorizer.fit_transform(df.Abstract)

    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")

    # count words
    tfidf_df = pd.DataFrame(X_tfidf.todense())
    tfidf_df.columns = list(vectorizer.get_feature_names_out())
    tfidf_df['paper_id'] = tfidf_df.index
    tfidf_df = tfidf_df.melt(
        id_vars='paper_id', var_name='word', value_name='count')
    tfidf_df = tfidf_df.groupby('word').sum()['count'].reset_index().sort_values(
        'count', ascending=False).reset_index(drop=True)

    # use only the top 25 and then remove excluded phrases
    tfidf_df_nofilter = tfidf_df.copy()
    tfidf_df = tfidf_df.head(50)
    tfidf_df = tfidf_df[~tfidf_df['word'].isin(excluded_phrases)]

    # build one big dataframe
    all_tfidf_list.append(tfidf_df)
    if not isinstance(all_tfidf_df, pd.DataFrame):
        # print('assigning')
        all_tfidf_df = tfidf_df.copy()
        all_tfidf_df_nofilter = tfidf_df_nofilter.copy()
    else:
        # print('concat')
        all_tfidf_df = pd.concat([tfidf_df, all_tfidf_df])
        all_tfidf_df_nofilter = pd.concat(
            [tfidf_df_nofilter, all_tfidf_df_nofilter])

# all_tfidf_df
processed_word_freq = all_tfidf_df.set_index('word').T.to_dict('list')
processed_word_freq = {
    i: processed_word_freq[i][0] for i in processed_word_freq}

# %%

wc = WordCloud(width=1200, height=600, margin=0,
               background_color="black", max_words=1000, random_state=3)
# generate word cloud
wc.generate_from_frequencies(processed_word_freq)

# show
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# save
save_name = "trbam2023aed50_papers_2023-01-07"
wc.to_file(f"{save_name}.png")

# for review
with pd.ExcelWriter(f"{save_name}.xlsx") as writer:
    all_tfidf_df.to_excel(writer, sheet_name="filtered_words")
    all_tfidf_df_nofilter.to_excel(writer, sheet_name="all_words")


# %%

# # method 2: all words and bigrams

# # iterate through the csv file
# for val in df.Abstract:

#     # typecaste each val to string
#     val = str(val)

#     # split the value
#     tokens = val.split()

#     # Converts each token into lowercase
#     for i in range(len(tokens)):
#         tokens[i] = tokens[i].lower()

#     comment_words += " ".join(tokens)+" "


# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white',
#                       stopwords=stopwords,
#                       min_font_size=10)
# wordcloud = wordcloud.generate(comment_words)

# # plot the WordCloud image
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)

# plt.show()
# # %%
