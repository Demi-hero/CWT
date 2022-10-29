import re
import string
import contractions
import nltk
import matplotlib.pyplot as plt

# Make sure you have downloaded the StanfordNLP English model and other essential tools using,
# stanfordnlp.download('en')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Intersection between the words used in the positve and negative sentiment reviews. In an attempt to help with
# differenciating between the classes
extra_stops = ['order', 'money', 'ordered', 'know', 'nothing', 'taste', 'came', 'quality', 'come', 'waiting', 'one',
               'charge', 'enjoyed', 'dessert', 'try', 'starter', 'served', 'rather', 'think', 'little', 'dinner',
               'poor', 'menu', 'arrived', 'definitely', 'like', 'time', 'portion', 'food', 'atmosphere', 'cocktail',
               'area', 'meal', 'back', 'well', 'go', 'going', 'visited', 'really', 'drink', 'night', 'disappointed',
               'expensive', 'manager', 'around', 'take', 'great', 'though', 'chip', 'felt', 'best', 'thing', 'staff',
               'hour', 'pub', 'get', 'overall', 'better', 'london', 'make', 'value', 'fish', 'got', 'said', 'price',
               'lot', 'even', 'also', 'special', 'however', 'course', 'review', 'waitress', 'worth', 'friendly',
               'selection', 'bill', 'still', 'delicious', 'another', 'good', 'disappointing', 'quite', 'u', 'told',
               'place', 'main', 'although', 'busy', 'made', 'wanted', 'visit', 'experience', 'would', 'cold', 'away',
               'took', 'always', 'never', 'steak', 'want', 'ever', 'left', 'choice', 'recommend', 'excellent', 'day',
               'wait', 'bit', 'first', 'eat', 'wine', 'minute', 'two', 'say', 'chicken', 'evening', 'average', 'small',
               'ok', 'table', 'rude', 'much', 'pay', 'lovely', 'way', 'cooked', 'lunch', 'amazing', 'dish', 'side',
               'bad', 'last', 'bar', 'plate', 'ask', 'many', 'full', 'nice', 'could', 'people', 'booked', 'waiter',
               'asked', 'burger', 'service', 'restaurant', 'customer', 'tasty', 'attentive', 'went', 'friend', 'u']

def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def lematize_text(text, lemanator, tokenizer):
    return [lemanator.lemmatize(w) for w in tokenizer.tokenize(text)]


def clean_text(dataframe, target_column, stop_words):
    lemanator = nltk.stem.WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    dataframe['clean_reviews'] = [x.lower() for x in dataframe[target_column]]
    dataframe['clean_reviews'] = [contractions.fix(x) for x in dataframe['clean_reviews']]
    dataframe['clean_reviews'] = [re.sub('\w*\d\w*', '', x) for x in dataframe['clean_reviews']]
    dataframe['clean_reviews'] = [re.sub('[^A-Za-z0-9 ]+', '', x) for x in dataframe['clean_reviews']]
    dataframe['clean_reviews'] = [remove_emojis(x) for x in dataframe['clean_reviews']]
    dataframe['clean_reviews'] = [re.sub(r'(.)\1{3,}', '\\1', x) for x in dataframe['clean_reviews']]
    dataframe['clean_reviews'] = [re.sub(f'{re.escape(string.punctuation)}', '', x) for x in dataframe['clean_reviews']]
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    dataframe['clean_reviews'] = dataframe['clean_reviews'].str.replace(pat, '', regex=True)
    dataframe['clean_reviews'] = [re.sub(' +', ' ', x) for x in dataframe['clean_reviews']]
    dataframe['lem_reviews'] = [lematize_text(x, lemanator, w_tokenizer) for x in dataframe['clean_reviews']]
    return dataframe


def stop_word_builder(data: object, stop_words: object) -> object:
    most_common_words = []
    for review_score in range(1, 6):
        print(review_score)
        test_data = data.loc[data['rating_review'] == review_score, :]
        # Make a plot and save the most common
        flat_list = [item for sublist in test_data['lem_reviews'].to_list() for item in sublist]
        nlp_words = nltk.FreqDist(flat_list)
        most_common_words += [nlp_words.most_common(100)]
        plt.figure(figsize=(16, 8), dpi=100)
        plt.xticks(rotation=45)
        plt.bar(*zip(*nlp_words.most_common(50)))
        plt.savefig(f"rating_{review_score}_top_words.png")
        plt.clf()

    intersection = []
    for ind in range(len(most_common_words)):
        source = set([x[0] for x in most_common_words[ind]])
        for ind2 in range(ind, (len(most_common_words) - 1)):
            set2 = set([x[0] for x in most_common_words[ind2]])
            intersection += source.intersection(set2)
    intersection = list(set(intersection))
    return intersection
