import re
import string
import contractions
import nltk

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
