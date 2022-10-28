import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

def conf_matrix(actual, predicted):
    labels = sorted(actual.unique().tolist())

    cm = confusion_matrix(actual, predicted, labels=labels)
    heatmap = sns.heatmap(cm, xticklabels=labels,
                yticklabels=labels, annot=True,
                fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu")
    heatmap.set(xlabel='True Values', ylabel='Predicted Values')
    #true_neg, false_pos = cm[0]
    #false_neg, true_pos = cm[1]

    #accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    #precision = round((true_pos) / (true_pos + false_pos),3)
    #recall = round((true_pos) / (true_pos + false_neg),3)
    #f1 = round(2 * (precision * recall) / (precision + recall),3)

    #cm_results = [accuracy, precision, recall, f1]
    #print(cm_results)
    #cm_results,
    return heatmap


def model_produce(train_data, train_lables, model, min_df, ngram_range, stop_words):
    m = model(stop_words=stop_words, min_df=min_df, max_df=.85, token_pattern="\\b[a-z][a-z]+\\b", ngram_range=ngram_range)
    m2 = m.fit_transform(train_data)
    X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(m2, train_lables)
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.3, random_state=42,
                                                        stratify=y_under)
    return X_train, X_test, y_train, y_test


def calculate_metrics(model, X_train, X_test, y_train, y_test, cv):
    y_pred = model.predict(X_test)
    test_accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    return dict([
            ("Model" , model.__class__.__name__),
            ("Train Accuracy", model.score(X_train,y_train)),
            ("Test Accuracy", test_accuracy),
            ("Precision" , precision ),
            ("recall",recall),
            ("f1",f1)]), y_pred


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['blue','pink','pink']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='blue', label='Negative')
            green_patch = mpatches.Patch(color='pink', label='Positive')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

