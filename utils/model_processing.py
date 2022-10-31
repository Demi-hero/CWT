import os
import pickle
import pandas as pd


def generate_new_model_directory():
    if not os.path.isdir('models'):
        os.mkdir('models')
    file_count = len(os.listdir('models'))
    path = f'models\\version_{file_count+1}'
    os.mkdir(path)
    return path


def save_model_data(model, report, outpath):
    model_file = os.path.join(outpath, "model.pickle")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    report_path = os.path.join(outpath, "model_accuracy.csv")
    acc_measures = pd.DataFrame(report['rf'])
    acc_measures['version'] = outpath[-1]
    acc_measures.to_csv(report_path)
    return acc_measures


def save_training_data(train_vars, train_labels, test_vars, test_labels, outpath):
    train_vars['label'] = train_labels
    test_vars['label'] = test_labels
    train_vars.to_csv(os.path.join(outpath, 'training_data.csv'))
    test_vars.to_csv(os.path.join(outpath, 'test_data.csv'))
    return 1
