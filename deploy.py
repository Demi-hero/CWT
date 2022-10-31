import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a model version to the api')
    parser.add_argument('-v', '--version',
                        help='the version number of the model to deploy')

    args = parser.parse_args()
    version = args.version
    dir = os.path.join("models", f"version_{args.version}")
    model_file = os.path.join(dir, "model.pickle")
    model_acc = os.path.join(dir, "model_accuracy.csv")
    if not os.path.isdir(dir):
        # stand in for actual logging
        print("There is no model registered to this version")
    if not os.path.isfile(model_file):
        print("There is no model in the version. Check calibration status")
    if not os.path.isfile(model_acc):
        print("There is not accuracy data. Test the model accuracy before deployment")

    new_flask_model = os.path.join("Flask", 'model', "model_new.pickle")
    flask_model = os.path.join("Flask", 'model', "model_new.pickle")
    new_flask_acc = os.path.join("Flask", 'model', "new_model_accuracy.csv")
    flask_acc = os.path.join("Flask", 'model', "model_accuracy.csv")

    # copy model accros
    if len(os.listdir("Flask\\model")) == 0:
        shutil.copyfile(model_file, flask_model)
        shutil.copyfile(model_acc, flask_acc)
    else:
        shutil.copyfile(model_file, new_flask_model)
        shutil.copyfile(model_acc, new_flask_acc)
        os.replace(new_flask_model, flask_model)
        os.replace(flask_acc, new_flask_acc)

