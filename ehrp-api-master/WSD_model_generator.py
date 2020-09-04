from WordSenseDisambiguation import WSD

def get_data():
    return test, train

def evaluate(model, data):
    # generate bunha info and scores
    return

def feature_engineer(data):
    return modified_data

def main():
    output_folder = '../WSD_Data'
    model_name = ''

    output_path = os.path.join(output_folder, model_name)

    train_data, test_data = get_data()
    train_data = feature_engineer(train_data)
    test_data = feature_engineer(test_data)

    wsd = WSD()
    # CREATE MODEL IN SOME FASHION
    wsd.model = SOMETHING

    evaluate(model, train_data, output_path)
    evaluate(model, test_data, output_path)

    wsd.save_model(output_path)
