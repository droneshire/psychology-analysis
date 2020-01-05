import tempfile
import pickle

import pandas

from sklearn import model_selection, svm
from sklearn.metrics import classification_report, confusion_matrix


def fill_empty_cells(csv_file, fill=str(0.0)):
    """ helper function that replaces empty csv cells with a specified fill value"""
    no_ext, _ = os.path.split(csv_file)
    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.basename(no_ext)).name
    with open(csv_file) as infile, open(output_csv, 'w') as outfile:
        reader = csv.reader(infile, delimiter=',')
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            out_row = row
            for i in range(len(row)):
                if row[i] == ' ':
                    print('WARNING: patching csv with {}'.format(fill))
                out_row[i] = '0.0' if row[i] == ' ' else row[i]
            writer.writerow(out_row)
    return output_csv
    
class DataTraining(object):
    """ Class that parses a dataset and trains it using svm """

    def __init__(self, input_csv, training_percentage, empty_cell_patch, classification_name):
        self.input = pandas.read_csv(fill_empty_cells(input_csv, fill=empty_cell_patch))
        self.X = self.input.drop(classification_name, axis=1)
        self.y = self.input[classification_name]
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.X, self.y, test_size=training_percentage)
        self.classifier = None
        self.prediction = None
        print('{0}\nData size: {1}\nFeatures: {2}\n{0}'.format(
            '-' * 30, self.X.shape[0], self.X.shape[1]))

    def train(self):
        self.classifier = svm.SVC(kernel='poly', degree=3)
        self.classifier.fit(self.X_train, self.y_train)

    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self.classifier, output)

    def predict(self):
        self.prediction = self.classifier.predict(self.X_test)
        # print('Confusion matrix:\n{}'.format(confusion_matrix(self.y_test, self.prediction)))
        print('Classification report:')
        print(classification_report(self.y_test, self.prediction))
        results = pandas.crosstab(
            self.y_test,
            self.prediction,
            rownames=['True'],
            colnames=['Predicted'],
            margins=True)
        print('{0}\n{1}\n{0}'.format('-' * 80, results))
