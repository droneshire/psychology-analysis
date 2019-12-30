"""
Parses csv file of psych data and runs it through a svm model
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import csv
import os
import pandas
import pickle
import tempfile

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


def analyze_data(args):
    data = DataTraining(args.input, args.train, args.patch, args.classname)
    data.train()
    data.predict()
    if args.output:
        data.save(args.output)

def non_zero_input(input):
    x = float(input)
    if x <= 0.0:
        raise argparse.ArgumentTypeError('Input must be > 0.0')
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True, help='path to input data')
    parser.add_argument('-c', '--classname', required=True, help='column name for the output class')
    parser.add_argument('-o', '--output', help='path to save the model to')
    parser.add_argument('-t', '--train', type=non_zero_input, default=0.5,
                        help='percentage of input file as training data')
    parser.add_argument('-p', '--patch', type=float, default=0.0,
                        help='patch empty csv cells with this value (doesn\'t modify input file)')
    analyze_data(parser.parse_args())
