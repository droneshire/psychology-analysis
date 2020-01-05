"""
Parses csv file of psych data and runs it through a svm model
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse

from datatraining import DataTraining


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
