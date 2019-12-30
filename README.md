# psychology-analysis
Experimenting with ML to predict psychology engagement

```
$ python analyze.py
usage: analyze.py [-h] -i INPUT -c CLASSNAME [-o OUTPUT] [-t TRAIN] [-p PATCH]

Parses csv file of psych data and runs it through a svm model

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input data
  -c CLASSNAME, --classname CLASSNAME
                        column name for the output class
  -o OUTPUT, --output OUTPUT
                        path to save the model to
  -t TRAIN, --train TRAIN
                        percentage of input file as training data
  -p PATCH, --patch PATCH
                        patch any empty csv cells with this value

```

To setup and run:

1) clone the repository
2) start the virtual environment within the cloned repo `virtualenv -p python2.7`
3) `pip install -r requirements.txt`
4) run the program:

`python analyze.py --input /path/to/dataset.csv --output /path/to/save/model.m --t 0.5 --classname Engaged --patch 0.0`

Example output:
```
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
WARNING: patching csv with 0.0
------------------------------
Data size: 601
Features: 32
------------------------------
Classification report:
             precision    recall  f1-score   support

          0       0.77      0.89      0.83       223
          1       0.44      0.26      0.33        78

avg / total       0.69      0.72      0.70       301

--------------------------------------------------------------------------------
Predicted    0   1  All
True
0          198  25  223
1           58  20   78
All        256  45  301
--------------------------------------------------------------------------------
```

Notes:

* The `--classname` argument is the csv column label that corresponds to the class specification (i.e. the answer/prediction)

* The `--patch` argument patches any empty cells in the input dataset with that specified value. Default is `0.0`.
