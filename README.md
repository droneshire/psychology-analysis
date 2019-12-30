# psychology-analysis
Experimenting with ML to predict psychology engagement

```
$ python analyze.py
usage: analyze.py [-h] -i INPUT -c CLASSNAME [-o OUTPUT] [-t TRAIN]
```

To setup and run:

1) clone the repository
2) start the virtual environment within the cloned repo `virtualenv -p python2.7`
3) `pip install -r requirements.txt`
4) run the program:

`python analyze.py --input /path/to/dataset.csv --output /path/to/save/model.m --t 0.5 --classname Engaged`
