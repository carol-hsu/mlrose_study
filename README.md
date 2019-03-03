# [Randomized Optimization with mlrose](https://github.com/carol-hsu/mlrose_study)

This repo is for studying randomized optimization, the program is based on [mlrose](https://mlrose.readthedocs.io/en/stable/index.html). 
I also refer author's online [tutorial](https://towardsdatascience.com/getting-started-with-randomized-optimization-in-python-f7df46babff0).

## Environment setup
Make sure you have Python3 on your machine.
After you pull this repo, following commands helps you installing the required packages.

```
// virtual environment is recommended, but optional
$ virtualenv venv

(venv) $ pip install -r requirements.txt -p python3
```

## Run randomized weights optimization

To run this problem, python file `face_recongnition.py` is the entry point. 

Check help messages `-h` for the details.
```
$ python face_recongnition.py -h
usage: face_recongnition.py [-h] -i TRAIN -t TEST [-r OPTIMIZATION_ALGO]
                            [-o OPTIONS]

optional arguments:
  -h, --help            show this help message and exit
  -i TRAIN, --train TRAIN
                        directory of training set/encodings of training set
  -t TEST, --test TEST  directory of testing set/encodings of testing set
  -r OPTIMIZATION_ALGO, --optimization-algo OPTIMIZATION_ALGO
                        Use which randomized optimization algorithms: 1)random
                        hill climbing 2)simulated annealing 3)genetic
                        algorithm, default=1
  -o OPTIONS, --options OPTIONS
                        setting options based on the algorithms, configure in
                        KEY=VALUE pair, and separate by comma. E.g.
                        max_iter=1000,lrate=0.1 check README for complete
                        configurable options
```
This program is fixed to use customed pickle files. Please apply the file in `./pickles`.
Other input parameters should be set along with the algorithm, check next subsection for the detail of parameters.

Example command line:
```
$ python face_recongnition.py -i pickles/pk_large.pickle -t pickles/pk_tiny.pickle -r 3 -o max_iter=1000,lrate=10,pop_size=200,mut_prob=0.4
```

### Paramters for algorithms
- `max_iter`: integer, the maximum number of training iteration, default `1500`
- `max_clip`: integer, the range of weights would be set as [-MAX_CLIP, MAX_CLIP], default `10`
- `lrate`: float, learning rate, default `10`
- `sched`: `0`, `1`, or `2`, choose the decay schedule only used for **simulated annealing**'s temperature parameter, 0=geometrically decay 1=arithmetically decay 2=exponentially decay, default `2`
- `pop_size`: integer, size of population, used only for **genetic algorithm**, default `100`
- `mut_prob`: float, limited to [0, 1], the probability of mutation, used only for **genetic algorithm**, default `0.4`

## Run 8-Queen problem with randomized optimization

To run this problem, python file `eight_queen.py` is the entry point. 

Check help messages `-h` for the details.

```
$ python eight_queen.py -h
usage: eight_queen.py [-h] [-n NUMBER] [-i ITERATION] [-p PROBLEM] [-a ALGO]

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
                        The number of queens, default: 8
  -i ITERATION, --iteration ITERATION
                        The number of iteration for optimizing, default: 1000
  -p PROBLEM, --problem PROBLEM
                        Use which of the optimzation problems: 0) discrete 1)
                        continuous 2) TSP, default: 0
  -a ALGO, --algo ALGO  Use which of the algorithms: 0) randomized hill
                        climbing 1) simulated annealing 2) genetic algorithm
                        3) MIMIC, default: 0
```

You may change the number of iterations and the number of queens as you wish.
```
// iteration=100 with 20 queens and using genetic algorithm
$ python eight_queen.py -i 100 -n 20 -a 2
The best state found is:  [6, 1, 16, 19, 0, 10, 18, 14, 17, 2, 11, 5, 14, 4, 12, 16, 4, 17, 14, 8]
The fitness at the best state is:  0.9421052631578948
```
