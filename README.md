# Randomized Optimization with mlrose

This repo is for studying randomized optimization, the program is based on [mlrose](https://mlrose.readthedocs.io/en/stable/index.html). 
I also refer author's online [tutorial](https://towardsdatascience.com/getting-started-with-randomized-optimization-in-python-f7df46babff0).

## Environment setup
Make sure you have Python on your machine.
After you pull this repo, following commands helps you installing the required packages.

```
// virtual environment is recommended, but optional
$ virtualenv venv

(venv) $ pip install -r requirements.txt
```

##Run randomized weights optimization

### Paramters for algorithms
- `max_iter`: integer, the maximum number of training iteration, default `1000`
- `max_attempt`: integer, the maximum number of attempts, the training may stop eariler while reaching a good enough state, default `100`
- `max_clip`: integer, the range of weights would be set as [-MAX_CLIP, MAX_CLIP], default `10`
- `lrate`: float, learning rate, default `0.1`
- `sched`: `0`, `1`, or `2`, choose the decay schedule only used for **simulated annealing**'s temperature parameter, 0=geometrically decay 1=arithmetically decay 2=exponentially decay, default `0`
- `pop_size`: integer, size of population, used only for **genetic algorithm**, default `1000`
- `mut_prob`: float, limited to [0, 1], the probability of mutation, used only for **genetic algorithm**

##Run 8-Queen problem with randomized optimization
