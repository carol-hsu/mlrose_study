## eight queen problem by mlrose
import mlrose
import numpy as np
import argparse

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            count_max+=1
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (round(state[j]) != round(state[i])) \
                and (round(state[j]) != round(state[i]) + (j - i)) \
                and (round(state[j]) != round(state[i]) - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1
    
    return fitness


if __name__ == "__main__":

    # handle input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number", type=int, default=8,
            help="The number of queens, default: 8")
    ap.add_argument("-i", "--iteration", type=int, default=1000,
            help="The number of iteration for optimizing, default: 1000")
    ap.add_argument("-p", "--problem", type=int, default=0,
            help="Use which of the optimzation problems: 0) discrete 1) continuous 2) TSP, default: 0")
    ap.add_argument("-a", "--algo", type=int, default=0,
            help="Use which of the algorithms: 0) randomized hill climbing 1) simulated annealing \
                    2) genetic algorithm 3) MIMIC, default: 0")
    params = vars(ap.parse_args())
    
    num_queen = params["number"]
    max_iter = params["iteration"]
    max_atp = max_iter/10
    #fitness funtion
    queens_fitness = mlrose.CustomFitness(queens_max)

    #optimization problems: DiscreteOpt / ContinuousOpt / TSPOpt
    # discrete
    problem = mlrose.DiscreteOpt(length=num_queen, fitness_fn=queens_fitness, maximize=True, max_val=num_queen)
    if params["problem"] == 1: # continuous
    # ContinuousOpt(length, fitness_fn, maximize=True, min_val=0, max_val=1, step=0.1)
        problem = mlrose.ContinuousOpt(length=num_queen, fitness_fn=queens_fitness, maximize=True, max_val=num_queen)
    elif params["problem"] == 2: # TSP
    #TSPOpt(length, fitness_fn=None, maximize=False, coords=None, distances=None)
        queens_fitness = mlrose.CustomFitness(queens_max, problem_type="tsp")
        problem = mlrose.TSPOpt(length=num_queen, fitness_fn=queens_fitness, maximize=True)

    
    # Define initial state, and best results
    init_state = np.array([i for i in range(num_queen)])
    best_state = np.array([i for i in range(num_queen)])
    best_fitness = 0
    
    # Set random seed
    np.random.seed(1)

    if params["algo"] == 0: #RHC
        # Random_hill_climb(problem, max_attempts=10, max_iters=inf, restarts=0, init_state=None, curve=False, random_state=None)
        best_state, best_fitness = mlrose.random_hill_climb(problem, max_attempts=max_atp, max_iters=max_iter, init_state=init_state)
    elif params["algo"] == 1: # SA
        # Define decay schedule
        schedule = mlrose.ExpDecay()
        # Solve problem using simulated annealing
        best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_atp, max_iters=max_iter, init_state=init_state)
    elif params["algo"] == 2: #GA
        # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=inf, curve=False, random_state=None)
        best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=100, mutation_prob=0.1, max_attempts=max_atp, max_iters=max_iter)
        best_state = [int(i) for i in best_state]
    else: #MIMIC
        # mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=inf, curve=False, random_state=None)
        best_state, best_fitness = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=max_atp, max_iters=max_iter)

    best_fit_value = params["number"]*(params["number"]-1)/2
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness/(best_fit_value))
