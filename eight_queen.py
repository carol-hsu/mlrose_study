## eight queen problem by mlrose
import mlrose
import numpy as np

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness


if __name__ == "__main__":

    #fitness funtion
    queens_fitness = mlrose.CustomFitness(queens_max)

    #optimization problems: DiscreteOpt / ContinuousOpt / TSPOpt
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=queens_fitness, maximize=False, max_val=8)

    # Define decay schedule
    schedule = mlrose.ExpDecay()
    
    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    # Set random seed
    np.random.seed(3)
    
    # Solve problem using simulated annealing
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=100, max_iters=10000, init_state=init_state)
    
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
