from P2 import Algorithm
from copy import deepcopy
import numpy as np

# create object of algorithm and create first population
alg = Algorithm()
first_parent = alg.create_first_population()
# set mutationRate and sigma
alg.set_att(0.05, 0.01)
parents = deepcopy(first_parent)

for i in range(alg.numberOfGenerations):
    print("iteration : ", i)
    # calculate the fitness
    parents_fitness = alg.calculate_fitness(parents)
    # choose the best
    selected_parents = alg.choose_parents(parents, parents_fitness)
    # cross over and mutation
    child = alg.crossover(selected_parents)
    # choose the best from parents and children
    child_fitness = alg.calculate_fitness(child)
    generation = np.concatenate((parents, child))
    generation_fitness = np.concatenate((parents_fitness, child_fitness))
    generation_fitness_sort = -np.sort(-generation_fitness)
    for j in range(alg.populationSize):
        parents[j] = generation[generation_fitness.tolist().index(generation_fitness_sort[j])]

# find the best
final_fit = alg.calculate_fitness(parents)
best_parent = parents[final_fit.tolist().index(final_fit.max())]
print("best parent :  ", best_parent, " with fitness : ", final_fit.max())

