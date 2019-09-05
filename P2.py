import numpy as np
import pandas as pd


class Algorithm:
    def __init__(self):
        self.numberOfGenerations = 10000
        self.populationSize = 50
        self.tornumentSize = 2
        self.mutationRate = 0.1
        self.sigma = 1
        self.X_array = np.arange(1, 101)
        data = pd.read_csv("my_numbers.csv")
        self.Y_array = data["Y"].values
        # for degree 3 we should predict 4 numbers : a0, a1, a2, a3
        self.degree = 4

    # set mutationRate and sigma
    def set_att(self, r, s):
        self.mutationRate = r
        self.sigma = s

    # create first population
    def create_first_population(self):
        return np.random.uniform(low=0, high=1, size=(self.populationSize, self.degree))

    # calculate fitness
    def calculate_fitness(self, my_array):
        fitness = np.zeros(len(my_array))
        for i in range(my_array.shape[0]):
            my_y = my_array[i][0] * np.power(self.X_array, 3) + my_array[i][1] * np.power(self.X_array, 2) + my_array[i][2] * self.X_array + my_array[i][3]
            fitness[i] = 1 / (1 + np.sum(np.power(np.subtract(my_y, self.Y_array), 2)) / len(my_y))

        return fitness

    # select best parents
    def choose_parents(self, parents, fitness):
        selected_parents = np.zeros(shape=(10, self.degree))
        for i in range(10):
            random_numbers = np.arange(len(parents))
            np.random.shuffle(random_numbers)
            random_numbers = random_numbers[:2]
            random_fitness = []
            for j in range(len(random_numbers)):
                random_fitness.append(fitness[random_numbers[j]])
            random_fitness.sort(reverse=True)
            selected_parents[i] = parents[fitness.tolist().index(random_fitness[0])]

        return selected_parents

    # cross over and mutation
    def crossover(self, parents):
        children = []
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                parent1 = parents[i]
                parent2 = parents[j]
                selected = np.arange(self.degree)
                np.random.shuffle(selected)
                final_parents = [parent1[selected[0]], parent1[selected[1]], parent2[selected[2]], parent2[selected[3]]]
                probability_mutation = np.random.uniform(low=0, high=100, size=self.degree)
                for k in range(len(probability_mutation)):
                    if probability_mutation[k] / 100 <= self.mutationRate:
                        final_parents[k] += np.random.normal(0, self.sigma, 1)[0]

                children.append(final_parents)

        return np.asarray(children)






