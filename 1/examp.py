import numpy as np
import math as m
import matplotlib.pyplot as plt
import os


def calculate_actual_precision(variable_ranges, precisions):
    actual_precisions = []
    chromosome_lengths = []

    for i in range(len(variable_ranges)):
        variable_range = variable_ranges[i]
        precision = precisions[i]

        # Обчислення кількості біт
        num_bits = int(np.ceil(np.log2((variable_range[1] - variable_range[0]) / precision + 1)))
        chromosome_lengths.append(num_bits)

        # Обчислення фактичного кроку квантування
        actual_precision = (variable_range[1] - variable_range[0]) / (2**num_bits - 1)
        actual_precisions.append(round(actual_precision, 4))

    return chromosome_lengths, actual_precisions


def kovdra(variable_ranges, population_size, precisions):
    population = []

    for i in range(population_size):
        individual = []
        for j in range(len(variable_ranges)):
            variable_range = variable_ranges[j]
            precision = precisions[j]

            value = np.random.uniform(variable_range[0], variable_range[1])
            quantized_value = round(value / precision) * precision

            individual.append(round(quantized_value, 3))

        population.append(individual)
    return population
    """ return([(1, 1.667),(1, 1.333),(2, 1.667),(2, 1.333),(3, 1.667),(3, 1.333)]) """
    

def evaluate_fitness(population, n):  # Оцінюємо пристосованість популяції
    fitness_values = np.array([round(3 * abs(individual[0]) + individual[1]**2, 2) for individual in population])
    fitness_values_sum = sum(fitness_values)
    return round(fitness_values_sum / n, 3), fitness_values


def int_to_bit(population, precisions, a, c, chromosome_lengths):
    results = []
    q = chromosome_lengths[0]
    w = chromosome_lengths[1]
    for item in population:
        result1 = str(bin(int(np.ceil((item[0] - a) / precisions[0]))))[2:].zfill(q)
        result2 = str(bin(int(np.ceil((item[1] - c) / precisions[1]))))[2:].zfill(w)
        results.append((result1, result2))
    return results


import numpy as np

def select(fitness, bit):
    if np.any(fitness < 0):
        min_f = np.min(fitness)
        fitness += 2 * abs(min_f)
    
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness  # Используем вероятности в долях единицы

    sorted_indices = np.argsort(probabilities)
    probabilities = probabilities[sorted_indices]
    bit = np.array(bit)[sorted_indices]

    selected_parents = []

    for _ in range(len(probabilities)):
        x = np.random.uniform(0, 1)  # Генерируем случайное число от 0 до 1
        current_sum = 0
        for i, prob in enumerate(probabilities):
            current_sum += prob
            if x <= current_sum:
                selected_parents.append(bit[i])
                break
    #print(probabilities)
    return selected_parents






def crossover(p1, p2, chromosome_lengths):
    x = chromosome_lengths[0]
    y = chromosome_lengths[1]
    xl = np.random.randint(1, x)
    yl = np.random.randint(1, y)

    gamma = np.random.rand()
    if 0.9 <= gamma:
        child1 = p1[0][:xl] + p2[0][xl:] , p1[1][:yl] + p2[1][yl:]
        child2 = p2[0][:xl] + p1[0][xl:] , p2[1][:yl] + p1[1][yl:]
        return child1, child2
    else:
        return p1, p2
    
def mytate(p1, rate):
    mutated_individual = []

    for gene in p1:
        mutated_gene = ""

        for bit in gene:
            bit_int = int(bit)
            if np.random.rand() < rate:
                #print("da")
                mutated_bit = bit_int ^ 1
            else:
                mutated_bit = bit_int
            mutated_gene += str(mutated_bit)
        mutated_individual.append(mutated_gene)

    return mutated_individual


def bit_to_int(population, precisions, a, c):
    results = []
    
    for item in population:
        
        result1 = round(a + int('0b' + item[0], 2) * precisions[0], 2)
        result2 = round(c + int('0b' + item[1], 2) * precisions[1], 2)
        results.append((result1, result2))
    
    return results
    

    


variable_ranges = [(-6, 0), (-2, 2)]  #  [(0, 4), (1, 2)]
precisions = [0.1, 0.1]
a = -6 # 0
c = -2 #1
population_size = 50
num_iterations = 500
mutation_rate = 0.001
fitness_history = []
max_fitness_value = float('-inf')

chromosome_lengths, actual_precisions = calculate_actual_precision(variable_ranges, precisions)
#print(chromosome_lengths, actual_precisions)

population = kovdra(variable_ranges, population_size, precisions)
#print(population)

fitnes_sum, fitnes = evaluate_fitness(population, population_size)
#print(fitnes_sum, fitnes)

for run in range(num_iterations):

    max_current_fitness = np.max(fitnes)
    #print(max_current_fitness)
    fitness_history.append(max_current_fitness)

    if max_current_fitness > max_fitness_value:
        max_fitness_value = max_current_fitness
        max_fitness_iteration = run + 1
        max_fitness_point = population[np.argmax(fitnes)]


    bit = int_to_bit(population, actual_precisions, a, c, chromosome_lengths)
    #print(bit, '1')
    #print(len(fitnes), 'f')

    selected_parents = select(fitnes, bit)
    #print(len(selected_parents), 's')

    new_population = []
    for i in range(0, len(selected_parents), 2):
        p1 = selected_parents[i]
        p2 = selected_parents[i + 1]
        child1, child2 = crossover(p1, p2, chromosome_lengths)
        new_population.append(child1)
        new_population.append(child2)
    #print(new_population, '2')


    mytate_population = []
    for i in range(0, len(new_population)):
        child = new_population[i]
        child = mytate(child, mutation_rate)
        mytate_population.append(child)
    #print(mytate_population, '3')
    
    population = []
    population = bit_to_int(mytate_population, actual_precisions, a, c)
    #print(population, '4')


    fitnes_sum, fitnes = evaluate_fitness(population, population_size)
    #print(fitnes_sum, fitnes)

print(f' Максимум fitness-функції {fitness_history[max_fitness_iteration-1]}, на якій ітерації {max_fitness_iteration}')
print(max_fitness_point)

plt.plot(fitness_history)
#plt.scatter(max_fitness_iteration - 1, max_fitness_value, color='red', label='Maximum Fitness')
plt.title('Maximum Fitness Function Value over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Maximum Fitness Value')
#plt.legend()
plt.show()