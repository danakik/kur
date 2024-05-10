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


def generate_initial_population(variable_ranges, population_size, precisions):
    """ population = []

    for i in range(population_size):
        individual = []
        for j in range(len(variable_ranges)):
            variable_range = variable_ranges[j]
            precision = precisions[j]

            # Генеруємо випадкову точку у діапазоні з урахуванням точності
            value = np.random.uniform(variable_range[0], variable_range[1])
            # Квантуємо значення з урахуванням точності
            quantized_value = round(value / precision) * precision

            individual.append(round(quantized_value, 4))

        population.append(individual) """

    """ population = []
    x = np.linspace(-1, 5, 60)
    y = np.linspace(-4, 0, 60)
    population = [(round(x_val, 2), round(y_val, 2)) for x_val, y_val in zip(x, y)]
    print(population)
    return population """
    return([(1, 1.667),(1, 1.333),(2, 1.667),(2, 1.333),(3, 1.667),(3, 1.333)])


def  int_to_bit(population, precisions, a, c, chromosome_lengths):
    results = []
    q = chromosome_lengths[0]
    w = chromosome_lengths[1]
    for item in population:
        result1 = str(bin(int(np.ceil((item[0] - a) / precisions[0]))))[2:].zfill(q)
        result2 = str(bin(int(np.ceil((item[1] - c) / precisions[1]))))[2:].zfill(w)
        results.append((result1, result2))
    return results


def bit_to_int(population, precisions, a, c):
    results = []
    for item in population:
        
        result1 = round(a + int('0b' + item[0], 2) * precisions[0], 3)
        result2 = round(c + int('0b' + item[1], 2) * precisions[1], 3)
        results.append((result1, result2))
    return results


def evaluate_fitness(population, n):  # Оцінюємо пристосованість популяції
    fitness_values = np.array([round(individual[0]**2 - individual[1]**2, 3) for individual in population])
    fitness_values_sum = sum(fitness_values)
    return round(fitness_values_sum / n, 3), fitness_values


def roulette_wheel_selection(population, fitness_values):  
   

    
    if np.any(fitness_values < 0):
        
        min_fitness = np.min(fitness_values)
        fitness_values = fitness_values + 2 * abs(min_fitness)
    
    total_fitness = np.sum(fitness_values)
    probabilities = np.round((fitness_values / total_fitness) * 100, 2)
    
    tf = []
    start = 0
    for count in probabilities:
        end = round(start + count, 2)
        tf.append((start, end))
        start = end        

    tf.sort()

    fitness_dict = {}
    for i in range(len(population)):
        fitness_dict[population[i]] = tf[i]

    

    selected_parents = []
    for i in range(len(tf) ):
        x = np.random.uniform(0, 100)
        x = round(x, 2)

        for range_min, range_max in tf:
            if range_min <= x <= range_max:
                for key, value in fitness_dict.items():
                    if value == (range_min, range_max):
                        selected_parents.append(key)

    return selected_parents


def crossover(parent1, parent2):     # схрещування

    gamma = np.random.rand()

    if 0.9 <= gamma:
        crossover_point1 = np.random.randint(1, len(parent1[0]))
        crossover_point2 = np.random.randint(1, len(parent1[1]))

        child1 = parent1[0][:crossover_point1] + parent2[0][crossover_point1:] , parent1[1][:crossover_point2] + parent2[1][crossover_point2:]
        child2 = parent2[0][:crossover_point1] + parent1[0][crossover_point1:] , parent2[1][:crossover_point2] + parent1[1][crossover_point2:]
        #print(f'p - {parent1, parent2} c - {child1, child2}')
        return child1, child2
    else:
        return parent1, parent2
    
def mutate(individual, mutation_rate, population):
    mutated_individual = []

    for gene in individual:
        mutated_gene = ""

        for bit in gene:
            bit_int = int(bit)
            if np.random.rand() < mutation_rate:
                mutated_bit = bit_int ^ 1
            else:
                mutated_bit = bit_int
            mutated_gene += str(mutated_bit)
        mutated_individual.append(mutated_gene)

    return mutated_individual



variable_ranges = [(0, 4), (1, 2)] # [(-1, 5), (-4, 0)]
a = 0 #1
c = 1 #-4
precisions = [0.1, 0.1]
population_size = 6
num_iterations = 50
mutation_rate = 0.01
fitness_history = []



# Обчислення розміру хромосоми та фактичного кроку квантування 
chromosome_lengths, actual_precisions = calculate_actual_precision(variable_ranges, precisions)
# Генерація початкової популяції
initial_population = generate_initial_population(variable_ranges, population_size, precisions)

chromosomes = int_to_bit(initial_population, actual_precisions, a, c, chromosome_lengths)
fitness_values_sum, fitness_values = evaluate_fitness(initial_population, population_size)


print(actual_precisions)
max_fitness_value = float('-inf')



""" print(f' iter - 0 {initial_population}')
print(fitness_values_sum, fitness_values) """
for run in range(num_iterations):

    # Зберігаємо максимальне значення фітнес-функції та відповідну ітерацію
    max_current_fitness = np.max(fitness_values_sum)
    fitness_history.append(max_current_fitness)

    if max_current_fitness > max_fitness_value:
        max_fitness_value = max_current_fitness
        max_fitness_iteration = run + 1

    
    selected_parents = roulette_wheel_selection(chromosomes, fitness_values)


    new_generation = []
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]


        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate, selected_parents)
        child2 = mutate(child2, mutation_rate, selected_parents)
        new_generation.extend([child1, child2])

    #print(new_generation)
    initial_population = new_generation
    initial_population = bit_to_int(initial_population, actual_precisions, a, c)
    #print(f' iter - {run+1} {initial_population}')


    fitness_values_sum, fitness_values = evaluate_fitness(initial_population, population_size)
    print(fitness_values_sum, fitness_values)


plt.plot(fitness_history)
plt.scatter(max_fitness_iteration - 1, max_fitness_value, color='red', label='Maximum Fitness')
plt.title('Fitness Function over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.legend()
plt.show()
