import numpy as np
import matplotlib.pyplot as plt

def calc_fitness(chromosome_lengths, precisions):
    actual_precisions = []
    lengths = []

    for i in range(len(chromosome_lengths)):
        length = chromosome_lengths[i]
        precision = precisions[i]

        num_bits = int(np.ceil(np.log2((length[1] - length[0]) / precision + 1)))
        lengths.append(num_bits)

        actual_precision = (length[1] - length[0]) / (2**num_bits - 1)
        actual_precisions.append(round(actual_precision, 4))

    return lengths, actual_precisions

def init_population(chromosome_lengths, pop_size, precisions, seed=40):
    np.random.seed(seed)
    population = []

    for _ in range(pop_size):
        individual = []
        for j in range(len(chromosome_lengths)):
            length = chromosome_lengths[j]
            precision = precisions[j]

            value = np.random.uniform(length[0], length[1])
            quantized_value = round(value / precision) * precision

            individual.append(round(quantized_value, 3))

        population.append(individual)
    return population

def fitness_evaluation(population, n):
    fitness_values = np.array([round(abs(individual[0]) - 2 * individual[1], 2) for individual in population])
    fitness_values_sum = sum(fitness_values)
    return round(fitness_values_sum / n, 3), fitness_values

def int_to_bit(population, precisions, a, c, lengths):
    results = []
    q = lengths[0]
    w = lengths[1]
    for item in population:
        result1 = str(bin(int(np.ceil((item[0] - a) / precisions[0]))))[2:].zfill(q)
        result2 = str(bin(int(np.ceil((item[1] - c) / precisions[1]))))[2:].zfill(w)
        results.append((result1, result2))
    return results

def selection(fitness, bit):
    if np.any(fitness < 0):
        min_f = np.min(fitness)
        fitness += 2 * abs(min_f)
    
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness  

    sorted_indices = np.argsort(probabilities)
    probabilities = probabilities[sorted_indices]
    bit = np.array(bit)[sorted_indices]

    selected_parents = []

    for _ in range(len(probabilities)):
        x = np.random.uniform(0, 1)  
        current_sum = 0
        for i, prob in enumerate(probabilities):
            current_sum += prob
            if x <= current_sum:
                selected_parents.append(bit[i])
                break

    return selected_parents

def crossover(p1, p2, lengths):
    x = lengths[0]
    y = lengths[1]
    xl = np.random.randint(1, x)
    yl = np.random.randint(1, y)

    gamma = np.random.rand()
    if 0.9 <= gamma:
        child1 = p1[0][:xl] + p2[0][xl:] , p1[1][:yl] + p2[1][yl:]
        child2 = p2[0][:xl] + p1[0][xl:] , p2[1][:yl] + p1[1][yl:]
        return child1, child2
    else:
        return p1, p2

def mutation(p1, rate):
    mutated_individual = []

    for gene in p1:
        mutated_gene = ""

        for bit in gene:
            bit_int = int(bit)
            if np.random.rand() < rate:
                mutated_bit = ~bit_int & 1
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

def main():
    ranges = [(-3, 3), (1, 5)]
    precisions = [0.1, 0.1]
    a = -3
    c = 1
    pop_size = 60
    iterations = 600
    mutation_rate = 0.001
    fitness_history = []
    max_fitness_value = float('-inf')

    lengths, actual_precisions = calc_fitness(ranges, precisions)
    population = init_population(ranges, pop_size, precisions)
    fit_sum, fit = fitness_evaluation(population, pop_size)

    for run in range(iterations):

        max_current_fitness = np.max(fit)
        fitness_history.append(max_current_fitness)

        if max_current_fitness > max_fitness_value:
            max_fitness_value = max_current_fitness
            max_fitness_iteration = run + 1

        bit = int_to_bit(population, actual_precisions, a, c, lengths)
        selected_parents = selection(fit, bit)

        new_population = []
        for i in range(0, len(selected_parents), 2):
            p1 = selected_parents[i]
            p2 = selected_parents[i + 1]
            child1, child2 = crossover(p1, p2, lengths)
            new_population.append(child1)
            new_population.append(child2)

        mutated_population = []
        for i in range(0, len(new_population)):
            child = new_population[i]
            child = mutation(child, mutation_rate)
            mutated_population.append(child)
        
        population = bit_to_int(mutated_population, actual_precisions, a, c)

        fit_sum, fit = fitness_evaluation(population, pop_size)

    print(fitness_history[max_fitness_iteration-1])
    print(max_fitness_iteration-1)

    plt.plot(fitness_history, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.show()

if __name__ == "__main__":
    main()
