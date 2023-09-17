# from genetic_algorithm.ga_operations import *
from genetic_algorithm.chromosome import *


class Algorithm:
    """
    Class representing the algorithm
    """

    def __init__(self, population, iterations, inputs, outputs, epoch_feedback=500):
        """
        Constructor for Alrogithm class
        @param: population - population for the current algorithm
        @param: iterations - number of iterations for the algorithm
        @param: inputs - inputs
        @param: outputs - outputs
        @param: epoch_feedback - number of epochs to show feedback
        """
        self.population = population
        self.iterations = iterations
        self.inputs = inputs
        self.outputs = outputs
        self.epoch_feedback = epoch_feedback
        # fitness is a list for saving best fitness in each age
        self.fitness = []

    def __one_step(self):
        """
        Function to do one step of the algorithm 
        """
        mother = self.__selection()
        father = self.__selection()
        # mother = self.__roulette_selection(self.population)
        # father = self.__roulette_selection(self.population)
        child = self.__cross_over(mother, father)
        child = self.__mutate(child)
        child.calculate_fitness(self.inputs, self.outputs)
        self.population = self.__replace_worst(child)

    def train(self):
        for i in range(len(self.population.list)):
            self.population.list[i].calculate_fitness(self.inputs, self.outputs)
        for i in range(self.iterations):
            best_so_far = self.__get_best()
            self.fitness.append(best_so_far.fitness)
            if i % self.epoch_feedback == 0:
                print(f"Best function: {best_so_far}")
                print(f"Best fitness: {best_so_far.fitness}")
            self.__one_step()

            if best_so_far.fitness == 1:
                break
        return self.__get_best()


    def __traversal(self, poz, chromosome: Chromosome):
        """
        Function to traverse the tree from the given poz
        @param: poz - start position
        @chromosome: chromosome to be traversed
        """
        if chromosome.gen[poz] in chromosome.terminal_set:
            return poz + 1
        elif chromosome.gen[poz] in chromosome.func_set[1]:
            return self.__traversal(poz + 1, chromosome)
        else:
            new_poz = self.__traversal(poz + 1, chromosome)
            return self.__traversal(new_poz, chromosome)

    def __mutate(self, chromosome: Chromosome):
        """
        Function to mutate a chromosome
        @param: chromsome - chromosome to be mutated
        @return: the mutated chromosome
        """
        poz = np.random.randint(len(chromosome.gen))
        if chromosome.gen[poz] in chromosome.func_set[1] + chromosome.func_set[2]:
            if chromosome.gen[poz] in chromosome.func_set[1]:
                chromosome.gen[poz] = random.choice(chromosome.func_set[1])
            else:
                chromosome.gen[poz] = random.choice(chromosome.func_set[2])
        else:
            chromosome.gen[poz] = random.choice(chromosome.terminal_set)
        return chromosome

    def __selection(self):

        population = self.population
        num_sel = self.population.num_selected
        """
        Function to select a member of the population for crossing over
        @param: population - population of chromosomes
        @param: num_sel - number of chromosome selected from the population
        @return: the selected chromosome
        """

        sample = random.sample(population.list, num_sel)
        best = sample[0]
        for i in range(1, len(sample)):
            if population.list[i].fitness > best.fitness:
                best = population.list[i]

        return best

    def __cross_over(self, mother, father):

        max_depth = self.population.max_depth
        """
        Function to cross over two chromosomes in order to obtain a child
        @param mother: - chromosome
        @param father: - chromosome
        @param max_depth - maximum_depth of a tree
        """
        child = Chromosome(mother.terminal_set, mother.func_set, mother.depth, None)
        start_m = np.random.randint(len(mother.gen))
        start_f = np.random.randint(len(father.gen))
        end_m = self.__traversal(start_m, mother)
        end_f = self.__traversal(start_f, father)
        child.gen = mother.gen[:start_m] + father.gen[start_f: end_f] + mother.gen[end_m:]
        if child.get_depth() > max_depth and random.random() > 0.2:
            child = Chromosome(mother.terminal_set, mother.func_set, mother.depth)
        return child

    def __get_best(self):

        population = self.population
        """
        Function to get the best chromosome from the population
        @param: population to get the best chromosome from
        @return: best chromosome from population
        """
        best = population.list[0]
        for i in range(1, len(population.list)):
            if population.list[i].fitness > best.fitness:
                best = population.list[i]

        return best

    def __get_worst(self):

        population = self.population
        """
        Function to get the worst chromosome of the population
        @param: population -
        @return: worst chromosome from the population
        """
        worst = population.list[0]
        for i in range(1, len(population.list)):
            if population.list[i].fitness < worst.fitness:
                worst = population.list[i]

        return worst

    def __replace_worst(self, chromosome):

        population = self.population
        """
        Function to change the worst chromosome of the population with a new one
        @param: population - population
        @param: chromosome - chromosome to be added
        """
        worst = self.__get_worst()
        if chromosome.fitness > worst.fitness:
            for i in range(len(population.list)):
                if population.list[i].fitness == worst.fitness:
                    population.list[i] = chromosome
                    break
        return population

    def __roulette_selection(self):
        population = self.population
        """
        Function to select a member of the population using roulette selection
        @param: population - population to be selected from
        """
        fitness = [chrom.fitness for chrom in population.list]
        order = [x for x in range(len(fitness))]
        order = sorted(order, key=lambda x: fitness[x])
        fs = [fitness[order[i]] for i in range(len(fitness))]
        sum_fs = sum(fs)
        max_fs = max(fs)
        min_fs = min(fs)
        p = random.random() * sum_fs
        t = max_fs + min_fs
        choosen = order[0]
        for i in range(len(fitness)):
            p -= (t - fitness[order[i]])
            if p < 0:
                choosen = order[i]
                break
        return population.list[choosen]



    def get_history_fitness_train(self):
        return self.fitness

    def get_number_iter_learn(self):
        return len(self.fitness)

