"""Evolutionary algorithm explorer for BADGERS.

Uses a genetic algorithm to evolve guide sequences that maximize
a fitness function (Cas13Mult or Cas13Diff).
"""

import random
import numpy as np

from adapt_reimpl.sequence_utils import (
    one_hot_encode_base,
    convert_to_nt,
    reverse_complement,
)


class EvolutionaryExplorer:
    """Evolutionary algorithm for guide design.

    Maintains a population of guide sequences and evolves them using
    mutation, crossover, and selection based on a fitness function.
    """

    def __init__(self, fitness_model, guide_length=28,
                 population_size=100, mutation_rate=0.01,
                 crossover_rate=0.7, elite_fraction=0.1,
                 num_generations=100, seed=None):
        """Initialize the evolutionary explorer.

        Args:
            fitness_model: object with get_fitness(guide_seqs) method
            guide_length: length of guide sequences
            population_size: number of individuals in population
            mutation_rate: per-base mutation probability
            crossover_rate: probability of crossover
            elite_fraction: fraction of top individuals preserved
            num_generations: number of generations to evolve
            seed: random seed
        """
        self.fitness_model = fitness_model
        self.guide_length = guide_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.num_generations = num_generations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.best_guide = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

    def _random_guide(self):
        """Generate a random guide sequence.

        Returns:
            random 28-nt guide string
        """
        bases = 'ACGT'
        return ''.join(random.choice(bases) for _ in range(self.guide_length))

    def _initialize_population(self):
        """Initialize population from parent sequences and random guides.

        Returns:
            list of guide sequences
        """
        population = []

        # Seed with parent sequences from fitness model
        if hasattr(self.fitness_model, 'parent_seqs'):
            for parent in self.fitness_model.parent_seqs:
                if len(parent) >= self.guide_length:
                    # Extract guide-length portion from middle
                    start = (len(parent) - self.guide_length) // 2
                    guide = parent[start:start + self.guide_length]
                    if 'N' not in guide and '-' not in guide:
                        population.append(guide)
                        if len(population) >= self.population_size:
                            return population

        # Fill rest with random guides
        while len(population) < self.population_size:
            population.append(self._random_guide())

        return population

    def _mutate(self, guide):
        """Mutate a guide sequence.

        Args:
            guide: guide sequence string

        Returns:
            mutated guide string
        """
        bases = 'ACGT'
        mutated = list(guide)
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(bases)
        return ''.join(mutated)

    def _crossover(self, parent1, parent2):
        """Single-point crossover between two guides.

        Args:
            parent1, parent2: guide sequence strings

        Returns:
            (child1, child2) from crossover
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _select_parents(self, population, fitnesses):
        """Select parents via tournament selection.

        Args:
            population: list of guide sequences
            fitnesses: numpy array of fitness scores

        Returns:
            (parent1, parent2) tuple
        """
        def tournament():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            if fitnesses[idx1] >= fitnesses[idx2]:
                return population[idx1]
            return population[idx2]

        return tournament(), tournament()

    def run(self):
        """Run the evolutionary algorithm.

        Returns:
            (best_guide, best_fitness) tuple
        """
        population = self._initialize_population()

        for gen in range(self.num_generations):
            # Evaluate fitness
            fitnesses = self.fitness_model.get_fitness(population)

            # Track best
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_guide = population[best_idx]
            self.fitness_history.append(self.best_fitness)

            # Elitism: preserve top individuals
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            new_population = [population[i] for i in elite_indices]

            # Generate offspring
            while len(new_population) < self.population_size:
                p1, p2 = self._select_parents(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(c2))

            population = new_population[:self.population_size]

            if gen % 10 == 0:
                print(f"  Gen {gen}: best fitness = {self.best_fitness:.4f}")

        return self.best_guide, self.best_fitness

    def get_best_guides(self, n=10):
        """Return the n best guides found.

        Must be called after run().

        Args:
            n: number of guides to return

        Returns:
            list of (guide_seq, fitness) tuples
        """
        if not hasattr(self, '_final_population'):
            return [(self.best_guide, self.best_fitness)]
        fitnesses = self.fitness_model.get_fitness(self._final_population)
        sorted_indices = np.argsort(fitnesses)[::-1]
        results = []
        for i in sorted_indices[:n]:
            results.append((self._final_population[i], fitnesses[i]))
        return results
