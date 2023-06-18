import os
import random
import itertools
from copy import copy

import inspyred
import numpy as np

from lamps import LampsEnvironmentConfig as LEConfig, LampsEnvironment as LE
from utils import parse_args, set_rand_gen

# lamp classical = [x, y]

def lamp_generator(random, args):
    env = args.get('env')
    problem_size = env.config.problem_size
    room_size = env.config.room_size
    number_lamps = random.randint(problem_size, problem_size*3) # according to the paper

    return [[random.uniform(0, 1) * room_size, random.uniform(0, 1) * room_size] 
            for _ in range(number_lamps)] 

# def decorator_bounder(func):
#     def wrapper(candidates, args):
#         room_size = args.get('env').config.room_size
#         func.lower_bound = itertools.repeat(0)
#         func.upper_bound = itertools.repeat(room_size)

#         return func(candidates, args)

#     return wrapper

# @decorator_bounder
def lamp_bounder(candidate, args):
    room_size = args.get('env').config.room_size
    for i, lamp in enumerate(candidate):
        x = max(min(lamp[0], room_size), 0)
        y = max(min(lamp[1], room_size), 0)
        candidate[i] = [x, y]
    return candidate

def lamp_evaluator(candidates, args):
    results = []
    env = args.get('env')
    for candidate in candidates:
        env.set_lamps(candidate)
        fitness = env.calc_enlightened_area() - env.config.alpha * env.calc_overlap_area()
        # results.append(env.calc_fitness())
        results.append(fitness)
    
    return results


def lamp_crossover(random, candidates, args):
    crossover_rate = args.setdefault('crossover_rate', 1.0)

    def crossover(random, mom, dad, crossover_rate):
        children = []
        
        if random.random() < crossover_rate:
            bro = copy(dad)
            sis = copy(mom)
            length = min(len(mom), len(dad))
            index = random.randint(0, length-1)
            child1 = bro[:index] + sis[index:]
            child2 = sis[:index] + bro[index:]
            children.append(child1)
            children.append(child2)
        else:
            children.append(mom)
            children.append(dad)
        return children

    if len(candidates) % 2 == 1:
        candidates = candidates[:-1]
    children = []
    for i, (mom, dad) in enumerate(zip(candidates[::2], candidates[1::2])):
        crossover.index = i
        offspring = crossover(random, mom, dad, crossover_rate)
        for o in offspring:
            children.append(o)
    return children


def lamp_mutator(random, candidates, args):
    mutation_rate = args.setdefault('mutation_rate', 0.1)
    problem_size = args.get('env').config.problem_size
    room_size = args.get('env').config.room_size
    bounder = args['_ec'].bounder

    for i, candidate in enumerate(candidates):
        for j, (lamp, lo, hi) in enumerate(zip(candidate, bounder.lower_bound, bounder.upper_bound)):
            if random.random() < mutation_rate:
                rd = random.random()
                if rd < 0.5:
                    # Change the position of the lamp
                    x = lamp[0] + random.gauss(0, 0.1) * (hi - lo)
                    y = lamp[1] + random.gauss(0, 0.1) * (hi - lo)
                    candidates[i][j] = [x, y]
                elif rd < 0.75 and len(candidate) < 3 * problem_size:
                    # Add a new lamp
                    candidate.append([random.uniform(0, room_size), random.uniform(0, room_size)])
                elif len(candidate) > problem_size:
                    # Remove a lamp
                    index = random.randint(0, len(candidate)-1)
                    del candidate[index]
                else:
                    # Do nothing
                    pass
        candidates[i] = bounder(candidates[i], args)
    return candidates
    
def lamp_observer(population, num_generations, num_evaluations, args):
    avg_num_lamps = np.mean([len(c.candidate) for c in population])
    best = max(population)
    print('{0:6} -- {1:.3f} : {2}'.format(num_generations, best.fitness, avg_num_lamps))

    return None

if __name__ == "__main__":
    args = parse_args()
    config = LEConfig(
        room_size = args.room_size,
        problem_size = args.problem_size,
        grid_scale = args.grid_scale,
        alpha = args.alpha,
    )
    env = LE(config)
    random_number_generator = set_rand_gen(args.seed)
    if args.output_dir != '':
        os.makedirs(args.output_dir, exist_ok=True)

    lamp_bounder.lower_bound = itertools.repeat(0)
    lamp_bounder.upper_bound = itertools.repeat(config.room_size)

    ec_algo = inspyred.ec.EvolutionaryComputation(random_number_generator)
    ec_algo.selector = inspyred.ec.selectors.default_selection # by default, tournament selection has tau=2 (two individuals), but it can be modified (see below)
    ec_algo.variator = [lamp_crossover, lamp_mutator] # the genetic operators are put in a list, and executed one after the other
    ec_algo.replacer = inspyred.ec.replacers.plus_replacement # "plus" -> "mu+lambda"
    ec_algo.observer = lamp_observer
    ec_algo.terminator = [inspyred.ec.terminators.evaluation_termination, inspyred.ec.terminators.average_fitness_termination] # the algorithm terminates when a given number of evaluations (see below) is reached

    print(f"Running ec ...with config {env.config}, lamp_radius = {env.lamp_radius:.3f}")
    print(f"    num_generations -- best_fitness : avg_num_lamps")
    final_population = ec_algo.evolve( 
                                generator = lamp_generator,
                                evaluator = lamp_evaluator,
                                pop_size = 100, # size of the population
                                bounder = lamp_bounder,
                                num_selected = 200, # size of the offspring (children individuals)
                                maximize = True, 
                                max_evaluations = 2000, # maximum number of evaluations before stopping, used by the terminator
                                tournament_size = 2, # size of the tournament selection; we need to specify it only if we need it different from 2
                                crossover_rate = 1.0, # probability of applying crossover
                                mutation_rate = 0.1, # probability of applying mutation
                                # customized args
                                env = env,
    )

    print('Terminated due to {0}.'.format(ec_algo.termination_cause))
    final_population.sort(reverse=True)
    final_individuals = [final_population[i].candidate for i in range(len(final_population))]
    print(f"Displaying the final population statistics ...")

    final_fitness, final_enlightened_areas, final_overlap_areas, final_num_lamps = [], [], [], []
    for i, lamps in enumerate(final_individuals):
        env.set_lamps(lamps)
        enlightened_area, overlap_area = env.calc_enlightened_area(), env.calc_overlap_area()
        fitness = env.calc_fitness()
        final_fitness.append(fitness)
        final_enlightened_areas.append(enlightened_area)
        final_overlap_areas.append(overlap_area)
        final_num_lamps.append(len(lamps))
    avg_fitness = np.mean(final_fitness)
    std_fitness = np.std(final_fitness)
    avg_enlightened_area = np.mean(final_enlightened_areas)
    std_enlightened_area = np.std(final_enlightened_areas)
    avg_overlap_area = np.mean(final_overlap_areas)
    std_overlap_area = np.std(final_overlap_areas)
    avg_num_lamps = np.mean(final_num_lamps)
    std_num_lamps = np.std(final_num_lamps)

    print(f"    avg_fitness = {avg_fitness:.4f}\n    std_fitness = {std_fitness:.4f}")
    print(f"    avg_enlightened_area = {avg_enlightened_area:.4f}\n    std_enlightened_area = {std_enlightened_area:.4f}")
    print(f"    avg_num_lamps = {avg_num_lamps:.4f}\n    std_num_lamps = {std_num_lamps:.4f}")
    print(f"    avg_overlap_area = {avg_overlap_area:.4f}\n    std_overlap_area = {std_overlap_area:.4f}")

    best_individual = final_individuals[0]
    print(f"Displaying the best individual in the final population ...")
    for i, lamp in enumerate(best_individual):
        print(f"Lamp {i} at ({lamp[0]:.2f}, {lamp[1]:.2f})")
    env.set_lamps(best_individual)
    if args.output_dir != '':
        path = os.path.join(args.output_dir, 'best_individual.png')
        print(f"Saving the best individual to {path} ...")
        env.display(path)