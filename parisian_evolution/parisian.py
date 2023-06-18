import random
import itertools
import os
from copy import copy, deepcopy

import inspyred
import numpy as np

from lamps import LampsEnvironmentConfig as LEConfig, LampsEnvironment as LE
from utils import parse_args, set_rand_gen

# parisian lamp = [x, y, e, local_fitness]

def lamp_bounder(candidate, args):
    room_size = args.get('env').config.room_size
    for i, lamp in enumerate(candidate):
        x = max(min(lamp[0], room_size), 0)
        y = max(min(lamp[1], room_size), 0)
        candidate[i] = [x, y, lamp[2], lamp[3]]
    return candidate

def parisian_selector(random, population, args):
    expressed_candidates = []
    for p in population:
        candidate = p.candidate
        expressed_candidate = [[c[0], c[1]] for c in candidate if c[2] > 0]
        expressed_candidates.append(expressed_candidate)
    return expressed_candidates

def parisian_generator(random, args):
    env = args.get('env')
    problem_size = env.config.problem_size
    room_size = env.config.room_size 
    number_lamps = random.randint(problem_size, problem_size*3)

    # individual = []
    # for _ in range(number_lamps): 
    #     x, y = random.uniform(0, 1) * room_size, random.uniform(0, 1) * room_size
    #     individual.append([x, y, 1, 0])
    individual = [[random.uniform(0, 1) * room_size, random.uniform(0, 1) * room_size, 1, 2] 
            for _ in range(number_lamps)] 
    # print(f"ind=\n{individual}")
    return individual

def calc_global_fitness(candidates, args):
    results = []
    env = args.get('env')
    for candidate in candidates:
        env.set_lamps(candidate)
        fitness = env.calc_enlightened_area() - env.config.alpha * env.calc_overlap_area()
        results.append(fitness)
    return results

def parisian_evaluator(candidates, args): 
    expressed_candidates = []
    for candidate in candidates:
        expressed_candidate = [[c[0], c[1]] for c in candidate if c[2] > 0]
        expressed_candidates.append(expressed_candidate)
    return calc_global_fitness(expressed_candidates, args)

# calculate the sharing fitness in Parisian algorithm
def calc_sharing_fitness(lamps, index, lamp_radius):
    sharings = []
    pos = np.array(lamps[index][:2])
    # print(f"lamps = {lamps}")
    # print(f"pos = {pos}")
    for lamp in lamps:
        distance = np.linalg.norm(pos - np.array(lamp[:2]))
        # print(f"d = {distance}, 2r = {2 * lamp_radius}, d>2r = {distance > 2 * lamp_radius}")
        sharing = 1 - distance / float(2 * lamp_radius) if (distance > 2 * lamp_radius) else 0
        sharings.append(sharing)
    # print(f"sharings = {sharings}")
    if sum(sharings) == 0:
        return 0
    return lamps[index][3] / sum(sharings)

def parisian_process_income(random, candidates, args): # TODO should be called before evaluator ... maybe args.sharing_fitness
    # new individual added, decide whether to delete one (addition or replacement)
    for candidate in candidates:
        if random.random() < 0.5:
            sharing_fitness = [calc_sharing_fitness(candidate, i, env.lamp_radius) for i in range(len(candidate) - 1)]
            # print(f"sharing_fitness={sharing_fitness}")
            index_min = np.argmin(np.array(sharing_fitness))
            del candidate[index_min]
        else:
            pass
    return candidates

def parisian_set_local_fitness(random, candidates, args): # TODO args.global_fitness
    for i, candidate in enumerate(candidates):
        new_individual = deepcopy(candidate[-1])
        del candidate[-1]
        global_fitness = parisian_evaluator([candidate], args)[0]
        candidate.append(new_individual)
        new_global_fitness = parisian_evaluator([candidate], args)[0]
        new_individual = candidate[-1]

        if new_individual[2] == 0:
            new_individual[3] = 1
        elif new_global_fitness > global_fitness:
            new_individual[3] = 2
        else: 
            new_individual[3] = 0

    return candidates

def parisian_crossover(random, candidates, args): 
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
    
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    if len(candidates) % 2 == 1:
        candidates = candidates[:-1]
    children = []
    for i, (mom, dad) in enumerate(zip(candidates[::2], candidates[1::2])):
        crossover.index = i
        offspring = crossover(random, mom, dad, crossover_rate)
        for o in offspring:
            children.append(o)
    return children

def parisian_mutator(random, candidates, args):
    mutation_rate = args.setdefault('mutation_rate', 0.1)
    problem_size = args.get('env').config.problem_size
    room_size = args.get('env').config.room_size
    bounder = args['_ec'].bounder

    for i, candidate in enumerate(candidates):
        for j, (lamp, lo, hi) in enumerate(zip(candidate, bounder.lower_bound, bounder.upper_bound)):
            if random.random() < mutation_rate:
                rd = random.random()
                e, local_fitness = lamp[2], lamp[3]
                if rd < 0.5:
                    # Change the position of the lamp
                    x = lamp[0] + random.gauss(0, 0.1) * (hi - lo)
                    y = lamp[1] + random.gauss(0, 0.1) * (hi - lo)
                    candidates[i][j] = [x, y, e, local_fitness]
                elif rd < 0.75 and len(candidate) < 2 * problem_size:
                    # Add a new lamp
                    candidate.append([random.uniform(0, room_size), random.uniform(0, room_size), 1, 0])
                elif len(candidate) > problem_size:
                    # Remove a lamp
                    index = random.randint(0, len(candidate)-1)
                    del candidate[index]
                else:
                    # Do nothing
                    pass
        candidates[i] = bounder(candidates[i], args)
    return candidates

def parisian_switch(random, candidates, args):
    # switch on/off the lamps
    switch_rate = args.setdefault('switch_rate', 0.1)
    for candidate in candidates:
        # print(candidate)
        for lamp in candidate:
            if random.random() < switch_rate:
                lamp[2] = 1 - lamp[2]
    return candidates
   
def parisian_observer(population, num_generations, num_evaluations, args):
    avg_num_lamps = np.mean([len(c.candidate) for c in population])
    best = max(population)
    print('{0:6} -- {1:.3f} : {2}'.format(num_generations, best.fitness, avg_num_lamps))

    return None

def evolve():
    pass

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
    # ec_algo.selector = parisian_selector
    ec_algo.variator = [parisian_process_income, parisian_crossover, parisian_mutator, parisian_switch]
    ec_algo.replacer = inspyred.ec.replacers.plus_replacement
    ec_algo.observer = parisian_observer
    ec_algo.terminator = [inspyred.ec.terminators.evaluation_termination, inspyred.ec.terminators.average_fitness_termination]

    print(f"Running parisian ... with config {env.config}, lamp_radius = {env.lamp_radius:.3f}")
    print(f"    num_generations -- best_fitness : avg_num_lamps")
    final_population = ec_algo.evolve( 
                                generator = parisian_generator,
                                evaluator = parisian_evaluator,
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
                                sharing_fitness = 0,
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