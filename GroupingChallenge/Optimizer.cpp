#include "Optimizer.h"

using namespace NGroupingChallenge;

COptimizer::COptimizer(CGroupingEvaluator& cEvaluator)
    : c_evaluator(cEvaluator) {
    random_device c_seed_generator;
    c_random_engine.seed(c_seed_generator());
}

COptimizer::~COptimizer()
{
	for (int i = 0; i < v_population.size(); i++) {
		delete v_population[i];
	}
	v_population.clear();

	delete pc_better_evaluator;
}

void COptimizer::vInitialize() {
    d_current_best_fitness = std::numeric_limits<double>::max();
    v_current_best.clear();
    v_current_best.resize(c_evaluator.iGetNumberOfPoints());
    v_population.clear();
    i_greedy_counter = 1;
	i_reset_counter = 1;

	pc_better_evaluator = new CBetterEvaluator(c_evaluator.iGetUpperBound(), c_evaluator.vGetPoints());


#pragma omp parallel for
    for (int i = 0; i < I_POP_SIZE; i++) {
        CIndividual* individual = new CIndividual(this);
        individual->dEvaluate();
#pragma omp critical
        v_population.push_back(individual);
    }
}

void COptimizer::vRunIteration() {
    vector<CIndividual*> v_new_population;

    // Parallelize creation of new individuals
#pragma omp parallel
    {
        vector<CIndividual*> thread_population;

#pragma omp for nowait
        for (int i = 0; i < I_POP_SIZE / 2; i++) {
            CIndividual* parent1 = pcTournament();
            CIndividual* parent2 = pcTournament();

            while (parent1->v_genotype == parent2->v_genotype) {
                delete parent2;
                parent2 = pcTournament();
            }

            CIndividual* child1 = new CIndividual(*parent1);
            CIndividual* child2 = new CIndividual(*parent2);
            parent1->vUniformCrossover(parent2, child1, child2);

            delete parent1;
            delete parent2;

            child1->vMutate();
            child2->vMutate();

            thread_population.push_back(child1);
            thread_population.push_back(child2);
        }

#pragma omp critical
        v_new_population.insert(v_new_population.end(), thread_population.begin(), thread_population.end());
    }

    for (int i = 0; i < v_population.size(); i++) {
        delete v_population[i];
    }
    v_population.swap(v_new_population);

    while (v_population.size() > I_POP_SIZE) {
        delete v_population.back();
        v_population.pop_back();
    }

    // Parallelize fitness evaluation
#pragma omp parallel for
    for (int i = 0; i < v_population.size(); i++) {
        v_population[i]->dEvaluate();
    }

    if (i_greedy_counter == I_GREEDY_ITERATIONS) {
        cout << "JAZDA Z KURWAMI AUUUUUU!" << endl;
        i_greedy_counter = 0;

#pragma omp parallel for
        for (int i = 0; i < I_POP_SIZE; i++) {
            CIndividual* new_individual = new CIndividual(this);
#pragma omp critical
            v_population.push_back(new_individual);
        }

#pragma omp parallel for
        for (int i = 0; i < v_population.size(); i++) {
            vFIHC(v_population[i]);
        }

        // Sort population by fitness (higher fitness first)
        std::sort(v_population.begin(), v_population.end(), [](CIndividual* a, CIndividual* b) {
            return a->dEvaluate() > b->dEvaluate();
            });

        // Remove and deallocate the less fit half of the population
        for (size_t i = I_POP_SIZE; i < v_population.size(); i++) {
            delete v_population[i];  // Free memory for removed individuals
        }
        v_population.resize(I_POP_SIZE);

        cout << "KONIEC JAZDY Z KURWAMI!" << endl;
    }

	if (i_reset_counter == I_RESET_ITERATIONS) {
		cout << "RESET!" << endl;
		i_reset_counter = 0;

		//Delete all individuals and create new ones
		for (int i = 0; i < v_population.size(); i++) {
			delete v_population[i];
		}
		v_population.clear();

#pragma omp parallel for
		for (int i = 0; i < I_POP_SIZE; i++) {
			CIndividual* individual = new CIndividual(this);
			individual->dEvaluate();
#pragma omp critical
			v_population.push_back(individual);
		}

		cout << "RESET ZAKONCZONY!" << endl;
	}

    i_greedy_counter++;
	i_reset_counter++;
}

CIndividual* COptimizer::pcTournament()
{
	uniform_int_distribution<int> parent_dist(0, I_POP_SIZE - 1);

	int parent1_index = parent_dist(c_random_engine);
	int parent2_index = parent_dist(c_random_engine);

    // Ensure parent2 is different from parent1
    while (parent2_index == parent1_index)
    {
        parent2_index = parent_dist(c_random_engine);
    }

    double parent1_fitness = v_population[parent1_index]->dEvaluate();
    double parent2_fitness = v_population[parent2_index]->dEvaluate();

    if (parent1_fitness > parent2_fitness)
    {
        return new CIndividual(*v_population[parent1_index]);
    }
    else
    {
        return new CIndividual(*v_population[parent2_index]);
    }
}

void COptimizer::vFIHC(CIndividual* pc_individual) {
    bool b_improved = true;

    // Initialize the current fitness sum
    double dCurrentDistanceSum = pc_individual->dEvaluate();

    while (b_improved) {
        b_improved = false;

        vector<int> order = vGenerateRandomOrder();

        // Parallelize the loop over gene offsets
#pragma omp parallel for shared(b_improved, dCurrentDistanceSum)
        for (int idx = 0; idx < order.size(); idx++) {
            int gene_offset = order[idx];
            int best_fitness_gene_value = pc_individual->v_genotype[gene_offset];
            double best_fitness = dCurrentDistanceSum;

            // Copy the individual's current fitness locally for each thread
            bool local_improvement = false;

            // Loop through the possible new gene values
            for (int i = c_evaluator.iGetLowerBound(); i <= c_evaluator.iGetUpperBound(); i++) {
                // Try the new gene value using cheap evaluation
                double dNewFitnessSum = dCurrentDistanceSum;
                double dFitnessChange = 0.0;

                // Evaluate the fitness difference caused by changing the gene
                dFitnessChange = pc_better_evaluator->dCheapEvaluate(pc_individual->v_genotype.data(), gene_offset, i, dNewFitnessSum);

                // If the new fitness is better, update the best value for this gene offset
                if (dNewFitnessSum < best_fitness) {
                    best_fitness = dNewFitnessSum;
                    best_fitness_gene_value = i;
                    local_improvement = true;

                    // Update the global improvement flag atomically
#pragma omp critical
                    b_improved = true;
                }
            }

            // If improvement was found, apply the best gene value
            if (local_improvement) {
                // Update the genotype with the best gene value
                pc_individual->v_genotype[gene_offset] = best_fitness_gene_value;

                // Update the current distance sum with the new best fitness
                dCurrentDistanceSum = best_fitness;
            }
        }
    }

	// Evaluate the final individual
	pc_individual->dEvaluate();
}



vector<int> COptimizer::vGenerateRandomOrder()
{
    vector<int> order;
    for (int i = 0; i < c_evaluator.iGetNumberOfPoints(); i++) {
        order.push_back(i);
    }
    shuffle(order.begin(), order.end(), c_random_engine);
    return order;
}

CIndividual::CIndividual(COptimizer* pcParent) : pc_parent(pcParent), d_fitness(numeric_limits<double>::max()) {
    uniform_int_distribution<int> c_gene_dist(pcParent->c_evaluator.iGetLowerBound(),
        pcParent->c_evaluator.iGetUpperBound());
    v_genotype.resize(pcParent->c_evaluator.iGetNumberOfPoints());
    for (size_t i = 0; i < v_genotype.size(); i++) {
        v_genotype[i] = c_gene_dist(pcParent->c_random_engine);
    }
}

CIndividual::CIndividual(const CIndividual& individual)
    : pc_parent(individual.pc_parent),
    v_genotype(individual.v_genotype),
    d_fitness(individual.d_fitness) {
}

CIndividual& CIndividual::operator=(const CIndividual& individual) {
    if (this != &individual) {
        v_genotype = individual.v_genotype;
        d_fitness = individual.d_fitness;
        pc_parent = individual.pc_parent;
    }
    return *this;
}

double CIndividual::dEvaluate() {
    d_fitness = pc_parent->c_evaluator.dEvaluate(v_genotype);

#pragma omp critical
    {
        if (d_fitness < pc_parent->d_current_best_fitness) {
            pc_parent->d_current_best_fitness = d_fitness;
            pc_parent->v_current_best = v_genotype;
            cout << "New optimum: " << std::fixed << std::setprecision(10) << d_fitness << endl;
        }
    }

    return d_fitness;
}

void CIndividual::vUniformCrossover(CIndividual* pc_other_parent, CIndividual* pc_offspring1, CIndividual* pc_offspring2) {
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_int_distribution<int> value_dist(0, 1);

    if (prob_dist(pc_parent->c_random_engine) < D_MUT_PROB) {
        for (size_t i = 0; i < v_genotype.size(); i++) {
            pc_offspring1->v_genotype[i] = value_dist(pc_parent->c_random_engine) ? v_genotype[i] : pc_other_parent->v_genotype[i];
			pc_offspring2->v_genotype[i] = value_dist(pc_parent->c_random_engine) ? pc_other_parent->v_genotype[i] : v_genotype[i];
        }
    } else {
	    for (size_t i = 0; i < v_genotype.size(); i++) {
			pc_offspring1->v_genotype[i] = v_genotype[i];
			pc_offspring2->v_genotype[i] = pc_other_parent->v_genotype[i];
		}
    }
}

void CIndividual::vMutate() {
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_int_distribution<int> value_dist(pc_parent->c_evaluator.iGetLowerBound(), pc_parent->c_evaluator.iGetUpperBound());

    for (int& gene : v_genotype) {
        if (prob_dist(pc_parent->c_random_engine) < D_MUT_PROB) {
            gene = value_dist(pc_parent->c_random_engine);
        }
    }
}
