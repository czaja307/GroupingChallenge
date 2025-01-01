#include "Optimizer.h"

using namespace NGroupingChallenge;

COptimizer::COptimizer(CGroupingEvaluator& cEvaluator)
    : c_evaluator(cEvaluator) {
    random_device c_seed_generator;
    c_random_engine.seed(c_seed_generator());

	i_greedy_counter = 0;
}

COptimizer::~COptimizer()
{
	for (int i = 0; i < v_population.size(); i++) {
		delete v_population[i];
	}
	v_population.clear();
}

void COptimizer::vInitialize() {
    d_current_best_fitness = std::numeric_limits<double>::max();

    v_current_best.clear();
    v_current_best.resize(c_evaluator.iGetNumberOfPoints());

    //TODO: get rid of that param
    v_population.clear();
    for (int i = 0; i < POP_SIZE; i++) {
        // TODO: pushback vs emplace
        v_population.push_back(new CIndividual(this));
        v_population.back()->dEvaluate();
    }
}

void COptimizer::vRunIteration() {
    vector<CIndividual*> v_new_population;

    uniform_int_distribution<int> individual_dist(0, POP_SIZE - 1);

    while (v_new_population.size() < POP_SIZE) {
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

		v_new_population.push_back(child1);
		v_new_population.push_back(child2);
	}

	for (int i = 0; i < v_population.size(); i++) {
		delete v_population[i];
	}
    v_population.swap(v_new_population);

	while (v_population.size() > POP_SIZE) {
		delete v_population.back();
		v_population.pop_back();
	}

	for (int i = 0; i < v_population.size(); i++) {
		v_population[i]->dEvaluate();
	}

    if (i_greedy_counter == GREEDY_ITERATIONS) {
        cout << "JAZDA Z KURWAMI AUUUUUU!" << endl;
		i_greedy_counter = 0;
		shuffle(v_population.begin(), v_population.end(), c_random_engine);

        for (int i = 0; i < GREEDY_INDIVIDUALS; i++) {
			vFIHC(v_population[i]);
        }

		cout << "KONIEC JAZDY Z KURWAMI!" << endl;
    }

    i_greedy_counter++; 
}

CIndividual* COptimizer::pcTournament()
{
	uniform_int_distribution<int> parent_dist(0, POP_SIZE - 1);

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

    while (b_improved) {
        b_improved = false;

        for (int gene_offset : vGenerateRandomOrder()) {
            double best_fitness = pc_individual->dEvaluate();
            int best_fitness_gene_value = pc_individual->v_genotype[gene_offset];
            for (int i = c_evaluator.iGetLowerBound(); i <= c_evaluator.iGetUpperBound(); i++) {
                pc_individual->v_genotype[gene_offset] = i;
                double current_fitness = pc_individual->dEvaluate();
                if (current_fitness < best_fitness) {
                    best_fitness_gene_value = i;
                    b_improved = true;
                }
            }
            pc_individual->v_genotype[gene_offset] = best_fitness_gene_value;
        }
    }
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

// void COptimizer::vRunIteration() {
//     vector<int> v_candidate(c_evaluator.iGetNumberOfPoints());
//
//     uniform_int_distribution<int> c_candidate_distribution(c_evaluator.iGetLowerBound(), c_evaluator.iGetUpperBound());
//
//     for (size_t i = 0; i < v_candidate.size(); i++) {
//         v_candidate[i] = c_candidate_distribution(c_random_engine);
//     }
//
//     double d_candidate_fitness = c_evaluator.dEvaluate(v_candidate);
//
//     if (d_candidate_fitness < d_current_best_fitness) {
//         v_current_best = v_candidate;
//         d_current_best_fitness = d_candidate_fitness;
//     }
//
//     cout << d_current_best_fitness << endl;
// }

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
    if (d_fitness < pc_parent->d_current_best_fitness) {
        pc_parent->d_current_best_fitness = d_fitness;
        pc_parent->v_current_best = v_genotype;
        cout << "New optimum: " << d_fitness << endl;
    }
    return d_fitness;
}

void CIndividual::vUniformCrossover(CIndividual* pc_other_parent, CIndividual* pc_offspring1, CIndividual* pc_offspring2) {
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_int_distribution<int> value_dist(0, 1);

    if (prob_dist(pc_parent->c_random_engine) < MUT_PROB) {
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
        if (prob_dist(pc_parent->c_random_engine) < MUT_PROB) {
            gene = value_dist(pc_parent->c_random_engine);
        }
    }
}
