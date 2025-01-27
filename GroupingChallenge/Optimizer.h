#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "GroupingEvaluator.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <iomanip>
#include "BetterEvaluator.h"

using namespace std;

const int POP_SIZE = 100;
const double CROSS_PROB = 0.8;
const double MUT_PROB = 0.05;
const int GREEDY_INDIVIDUALS = 1.0 * POP_SIZE;
const int GREEDY_ITERATIONS = 10;


namespace NGroupingChallenge {
    class COptimizer {
        friend class CIndividual;

    public:
        COptimizer(CGroupingEvaluator& cEvaluator);
        ~COptimizer();

        void vInitialize();
        void vRunIteration();

        vector<int>* pvGetCurrentBest() { return &v_current_best; }

    private:
        CGroupingEvaluator& c_evaluator;
		CBetterEvaluator* c_better_evaluator;
        mt19937 c_random_engine;

        double d_current_best_fitness;
        vector<int> v_current_best;

        vector<CIndividual*> v_population;

		int i_greedy_counter;

		CIndividual* pcTournament();
        void vFIHC(CIndividual* pc_individual);
		vector<int> vGenerateRandomOrder();
    };

    class CIndividual {
		friend class COptimizer;

    public:
        CIndividual(COptimizer* pcParent);
        CIndividual(const CIndividual& other);

        CIndividual& operator=(const CIndividual& other);

        double dEvaluate();
        double dCheapEvaluate(int iChangedGene, int iNewValue);

        void vUniformCrossover(CIndividual* pc_other_parent, CIndividual* pc_offspring1, CIndividual* pc_offspring2);
        void vMutate();

    private:
        COptimizer* pc_parent;
        vector<int> v_genotype;
        double d_fitness;
    };
}

#endif
