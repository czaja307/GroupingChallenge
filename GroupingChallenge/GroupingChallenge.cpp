#include "GaussianGroupingEvaluatorFactory.h"
#include "GroupingEvaluator.h"
#include "Optimizer.h"
//TODO: remove
#include <iostream>
#include <chrono>
#include <windows.h>

using namespace NGroupingChallenge;

int main()
{
	auto start_time = std::chrono::high_resolution_clock::now();
	CGaussianGroupingEvaluatorFactory c_evaluator_factory(10, 1000, 1);

	c_evaluator_factory
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0);

	CGroupingEvaluator* pc_evaluator = c_evaluator_factory.pcCreateEvaluator(0);

	COptimizer c_optimizer(*pc_evaluator);

	c_optimizer.vInitialize();

	//#pragma omp parallel for
	//for (int i = 0; i < 10; i++) {
	//	Sleep(1000);  // Sleep for 1 second to simulate work
	//	cout << "dupa" << endl;	
	//}

	for (int i = 0; i < 10; i++)
	{
		cout << "Iteration: " << i << endl;
		c_optimizer.vRunIteration();
	}

	delete pc_evaluator;

	auto end_time = std::chrono::high_resolution_clock::now(); // End timing
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

	std::cout << "Program runtime: " << duration.count() << " milliseconds" << std::endl;

	return 0;
}