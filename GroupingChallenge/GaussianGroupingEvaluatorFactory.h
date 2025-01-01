#ifndef GAUSSIAN_GROUPING_EVALUATOR_FACTORY_H
#define GAUSSIAN_GROUPING_EVALUATOR_FACTORY_H

#include "GroupingEvaluator.h"
#include "Point.h"

#include <random>
#include <vector>

using namespace std;

namespace NGroupingChallenge
{
	class CDimension
	{
	public:
		CDimension(double dMeanMin, double dMeanMax, double dStandardDeviationMin, double dStandardDeviationMax, mt19937& cRandomEngine);

		double dGenerateRandomMean();
		double dGenerateRandomStandardDeviation();

	private:
		const double d_DEFAULT_STANDARD_DEVIATION_VALUE = 1.0;

		uniform_real_distribution<double> c_mean_uniform_distribution;
		uniform_real_distribution<double> c_standard_deviation_uniform_distribution;
		mt19937& c_random_engine;
	};

	class CGaussianDistribution
	{
	public:
		CGaussianDistribution(double dMean, double dStandardDeviation, mt19937& cRandomEngine);

		double dGenerateRandomNumber();

	private:
		normal_distribution<double> c_normal_distribution;
		mt19937& c_random_engine;
	};

	class CGaussianGroupingEvaluatorFactory
	{
	public:
		CGaussianGroupingEvaluatorFactory(int iNumberOfGroups, int iNumberOfPoints);

		CGaussianGroupingEvaluatorFactory& cAddDimension(double dMeanMin, double dMeanMax, double dStandardDeviationMin, double dStandardDeviationMax);

		CGroupingEvaluator* pcCreateEvaluator();
		CGroupingEvaluator* pcCreateEvaluator(unsigned int iSeed);

	private:
		const int i_NUMBER_OF_GROUPS_MIN_VALUE = 1;
		const int i_NUMBER_OF_POINTS_MIN_VALUE = 1;

		vector<CGaussianDistribution>* pv_create_gaussian_distributions();
		vector<CPoint>* pv_create_points(vector<CGaussianDistribution>& vGaussianDistributions);

		int i_number_of_groups;
		int i_number_of_points;

		vector<CDimension> v_dimensions;
		mt19937 c_random_engine;
	};
}

#endif