#include "GaussianGroupingEvaluatorFactory.h"

using namespace NGroupingChallenge;

CDimension::CDimension(double dMeanMin, double dMeanMax, double dStandardDeviationMin, double dStandardDeviationMax, mt19937 &cRandomEngine)
	: c_mean_uniform_distribution(min(dMeanMin, dMeanMax), max(dMeanMin, dMeanMax)), 
	c_standard_deviation_uniform_distribution(min(dStandardDeviationMin, dStandardDeviationMax), max(dStandardDeviationMin, dStandardDeviationMax)),
	c_random_engine(cRandomEngine)
{

}

double CDimension::dGenerateRandomMean()
{
	return c_mean_uniform_distribution(c_random_engine);
}

double CDimension::dGenerateRandomStandardDeviation()
{
	double d_standard_deviation = c_standard_deviation_uniform_distribution(c_random_engine);

	if (d_standard_deviation <= 0)
	{
		d_standard_deviation = d_DEFAULT_STANDARD_DEVIATION_VALUE;
	}

	return d_standard_deviation;
}

CGaussianDistribution::CGaussianDistribution(double dMean, double dStandardDeviation, mt19937& cRandomEngine)
	: c_normal_distribution(dMean, dStandardDeviation), c_random_engine(cRandomEngine)
{

}

double CGaussianDistribution::dGenerateRandomNumber()
{
	return c_normal_distribution(c_random_engine);
}

CGaussianGroupingEvaluatorFactory::CGaussianGroupingEvaluatorFactory(int iNumberOfGroups, int iNumberOfPoints)
	: i_number_of_groups(max(iNumberOfGroups, i_NUMBER_OF_GROUPS_MIN_VALUE)), i_number_of_points(max(iNumberOfPoints, i_NUMBER_OF_GROUPS_MIN_VALUE))
{

}

CGaussianGroupingEvaluatorFactory& CGaussianGroupingEvaluatorFactory::cAddDimension(double dMeanMin, double dMeanMax, double dStandardDeviationMin, double dStandardDeviationMax)
{
	v_dimensions.push_back(CDimension(dMeanMin, dMeanMax, dStandardDeviationMin, dStandardDeviationMax, c_random_engine));

	return *this;
}

CGroupingEvaluator* CGaussianGroupingEvaluatorFactory::pcCreateEvaluator()
{
	random_device c_seed_generator;

	return pcCreateEvaluator(c_seed_generator());
}

CGroupingEvaluator* CGaussianGroupingEvaluatorFactory::pcCreateEvaluator(unsigned int iSeed)
{
	c_random_engine.seed(iSeed);

	vector<CGaussianDistribution>* pv_gaussian_distributions = pv_create_gaussian_distributions();
	vector<CPoint>* pv_points = pv_create_points(*pv_gaussian_distributions);

	CGroupingEvaluator* pc_evaluator = new CGroupingEvaluator(i_number_of_groups, *pv_points);

	delete pv_gaussian_distributions;
	delete pv_points;

	return pc_evaluator;
}


vector<CGaussianDistribution>* CGaussianGroupingEvaluatorFactory::pv_create_gaussian_distributions()
{
	vector<CGaussianDistribution>* pv_gaussian_distributions = new vector<CGaussianDistribution>();

	pv_gaussian_distributions->reserve(v_dimensions.size());

	for (size_t i = 0; i < v_dimensions.size(); i++)
	{
		pv_gaussian_distributions->push_back(CGaussianDistribution(v_dimensions[i].dGenerateRandomMean(), v_dimensions[i].dGenerateRandomStandardDeviation(), c_random_engine));
	}

	return pv_gaussian_distributions;
}

vector<CPoint>* CGaussianGroupingEvaluatorFactory::pv_create_points(vector<CGaussianDistribution>& vGaussianDistributions)
{
	vector<CPoint>* pv_points = new vector<CPoint>();

	pv_points->reserve(i_number_of_points);

	for (int i = 0; i < i_number_of_points; i++)
	{
		CPoint c_point;

		for (size_t j = 0; j < vGaussianDistributions.size(); j++)
		{
			c_point.vAddCoordinate(vGaussianDistributions[j].dGenerateRandomNumber());
		}

		pv_points->push_back(c_point);
	}

	return pv_points;
}