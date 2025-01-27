#pragma once
#include "GroupingEvaluator.h"

namespace NGroupingChallenge
{
	class CBetterEvaluator :
		public CGroupingEvaluator
	{
	public:
		CBetterEvaluator(int iNumberOfGroups, const std::vector<CPoint>& vPoints);

		double dCheapEvaluate(const int* piSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const;
		double dCheapEvaluate(const vector<int>* pvSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const;
		double dCheapEvaluate(const vector<int>& vSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const;
	private:
		const double d_WRONG_VALUE = -1;
	};
}

