#include "BetterEvaluator.h"

using namespace NGroupingChallenge;

CBetterEvaluator::CBetterEvaluator(int iNumberOfGroups, const std::vector<CPoint>& vPoints)
    : CGroupingEvaluator(iNumberOfGroups, vPoints)
{
}

double CBetterEvaluator::dCheapEvaluate(const int* piSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const
{
    const vector<CPoint>& v_points = vGetPoints();

    if (!piSolution || v_points.empty()) {
        return d_WRONG_VALUE;
    }

    double d_difference = 0.0;
    bool b_error = false;
    int iOldValue = piSolution[iChangedGene];

#pragma omp parallel
    {
        double d_private_difference = 0.0;
        bool b_private_error = false;

        // Remove the old contribution of the changed gene
#pragma omp for
        for (int j = 0; j < v_points.size(); j++) {
            if (j != iChangedGene && piSolution[j] == iOldValue) { // Old group
                double d_distance = v_points[iChangedGene].dCalculateDistance(v_points[j]);
                if (d_distance >= 0) {
                    d_private_difference -= 2.0 * d_distance;
                }
                else {
                    b_private_error = true;
                }
            }
        }

        // Add the new contribution of the changed gene
#pragma omp for
        for (int j = 0; j < v_points.size(); j++) {
            if (j != iChangedGene && piSolution[j] == iNewValue) { // New group
                double d_distance = v_points[iChangedGene].dCalculateDistance(v_points[j]);
                if (d_distance >= 0) {
                    d_private_difference += 2.0 * d_distance;
                }
                else {
                    b_private_error = true;
                }
            }
        }

        // Combine results from all threads
#pragma omp critical
        {
            d_difference += d_private_difference;
            b_error = b_error || b_private_error;
        }
    }

    if (b_error) {
        return d_WRONG_VALUE;
    }

    // Update the total distance sum
    dCurrentDistanceSum += d_difference;

    return dCurrentDistanceSum;
}

double CBetterEvaluator::dCheapEvaluate(const vector<int>* pvSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const
{
	if (!pvSolution) {
		return d_WRONG_VALUE;
	}

	return dCheapEvaluate(*pvSolution, iChangedGene, iNewValue, dCurrentDistanceSum);
}

double CBetterEvaluator::dCheapEvaluate(const vector<int>& vSolution, int iChangedGene, int iNewValue, double& dCurrentDistanceSum) const
{
    const vector<CPoint>& v_points = vGetPoints();

	if (vSolution.size() != v_points.size()) {
		return d_WRONG_VALUE;
	}

	return dCheapEvaluate(vSolution.data(), iChangedGene, iNewValue, dCurrentDistanceSum);
}
