#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>

#include "../include/csv_utils.hpp"
#include "../include/KDTree.hpp"

using namespace std;

/**
 * @brief Generates random centroids within given bounds.
 * @param num_centroids Number of centroids to generate.
 * @param dim Data dimension.
 * @param max_values Maximum values for each dimension.
 * @param min_values Minimum values for each dimension.
 * @return A vector containing the generated centroids.
 */
vector<vector<double>> generate_centroids(const int num_centroids,
										 const size_t dim,
										 const vector<double>& max_values,
										 const vector<double>& min_values);

/**
 * @brief Compute the closest centroid to the point.
 * @param point The point to be analyzed.
 * @param centroids The centroids to be analyzed.
 * @param candidates List of indexes to check in the centroids vector.
 * @return The index of the closest centroid.
 */
int closest_centroid(const vector<double>& point,
					 const vector<vector<double>>& centroids, vector<int> candidates);

vector<vector<double>> generate_centroids_plus_plus(
    const int num_centroids,
    const size_t dim,
    const vector<vector<double>>& data);

/**
 * @brief Tells if the node is farther from centroid in position c2 than the one in position c1
 * @param node The node to consider.
 * @param centroids The vector of centroids.
 * @param c1 The index in the centroids vector of first centroid to consider.
 * @param c2 The index in the centroids vector of second centroid to consider.
 */
bool is_farther(const KDTree::Node* node,
				const vector<vector<double>>& centroids,
				const int c1,
				const int c2);

/**
 * @brief An algorithm to calculate new centroids coordinated based on KD-tree.
 * @param u root of the subtree we want to analyze.
 * @param candidates Vector of possible clusters to assign to the subtree.
 * @param ccentroids 2D vector of all centroids.
 * @param wgtCent 2D vector of sum of coordinates of points assigned to each centroid.
 * @param counts Vector of number of points assigned to each centroid.
 * @param sum_internal_it Returns the number of recursive call done inside the filter function.
 * @return void
 */
void filter(const KDTree::Node* u, vector<int> candidates, vector<vector<double>>& centroids, 
		    vector<vector<double>>& wgtCent, vector<int>& counts, int& sum_internal_it);
		
/**
 * @brief Calculate squared Euclidean distance between two points.
 * @param a First point.
 * @param b Second point.
 * @return Squared Euclidean distance.
 */
double distance(vector<double> a, vector<double> b);

/**
 * @brief Sequential K-means algorithm, will update centroids with the new calculated centroids
 * @param centroids Centroids we want to update.
 * @param data_to_work The data to work on. 
 */
void kmeans_sequential(vector<vector<double>>& centroids,
			std::vector<std::vector<double>> data_to_work, int& sum_internal_it);

/**
 * @brief Parallel K-means algorithm, will update centroids with the new calculated centroids
 * @param rank Rank of the process entering the function.
 * @param size Total number of processes entering the function.
 * @param centroids Centroids we want to update.
 * @param data_to_work The data to work on. 
 */
vector<vector<double>> kmeans_parallel(int rank, int size, vector<vector<double>>& centroids,
			std::vector<std::vector<double>> data_to_work, int& iterations, int& sum_internal_it);