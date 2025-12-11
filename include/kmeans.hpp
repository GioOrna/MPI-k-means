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
 * @return The index of the closest centroid.
 */
int closest_centroid(const vector<double>& point,
					 const vector<vector<double>>& centroids);

/**
 * @brief Puts the cluster field of all nodes in the specified subtree equal to the specified value.
 * @param node The root of the subtree.
 * @param cluster The index of the cluster we want to assign.
 * @return void
 */
void assign_to_cluster(const KDTree::Node* node, const int cluster);

/**
 * @brief Calculates the midpoint of the subtree specified.
 * @param node The root of the subtree.
 * @return A vector of double containing the mipoint coordinates
 */
vector<double> midpoint(const KDTree::Node* node);

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
 * @return void
 */
void filter(const KDTree::Node* u, vector<int> candidates, vector<vector<double>>& centroids, 
		    vector<vector<double>>& wgtCent, vector<int>& counts);

/**
 * @brief Calculate squared Euclidean distance between two points.
 * @param a First point.
 * @param b Second point.
 * @return Squared Euclidean distance.
 */
double distance(vector<double> a, vector<double> b);

/**
 * @brief Sums coordinates of all nodes in the subtree and keeps count of how many have been summed.
 * @param wgtCent Vector to which coordinates will be summed (it won't be resetted).
 * @param counts Int where the number of nodes will be summed (it won't be resetted).
 * @param node Root of the subtree.
 * @return void
 */
void assign_subtree_to_cluster(vector<double>& wgtCent, int& counts,const KDTree::Node* node);

/**
 * @brief Adds the centroids as points in the tree with new values in the cluster field to distinguish them from the other points.
 * @param tree The tree where we want to add the centroids
 * @param centroids Centroids we want to add.
 * @return void
 */
void insert_centroid_in_tree(KDTree& tree, vector<vector<double>>& centroids);

/**
 * @brief Sequential K-means algorithm, will update centroids with the new calculated centroids
 * @param centroids Centroids we want to update.
 * @param node The root of the tree we want to use. 
 */
void Kmeans_sequential(vector<vector<double>>& centroids,
			const KDTree::Node* node);
