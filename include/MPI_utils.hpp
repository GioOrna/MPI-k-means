#ifndef MPI_UTILS_HPP
#define MPI_UTILS_HPP
#include <mpi.h>
#include <vector>

using namespace std;

/**
 * @brief Macro to execute a block of code only on a specific MPI rank.
 * @param x The rank on which to execute the code block.
 * \internal
 * Similar to OpenMP's single directive.
 */
#define MPI_single(x) if(rank == x)

/**
 * @brief Macro to execute a block of code only on the master MPI rank (rank 0).
 * \internal
 * Similar to OpenMP's master directive.
 */
#define MPI_master() MPI_single(0)

/**
 * @brief Evenly scatter a 2D vector of doubles among all MPI processes.
 * @param data The 2D vector to scatter (only needed on master process).
 * @param mpi_comm The MPI communicator.
 * @return The portion of the data assigned to the calling process.
 * @detail
 * The input parameter 'data' is only required on the master process and the
 * resulting scattered data is returned to all processes.
 */
vector<vector<double>> MPI_evenlyScatterData(const vector<vector<double>>&,
										 MPI_Comm);

/**
 * @brief Gathers a vector of integers from all MPI processes, handling
 * unbalanced sizes.
 * @param local_clustering The local vector of integers from each process.
 * @param comm The MPI communicator.
 * @return The gathered vector of integers.
 * @detail
 * Only the master process will receive the complete gathered vector; other
 * processes will receive an empty vector.
 */
vector<int> MPI_gatherUnbalancedData(const vector<int>&, MPI_Comm);

/**
 * @brief Computes the clustering employing all MPI processes.
 * @param data The whole dataset.
 * @param centroids The centroids on which to base the clustering.
 * @return The clustering result as a vector of cluster indices.
 * @detail
 * The resulting vector has the same length as the number of data points and
 * contains the index of the assigned centroid for each data point. Only the
 * master process will have the complete clustering result; as well as the input
 * parameters are expected to be valid only on the master process.
 */
vector<int> MPI_computeClustering(const vector<vector<double>>&,
								  const vector<vector<double>>&);

/**
 * @brief Flattens a 2D vector into a 1D vector.
 * @param mat The 2D vector to flatten.
 * @return The flattened 1D vector.
 */
vector<double> flatten(const vector<vector<double>>& mat);

/**
 * @brief Unflattens a 1D vector into a 2D vector.
 * @param flat The 1D vector to unflatten.
 * @param rows Number of rows in the resulting 2D vector.
 * @param cols Number of columns in the resulting 2D vector.
 * @return The unflattened 2D vector.
 */
vector<vector<double>> unflatten(const vector<double>& flat, int rows, int cols);

/**
 * @brief Broadcasts centroids from the sender rank to all other ranks.
 * @param centroids The centroids to broadcast.
 * @param rank The rank of the calling process.
 * @param sender_rank The rank that holds the centroids to broadcast.
 * @param mpi_comm The MPI communicator.
 * @return void
 */
void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank);
#endif // MPI_UTILS_HPP

void gather_results(vector<vector<double>>& wgtCent, vector<int>& counts, int to_rank, int rank, int size);

void send_result(vector<vector<double>>& wgtCent, vector<int> counts, int to_rank);

void receive_result(vector<vector<double>>& wgtCent, vector<int>& counts, int sender_rank);