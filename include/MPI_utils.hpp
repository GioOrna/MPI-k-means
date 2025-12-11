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
 */
vector<vector<double>> MPI_evenlyScatterData(const vector<vector<double>>&,
										 MPI_Comm);

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
void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank, MPI_Comm mpi_comm);
