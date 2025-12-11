#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>

#include "../include/csv_utils.hpp"
#include "../include/KDTree.hpp"
#include "../include/kmeans.hpp"

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

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	vector<vector<double>> data;
	vector<vector<double>> centroids;
	MPI_master() {
		if(argc != 3) {
			cerr << "Usage: " << argv[0] << " <path_to_file> <number_of_centroids>" << endl;
			MPI_Abort(mpi_comm, 1);
		}
		int num_centroids;
		try {
			num_centroids = std::stoul(argv[2]);
			if(num_centroids < 1){
				throw std::invalid_argument("Number of centroids must be at least 1.");
			}
		} catch(const std::exception& e) {
			cerr << "Invalid number of centroids: " << argv[2] << endl;
			MPI_Abort(mpi_comm, 1);
		}

		// File loading and parsing.
		// It isn't worth to concurrently load and parse the file; the required
		// operations for each process would be:
		// - open and read the file (a csv file is not splittable a priori), so
		//   we would lose the advantage of parallelism.
		// - parse the data
		// - send/receive the information about what slice of data each process
		//   should work on.
		// Finally, it's easier for code readability and ease of coding to just
		// do it on the master process and then scatter the data to the other
		// processes.
		vector<double> max_values;
		vector<double> min_values;
		try {
			data = readCSV(argv[1], true, max_values, min_values);
		} catch(const exception& e) {
			cerr << "Error reading CSV file: " << e.what() << endl;
			MPI_Abort(mpi_comm, 1);
		}
			centroids = generate_centroids(
			num_centroids, data[0].size(), max_values, min_values);		
	}
	MPI_Barrier(mpi_comm); // synchronize all ranks
	broadcast_centroids(centroids, rank, 0, mpi_comm); 	// Broadcast centroids to all processes.
	
	//rank0 iterate over points in the tree and check closest centroid
	//if two points have different closest centroids, split the tree and send to the other ranks
	//(if there are ranks available)
	//if all points have the same closest centroid, send back summation andd count to rank0
	//recurseively do the same on each subtree until no ranks are available or no more splits are possible
	if(rank == 0){
		KDTree tree(data);
		const KDTree::Node* root = tree.getRoot();
		Kmeans_sequential(centroids, root);
		insert_centroid_in_tree(tree, centroids); // to test the output file
		tree.writeCSV("output_cluster.csv");
	}
	MPI_Barrier(mpi_comm); // synchronize all ranks
	
	// Scatter the data evenly among all processes.
	//vector<vector<double>> local_data = MPI_evenlyScatterData(data, mpi_comm);

	// All processes can now work on their portion of the data
	
	MPI_Finalize();
}

void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank, MPI_Comm mpi_comm) {
	// Broadcast centroids to all processes.
	vector<double> flat_centroids;
	int rows, cols;
	if(rank==sender_rank){
		flat_centroids = flatten(centroids);
		rows = centroids.size();
		cols = centroids[0].size();
	}
	MPI_Bcast(&rows, 1, MPI_INT, sender_rank, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, sender_rank, MPI_COMM_WORLD);
	if(rank!=sender_rank){
		flat_centroids.resize(rows * cols);
	}
    MPI_Bcast(flat_centroids.data(), flat_centroids.size(), MPI_DOUBLE, sender_rank, MPI_COMM_WORLD);
	centroids = unflatten(flat_centroids, rows, cols);
	return;
}

// Flatten a 2D vector into a 1D vector
vector<double> flatten(const vector<vector<double>>& mat) {
    vector<double> flat;
    for (const auto& row : mat) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Unflatten a 1D vector into a 2D vector
vector<vector<double>> unflatten(const vector<double>& flat, int rows, int cols) {
    if (flat.size() != rows * cols) {
        throw invalid_argument("Size of flat vector does not match rows*cols");
    }
    vector<vector<double>> mat(rows, vector<double>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = flat[i * cols + j];
    return mat;
}

vector<vector<double>> MPI_evenlyScatterData(const vector<vector<double>>& data,
										 MPI_Comm mpi_comm) {
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	int total_cols, per_process_rows, remainder;
	vector<double> flat_data;

	MPI_master() {
		int total_rows = data.size();
		total_cols = data[0].size();

		// Calculate the number of rows per process.
		per_process_rows = total_rows /size;
		remainder = total_rows % per_process_rows;

		// Flatten the data.
		flat_data.reserve(data.size() * total_cols);
		for(const auto& row : data)
			flat_data.insert(flat_data.end(), row.begin(), row.end());
	}

	// Send to all processes the data dimensions.
	MPI_Bcast(&per_process_rows, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&remainder, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&total_cols, 1, MPI_INT, 0, mpi_comm);

	// Buffer send and displacements.
	// Note: the master process gets more rows beacause it's a fake comunication
	// the one between master and itself (especially in case of clusters), so it
	// should be faster than the others (this is a naive optimization).
	int send_counts[size], displacements[size];

	for(int i = 0; i < size; ++i) {
		send_counts[i] = per_process_rows * total_cols;
		displacements[i] = i * per_process_rows * total_cols;

		// Master gets the remainder.
		if(i == 0)
			send_counts[i] += remainder * total_cols;
		else
			displacements[i] += remainder * total_cols;
	}

	MPI_master() {
		per_process_rows += remainder;
	}

	// Each process allocates its local data.
	vector<double> local_data(per_process_rows * total_cols);

	MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr,
				 send_counts,
				 displacements,
				 MPI_DOUBLE,
				 local_data.data(),
				 per_process_rows * total_cols,
				 MPI_DOUBLE,
				 0,
				 mpi_comm);

	// Reshape the local data into a 2D vector.
	vector<vector<double>> reshaped(per_process_rows,
									vector<double>(total_cols));
	for(int i = 0; i < per_process_rows; ++i)
		copy(local_data.begin()+i*total_cols,
			 local_data.begin()+(i+1)*total_cols,
			 reshaped[i].begin());

	return reshaped;
}
