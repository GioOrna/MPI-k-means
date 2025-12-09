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

										 
vector<vector<double>> generate_centroids(const int num_centroids,
										 const size_t dim,
										 const vector<double>& max_values,
										 const vector<double>& min_values);

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	vector<vector<double>> data;
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
		KDTree tree(data); // generate KDTree from data
		vector<vector<double>> centroids = generate_centroids(
			num_centroids, data[0].size(), max_values, min_values);
		
		int l =0;
		for (auto& centroid : centroids){
			cout << "centroid:" << l << endl;
			++l;
			for(size_t i=0; i < centroid.size(); ++i){
				cout << "Centroid dim " << i << ": " << centroid[i] << endl;
			}
		}
		
	}
	MPI_Barrier(mpi_comm); // synchronize all ranks


	// Scatter the data evenly among all processes.
	//vector<vector<double>> local_data = MPI_evenlyScatterData(data, mpi_comm);

	// All processes can now work on their portion of the data
	
	MPI_Finalize();
}

vector<vector<double>> generate_centroids(const int num_centroids,
										 const size_t dim,
										 const vector<double>& max_values,
										 const vector<double>& min_values) {
	random_device rd;                // non-deterministic seed
    mt19937 gen(rd());               // Mersenne Twister engine
    vector<uniform_real_distribution<double>> dist;  // ranges
	dist.resize(dim);
	for(size_t i = 0; i < dim; ++i) {  // for each dimension define max and min
		dist[i] = std::uniform_real_distribution<double>(min_values[i], max_values[i]);
	}
	vector<vector<double>> centroids;
	centroids.resize(num_centroids); 
	for(size_t i = 0; i < num_centroids; ++i) { // for each centroid
		centroids[i].resize(dim);
		for(size_t j = 0; j < dim; ++j){  // for each dimension
			centroids[i][j] = dist[j](gen);
		}
	}
	return centroids;
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
