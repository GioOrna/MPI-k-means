#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>

#include "../include/csv_utils.hpp"
#include "../include/KDTree.hpp"
#include "../include/kmeans.hpp"
#include "../include/MPI_utils.hpp"

using namespace std;

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
	
	MPI_Finalize();
}
