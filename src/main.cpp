#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <filesystem>
#include <cassert>
#include "../include/config.hpp"
#ifdef test
	#include <chrono>
#endif
#include <algorithm>

#include "../include/csv_utils.hpp"
#include "../include/kmeans.hpp"
#include "../include/KDTree.hpp"
#include "../include/MPI_utils.hpp"


using namespace std;

void shuffleRows(std::vector<std::vector<double>>& data) {
	#ifdef not_random
		mt19937 gen(42);
	#else
		random_device rd;
    	mt19937 gen(rd());   // Mersenne Twister RNG
	#endif
    shuffle(data.begin(), data.end(), gen);
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);
		
	#ifdef test
		auto start = std::chrono::high_resolution_clock::now();
		auto start_total = std::chrono::high_resolution_clock::now();
	#endif
	
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

		vector<double> max_values;
		vector<double> min_values;
		try {
			data = readCSV(argv[1], true, max_values, min_values);
		} catch(const exception& e) {
			cerr << "Error reading CSV file: " << e.what() << endl;
			MPI_Abort(mpi_comm, 1);
		}		
		centroids = generate_centroids_plus_plus(
		num_centroids, data[0].size(), data);	
	
	}
	shuffleRows(data); //shuffles data to better distribute it
	vector<vector<double>> data_to_work = MPI_evenlyScatterData(data, mpi_comm);
	#ifdef test
		if(rank==0){
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			std::cout << "Elapsed time Sequential: " << elapsed.count() << " seconds\n";
			start = std::chrono::high_resolution_clock::now();
		}
	#endif
	int it=0;
	int sum_internal_it=0;
	centroids = kmeans_parallel(rank, size, centroids, data_to_work, it, sum_internal_it);
	#ifdef test
		if(rank==0){
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			std::cout << "Elapsed time Parallel: " << elapsed.count() << " seconds\n";
			start = std::chrono::high_resolution_clock::now();
			cout << "iterations: " << it << endl;
			auto end_total = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_total = end_total - start_total;
			std::cout << "Elapsed time Total: " << elapsed_total.count() << " seconds\n";
		}
		cout << "iterations of filter algorithm on rank: "<<rank <<" : " << sum_internal_it << endl;
	#endif
	broadcast_centroids(centroids, rank, 0);
	vector<int> clustering = MPI_computeClustering(data, centroids);
	MPI_master() {
		writeCSV(argv[1], clustering, data, centroids);
	}

	MPI_Finalize();
}
