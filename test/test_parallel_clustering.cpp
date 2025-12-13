#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cassert>

#include "../include/MPI_utils.hpp"
#include "../include/csv_utils.hpp"
#include "../include/kmeans.hpp"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	vector<vector<double>> data;
	vector<vector<double>> centroids;
	MPI_master() {
		if(argc != 3) {
			cerr << "Usage: " << argv[0] << " <path_to_file>"
				 << " <number_of_centroids>" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		int num_centroids;
		try {
			num_centroids = stoul(argv[2]);
			if(num_centroids < 1)
				throw invalid_argument("Number of centroids must be at least 1");
		} catch (const exception& e) {
			cerr << "Invalid number of centroids: " << argv[2] << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		vector<double> max_values;
		vector<double> min_values;
		try {
			data = readCSV(argv[1], true, max_values, min_values);
		} catch(const exception& e) {
			cerr << "Error reading CSV file: " << e.what() << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		centroids = generate_centroids(num_centroids, data[0].size(),
									   max_values, min_values);
	}

	// Compute clustering in parallel
	vector<int> clustering = MPI_computeClustering(data, centroids);

	assert(clustering.size() == data.size() &&
		   "Clustering result size must match data size");

	// Copy input file and append clustering results
	MPI_master() {
		string input_file = argv[1];
		fs::path input_path(input_file);
		string output_file = input_path.stem().string() + "_cluster" +
							 input_path.extension().string();

		try {
			// Open input and output files.
			ifstream infile(input_file);
			if(!infile.is_open())
				throw runtime_error("Could not open input file for reading");
			ofstream outfile(output_file);
			if(!outfile.is_open())
				throw runtime_error("Could not open output file for writing");

			// Writing the file
			string line;
			size_t row_index = 0;
			// Header line
			getline(infile, line);
			outfile << "x0,x1,x2,BelongingCluster\n";
			// Data lines
			while(getline(infile, line)) {
				// Append clustering result to the line
				if(row_index < clustering.size())
					outfile << line << "," << clustering[row_index++] << "\n";
				else
					assert(false && "Row index exceeds clustering size");
			}
			infile.close();
			outfile.close();
			assert(row_index == clustering.size() &&
				   "All clustering results must be written to output file");

			// Write centroids on a new file
			ofstream centroid_file("output_centroids.csv");
			if(!centroid_file.is_open())
				throw runtime_error("Could not open centroids file");
			centroid_file << "x0,x1,x2,CentroidID\n";
			for(size_t i = 0; i < centroids.size(); ++i) {
				for(size_t j = 0; j < centroids[i].size(); ++j) {
					centroid_file << centroids[i][j];
					if(j < centroids[i].size() - 1)
						centroid_file << ",";
				}
				centroid_file << "," << i << "\n";
			}
			centroid_file.close();
		} catch(const exception& e) {
			cerr << "Error writing output CSV file: " << e.what() << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		cout << "Clustering results written to " << output_file << endl;
		cout << "Notice: the result won't be a perfect clustering, but it gives"
			 << " a rough idea if the clustering of the row data points works "
			 << "properly or not." << endl;
	}

	MPI_Finalize();
}
