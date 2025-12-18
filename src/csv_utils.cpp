#include "../include/csv_utils.hpp"

#include <fstream>
#include <sstream>
#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <filesystem>
#include <cassert>

using namespace std;
namespace fs = std::filesystem;

void writeCSV(string input_file, std::vector<int> clustering, std::vector<std::vector<double>> data, std::vector<std::vector<double>> centroids){
	assert(clustering.size() == data.size() &&
		   "Clustering result size must match data size");
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
			for(size_t j = 0; j < centroids[0].size(); ++j) {
				outfile << "x"<<j<<",";
			}
			outfile <<"BelongingCluster\n";
			// Data lines
			while(getline(infile, line)) {
				// Append clustering result to the line
				if(row_index < clustering.size())
					outfile << line << "," << clustering[row_index++] << "\n";
				else{
				cout << row_index << "asd" << endl;
					assert(false && "Row index exceeds clustering size");
				}
			}
			infile.close();
			outfile.close();
			assert(row_index == clustering.size() &&
				   "All clustering results must be written to output file");

			// Write centroids on a new file
			string centroid_file_name = input_path.stem().string() + "_centroids" +
			input_path.extension().string();
			ofstream centroid_file(centroid_file_name);
			if(!centroid_file.is_open())
				throw runtime_error("Could not open centroids file");
			for(size_t j = 0; j < centroids[0].size(); ++j) {
				centroid_file << "x"<<j<<",";
			}
			centroid_file <<"CentroidID\n";
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

vector<vector<double>> readCSV(const string& filename, bool skip_header, vector<double>& max_values, vector<double>& min_values) {
	vector<vector<double>> data;
	ifstream file(filename);

	if(!file.is_open())
		throw runtime_error("Could not open file: " + filename);

	string line;
	int count;
	int previous_count = 0;
	bool first_iter = true;
	while(getline(file, line)) {
		stringstream ss(line);
		string cell;
		vector<double> row;
		count = 0;

		// Skip the header row if specified.
		if(skip_header) {
			skip_header = false;
			continue;
		}

		while(getline(ss, cell, ',')){
			row.push_back(stod(cell));
			++count;
		}

		if(first_iter) {
			previous_count = count;
			first_iter = false;
			max_values.resize(count);
			min_values.resize(count);
		} else {
			if(count != previous_count) {
				throw runtime_error("Inconsistent number of columns in row "
									+ to_string(data.size() + 1));
			}
		}
		for(size_t i = 0; i < row.size(); ++i) {
			if(row[i] > max_values[i]) {
				max_values[i] = row[i];
			}
			if(row[i] < min_values[i]) {
				min_values[i] = row[i];
			}
		}
		data.push_back(row);
	}

	file.close();
	return data;
}
