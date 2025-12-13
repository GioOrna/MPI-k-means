#include "../include/csv_utils.hpp"

#include <fstream>
#include <sstream>

using namespace std;

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
