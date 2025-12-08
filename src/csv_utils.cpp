#include "../include/csv_utils.hpp"

#include <fstream>
#include <sstream>

using namespace std;

vector<vector<double>> readCSV(const string& filename, bool skip_header) {
	vector<vector<double>> data;
	ifstream file(filename);

	if(!file.is_open())
		throw runtime_error("Could not open file: " + filename);

	string line;

	while(getline(file, line)) {
		stringstream ss(line);
		string cell;
		vector<double> row;

		// Skip the header row if specified.
		if(skip_header) {
			skip_header = false;
			continue;
		}

		while(getline(ss, cell, ','))
			row.push_back(stod(cell));

		data.push_back(row);
	}

	file.close();
	return data;
}
