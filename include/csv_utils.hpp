#ifndef CSV_UTILS_HPP
#define CSV_UTILS_HPP

#include <string>
#include <vector>
#include <array>

/**
 * @brief Write a CSV file of the clustered data and a CSV file of the centroids.
 * @param input_file The initial part of the name we want to use.
 * @param clustering The clustering of the data.
 * @param data The data we want to write.
 * @param centroids The centroids we want to write.
 */
void writeCSV(std::string input_file, std::vector<int> clustering, std::vector<std::vector<double>> data, std::vector<std::vector<double>> centroids);

/**
 * @brief Reads a CSV file and returns its contents as a 2D vector of doubles.
 * @param filename The path to the CSV file.
 * @param skip_header If true, skips the first line of the CSV file.
 * @param max_values A reference to a vector that will be populated with the
 * maximum value of each column.
 * @param min_values A reference to a vector that will be populated with the
 * minimum value of each column.
 * @return A 2D vector where each inner vector represents a row of doubles from
 *	the CSV file.
 * @details
 * @p max_values and @p min_values are computed while reading the file for a
 * performance boost.
 */
std::vector<std::vector<double>> readCSV(const std::string& filename,
										 bool skip_header = false,
										 std::vector<double>& max_values = *(new std::vector<double>()),
										 std::vector<double>& min_values = *(new std::vector<double>()));
#endif // CSV_UTILS_HPP
