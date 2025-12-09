#ifndef CSV_UTILS_HPP
#define CSV_UTILS_HPP

#include <string>
#include <vector>
#include <array>

/**
 * @brief Reads a CSV file and returns its contents as a 2D vector of doubles.
 * @param filename The path to the CSV file.
 * @return A 2D vector where each inner vector represents a row of doubles from
 *	the CSV file.
 */
std::vector<std::vector<double>> readCSV(const std::string& filename,
										 bool skip_header = false,
										 std::vector<double>& max_values = *(new std::vector<double>()),
										 std::vector<double>& min_values = *(new std::vector<double>()));

#endif // CSV_UTILS_HPP
