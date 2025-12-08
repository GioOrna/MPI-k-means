
#include "KDTree.hpp"
#include <iostream>
#include <array>
#include <iomanip>
#include <vector>

using namespace std;


void printTestHeader(const string& title) {
    cout << "\n" << string(70, '=') << endl;
    cout << " " << title << endl;
    cout << string(70, '=') << endl;
}

int main() {
    int passedTests = 0;
    int totalTests = 0;
    
    printTestHeader("Balanced 2D Tree - Median-Based Construction");
    
    vector<array<double, 2>> points2D = {
        {3, 6}, {2, 2}, {4, 7}, {1, 3}, {2, 4}, {5, 4}, {7, 2}
    };
    
    KDTree<2> tree2D(points2D);
    
    cout << "Building balanced tree from 7 points." << endl;
    tree2D.print();

    

    printTestHeader("Search for All Inserted Points");
    
    bool allFound = true;
    for (const auto& point : points2D) {
        bool found = tree2D.search(point);
        cout << "Searching for (" << point[0] << ", " << point[1] << "): " 
             << (found ? "Found" : "Not found") << endl;
        if (!found) allFound = false;
    }
    
    
    printTestHeader("Search for Non-existing Points");
    
    vector<array<double, 2>> nonExistingPoints = {
        {0, 0}, {6, 3}, {10, 10}, {3, 3}, {5, 5}
    };
    
    bool noneFound = true;
    for (const auto& point : nonExistingPoints) {
        bool found = tree2D.search(point);
        cout << "Searching for (" << point[0] << ", " << point[1] << "): " 
             << (found ? "Found" : "Not found") << endl;
        if (found) noneFound = false;
    }

    
    return 0;
}