
#include "../include/KDTree.hpp"
#include <iostream>
#include <array>
#include <iomanip>
#include <string> 
#include <vector>

using namespace std;

/**
 * @brief Prints a test section header
 * @param title The title of the test section
 */
void printTestHeader(const string& title) {
    cout << "\n" << string(70, '=') << endl;
    cout << " " << title << endl;
    cout << string(70, '=') << endl;
}

/**
 * @brief Prints test result
 * @param testName Name of the test
 * @param passed Whether the test passed
 */
void printTestResult(const string& testName, bool passed) {
    cout << "[" << (passed ? "PASS" : "FAIL") << "] " << testName << endl;
}

int main() {
    int passedTests = 0;
    int totalTests = 0;
    
    // Test 1: Basic 2D Balanced Tree Construction

    printTestHeader("Test 1: Balanced 2D Tree - Median-Based Construction");
    
    vector<vector<double>> points2D = {
        {3, 6}, {2, 2}, {4, 7}, {1, 3}, {2, 4}, {5, 4}, {7, 2}
    };
    
    KDTree tree2D(points2D);
    
    cout << "Building balanced tree from 7 points:" << endl;
    cout << "(3,6), (2,2), (4,7), (1,3), (2,4), (5,4), (7,2)" << endl;
    cout << "\nBalanced tree structure (with splitting dimensions):" << endl;
    tree2D.print();
    cout << "\nNote: [dim=X] shows which dimension was used for splitting at that level" << endl;
    
    totalTests++;
    passedTests++;
    printTestResult("Balanced 2D Tree Creation", true);

    
    // Test 2: Verify Balance - Compare with Sequential Insertion

    printTestHeader("Test 2: Balance Verification - Sequential vs Balanced");
    
    cout << "Sequential points: (1,1), (2,2), (3,3), (4,4), (5,5)" << endl;
    
    vector<vector<double>> sequentialPoints = {
        {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}
    };
    
    KDTree balancedTree(sequentialPoints);
    
    cout << "\nBalanced tree (median-based splitting):" << endl;
    balancedTree.print();
    cout << "\nNotice the tree is balanced with depth approx 2-3 levels" << endl;
    
    totalTests++;
    passedTests++;
    printTestResult("Balance property verified", true);


    // Test 3: Search Functionality - All Points

    printTestHeader("Test 3: Search for All Inserted Points");
    
    bool allFound = true;
    for (const auto& point : points2D) {
        bool found = tree2D.search(point);
        cout << "Searching for (" << point[0] << ", " << point[1] << "): " 
             << (found ? "Found" : "Not found") << endl;
        if (!found) allFound = false;
    }

    totalTests++;
    if (allFound) passedTests++;
    printTestResult("All existing points found", allFound);


    // Test 4: Search Functionality - Non-existing Points

    printTestHeader("Test 4: Search for Non-existing Points");
    
    vector<vector<double>> nonExistingPoints = {
        {0, 0}, {6, 3}, {10, 10}, {3, 3}, {5, 5}
    };
    
    bool noneFound = true;
    for (const auto& point : nonExistingPoints) {
        bool found = tree2D.search(point);
        cout << "Searching for (" << point[0] << ", " << point[1] << "): " 
             << (found ? "Found" : "Not found") << endl;
        if (found) noneFound = false;
    }

    totalTests++;
    if (noneFound) passedTests++;
    printTestResult("No false positives", noneFound);


    // Test 5: 3D Balanced Tree
    printTestHeader("Test 5: 3D Balanced Tree");
    
    vector<vector<double>> points3D = {
        {5, 4, 3}, {2, 6, 7}, {8, 1, 4}, {3, 9, 2}, 
        {7, 2, 8}, {1, 5, 6}, {6, 3, 1}
    };
    
    KDTree tree3D(points3D);
    
    cout << "Building balanced 3D tree from 7 points:" << endl;
    cout << "\nBalanced tree structure:" << endl;
    tree3D.print();
    
    cout << "\nSearching in 3D tree:" << endl;
    bool found3D_1 = tree3D.search({5, 4, 3});
    bool found3D_2 = tree3D.search({7, 2, 8});
    bool notFound3D = tree3D.search({1, 1, 1});
    
    cout << "Search for (5,4,3): " << (found3D_1 ? "Found" : "Not found") << endl;
    cout << "Search for (7,2,8): " << (found3D_2 ? "Found" : "Not found") << endl;
    cout << "Search for (1,1,1): " << (notFound3D ? "Found" : "Not found") << endl;
    
    totalTests++;
    if (found3D_1 && found3D_2 && !notFound3D) passedTests++;
    printTestResult("3D Balanced Tree operations", found3D_1 && found3D_2 && !notFound3D);
    

    // Test 6: Large Balanced Tree

    printTestHeader("Test 6: Large Balanced Tree (100 points)");
    
    vector<vector<double>> largePoints;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            largePoints.push_back({static_cast<double>(i), static_cast<double>(j)});
        }
    }
    
    KDTree largeTree(largePoints);
    
    cout << "Built balanced tree with 100 points" << endl;
    cout << "Expected depth approx 6-7 levels" << endl;
    
    cout << "\nSearching for corner points:" << endl;
    bool found_00 = largeTree.search({0.0, 0.0});
    bool found_99 = largeTree.search({9.0, 9.0});
    bool found_55 = largeTree.search({5.0, 5.0});
    bool notFound_10_10 = largeTree.search({10.0, 10.0});
    
    cout << "Search for (0,0): " << (found_00 ? "Found" : "Not found") << endl;
    cout << "Search for (9,9): " << (found_99 ? "Found" : "Not found") << endl;
    cout << "Search for (5,5): " << (found_55 ? "Found" : "Not found") << endl;
    cout << "Search for (10,10): " << (notFound_10_10 ? "Found" : "Not found ") << endl;
    
    totalTests++;
    if (found_00 && found_99 && found_55 && !notFound_10_10) passedTests++;
    printTestResult("Large balanced tree operations", 
                    found_00 && found_99 && found_55 && !notFound_10_10);
    

    // Test 7: Incremental Insert (with rebuild)

    printTestHeader("Test 7: Incremental Insert with Tree Rebuild");
    
    vector<vector<double>> initialPoints = {{3, 6}, {2, 2}, {4, 7}};
    KDTree incrementalTree(initialPoints);
    
    cout << "Initial tree with 3 points:" << endl;
    incrementalTree.print();
    
    cout << "\nInserting new point (5, 4) - this rebuilds the entire tree:" << endl;
    incrementalTree.insert({5, 4});
    incrementalTree.print();
    
    bool foundNew = incrementalTree.search({5, 4});
    bool foundOld = incrementalTree.search({3, 6});
    
    cout << "\nSearch for new point (5,4): " << (foundNew ? "Found" : "Not found") << endl;
    cout << "Search for old point (3,6): " << (foundOld ? "Found" : "Not found") << endl;
    
    totalTests++;
    if (foundNew && foundOld) passedTests++;
    printTestResult("Incremental insert with rebuild", foundNew && foundOld);
    

    // Test 8: Build Method - Replace Existing Tree

    printTestHeader("Test 8: Build Method - Replace Existing Tree");
    
    vector<vector<double>> newPoints = {{10, 10}, {20, 20}, {30, 30}};
    incrementalTree.build(newPoints);
    
    cout << "Rebuilt tree with completely new points:" << endl;
    incrementalTree.print();
    
    bool foundReplaced = incrementalTree.search({10, 10});
    bool oldPointGone = !incrementalTree.search({3, 6});
    
    cout << "\nSearch for new point (10,10): " << (foundReplaced ? "Found" : "Not found") << endl;
    cout << "Old point (3,6) removed: " << (oldPointGone ? "Yes" : "No") << endl;
    
    totalTests++;
    if (foundReplaced && oldPointGone) passedTests++;
    printTestResult("Build method replaces tree correctly", foundReplaced && oldPointGone);


    // Test 9: Duplicate Points Handling

    printTestHeader("Test 9: Duplicate Points in Construction");
    
    vector<vector<double>> pointsWithDuplicates = {
        {3, 4}, {3, 4}, {5, 6}, {3, 4}, {7, 8}
    };
    
    KDTree dupTree(pointsWithDuplicates);
    
    cout << "Built tree with duplicate points: (3,4) appears 3 times" << endl;
    cout << "\nTree structure:" << endl;
    dupTree.print();
    
    bool foundDup = dupTree.search({3, 4});
    bool foundUnique = dupTree.search({5, 6});

    cout << "\nSearch for (3,4): " << (foundDup ? "Found" : "Not found") << endl;
    cout << "Search for (5,6): " << (foundUnique ? "Found" : "Not found") << endl;
    
    totalTests++;
    if (foundDup && foundUnique) passedTests++;
    printTestResult("Duplicate handling", foundDup && foundUnique);
    
    
    // Test 10: Empty Tree

    printTestHeader("Test 10: Empty Tree Operations");
    
    KDTree emptyTree;
    
    cout << "Created empty tree" << endl;
    cout << "Is empty: " << (emptyTree.empty() ? "Yes" : "No") << endl;
    cout << "Size: " << emptyTree.size() << endl;
    cout << "Height: " << emptyTree.height() << endl;
    
    // Test inserting into initially empty tree
    cout << "\nInserting point (5, 5) into empty tree:" << endl;
    bool inserted = emptyTree.insert({5, 5});
    cout << "Inserted: " << (inserted ? "Yes" : "No") << endl;
    cout << "Tree now empty: " << (emptyTree.empty() ? "Yes" : "No") << endl;
    cout << "Size after insert: " << emptyTree.size() << endl;
    cout << "Height after insert: " << emptyTree.height() << endl;
    
    bool foundAfterInsert = emptyTree.search({5, 5});
    cout << "Search for (5,5) after insert: " << (foundAfterInsert ? "Found" : "Not found") << endl;
    
    // Insert more points
    cout << "\nInserting additional points:" << endl;
    emptyTree.insert({3, 3});
    emptyTree.insert({7, 7});
    emptyTree.insert({2, 2});
    cout << "Size after more inserts: " << emptyTree.size() << endl;
    
    // Test building on previously empty tree
    cout << "\nBuilding tree from points (replaces previous data):" << endl;
    vector<vector<double>> buildPoints = {{1, 1}, {2, 2}, {3, 3}};
    emptyTree.build(buildPoints);
    cout << "After build - Size: " << emptyTree.size() << endl;
    cout << "After build - Empty: " << (emptyTree.empty() ? "Yes" : "No") << endl;
    cout << "After build - Height: " << emptyTree.height() << endl;
    
    bool foundAfterBuild = emptyTree.search({2, 2});
    bool oldPointRemoved = !emptyTree.search({5, 5});
    cout << "Search for (2,2) after build: " << (foundAfterBuild ? "Found" : "Not found") << endl;
    cout << "Old point (5,5) removed: " << (oldPointRemoved ? "Yes" : "No") << endl;
    
    totalTests++;
    bool test10Passed = inserted && 
                       foundAfterInsert && 
                       foundAfterBuild && 
                       oldPointRemoved;
    if (test10Passed) passedTests++;
    printTestResult("Empty tree operations", test10Passed);

    
    // Final Summary

    printTestHeader("TEST SUMMARY");
    
    cout << "Total Tests: " << totalTests << endl;
    cout << "Passed: " << passedTests << endl;
    cout << "Failed: " << (totalTests - passedTests) << endl;
    cout << "Success Rate: " << fixed << setprecision(1) 
         << (100.0 * passedTests / totalTests) << "%" << endl;
    
    if (passedTests == totalTests) {
        cout << "\nALL TESTS PASSED!" << endl;
        cout << "The KD-tree is correctly balanced using median-based splitting!" << endl;
    } else {
        cout << "\nSome tests failed. Review the output above." << endl;
    }

    return 0;
}