#include "../include/KDTree.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using namespace std;


// Utility functions
void printTestHeader(const string& title) {
    cout << "\n" << string(70, '=') << "\n";
    cout << " " << title << "\n";
    cout << string(70, '=') << "\n";
}

void printTestResult(const string& testName, bool passed) {
    cout << "[" << (passed ? "PASS" : "FAIL") << "] " << testName << endl;
}


// Test 1: Basic 2D Balanced Tree Construction
bool testBalanced2DTree() {
    vector<vector<double>> points2D = {{3,6}, {2,2}, {4,7}, {1,3}, {2,4}, {5,4}, {7,2}};
    KDTree tree(points2D);
    tree.print();

    for (const auto& pt : points2D) {
        if (!tree.search(pt)) return false;
    }
    return true;
}


// Test 2: Search for non-existing points

bool testSearchNonExisting() {
    vector<vector<double>> points = {{1,1}, {2,2}, {3,3}};
    KDTree tree(points);

    vector<vector<double>> nonExist = {{0,0}, {4,4}, {5,5}};
    for (const auto& pt : nonExist) {
        if (tree.search(pt)) return false;
    }
    return true;
}


// Test 3: 3D Tree Construction and Search

bool test3DTree() {
    vector<vector<double>> points3D = {{5,4,3}, {2,6,7}, {8,1,4}, {3,9,2}, {7,2,8}, {1,5,6}, {6,3,1}};
    KDTree tree(points3D);

    return tree.search({5,4,3}) &&
           tree.search({7,2,8}) &&
           !tree.search({0,0,0});
}


// Test 4: Empty tree behavior

bool testEmptyTree() {
    KDTree tree;
    if (!tree.empty() || tree.size() != 0 || tree.height() != -1) return false;

    tree.insert({1,1});
    if (tree.empty() || tree.size() != 1) return false;

    tree.build({{2,2},{3,3}});
    if (tree.empty() || tree.size() != 2) return false;

    return true;
}


// Test 5: Incremental insertion and duplicates

bool testIncrementalInsert() {
    KDTree tree;
    tree.insert({1,2});
    tree.insert({3,4});
    bool duplicateInsert = !tree.insert({1,2}); // should not insert
    return tree.size() == 2 && tree.search({1,2}) && tree.search({3,4}) && duplicateInsert;
}


// Test 6: appendNode() and cluster

bool testAppendNodeAndCluster() {
    vector<vector<double>> points = {{1,1}, {2,2}};
    KDTree tree(points);

    KDTree::Node* newNode = tree.appendNode({5,5});
    newNode->setCluster(99);

    return newNode && tree.size() == 3 && tree.search({5,5}) && newNode->getCluster() == 99;
}


// Test 7: writeCSV()

bool testWriteCSV() {
    vector<vector<double>> points = {{1,2}, {3,4}};
    KDTree tree(points);

    const string filename = "kdtree_test.csv";
    tree.writeCSV(filename);

    ifstream file(filename);
    bool exists = file.is_open();
    file.close();
    remove(filename.c_str()); // cleanup
    return exists;
}


// Test 8: getRoot(), height(), size()

bool testRootHeightSize() {
    vector<vector<double>> points = {{1,1}, {2,2}, {3,3}};
    KDTree tree(points);

    const KDTree::Node* root = tree.getRoot();
    if (!root) return false;
    if (tree.size() != 3) return false;
    int h = tree.height();
    return h >= 1;
}


// Test 9: Approximate search

bool testApproximateSearch() {
    KDTree tree({{1,1}, {2,2}});
    return tree.search({1+1e-10,1-1e-10});
}


// Test 10: Exceptions (wrong dimension)

bool testDimensionExceptions() {
    bool caughtBuild = false, caughtInsert = false;
    try {
        KDTree t;
        t.build({{1,2},{3,4,5}});
    } catch (const invalid_argument&) { caughtBuild = true; }

    try {
        KDTree t(2);
        t.insert({1,2,3});
    } catch (const invalid_argument&) { caughtInsert = true; }

    return caughtBuild && caughtInsert;
}


// Test 11: appendNode() on empty tree throws

bool testAppendEmptyThrows() {
    bool caught = false;
    try {
        KDTree t;
        t.appendNode({1,1});
    } catch (const runtime_error&) { caught = true; }
    return caught;
}


// Main function: run all tests

int main() {
    struct Test { string name; bool (*func)(); };
    vector<Test> tests = {
        {"Balanced 2D Tree", testBalanced2DTree},
        {"Search Non-existing Points", testSearchNonExisting},
        {"3D Tree Operations", test3DTree},
        {"Empty Tree Behavior", testEmptyTree},
        {"Incremental Insert & Duplicates", testIncrementalInsert},
        {"Append Node & Cluster", testAppendNodeAndCluster},
        {"Write CSV", testWriteCSV},
        {"Root, Height, Size", testRootHeightSize},
        {"Approximate Search", testApproximateSearch},
        {"Dimension Exceptions", testDimensionExceptions},
        {"Append Node on Empty Throws", testAppendEmptyThrows}
    };
    

    int totalTests = 0, passedTests = 0;
    for (const auto& t : tests) {
        totalTests++;
        printTestHeader("Running: " + t.name);
        bool passed = t.func();
        if (passed) passedTests++;
        printTestResult(t.name, passed);
    }

    printTestHeader("TEST SUMMARY");
    cout << "Total Tests: " << totalTests << "\n";
    cout << "Passed: " << passedTests << "\n";
    cout << "Failed: " << totalTests - passedTests << "\n";
    cout << fixed << setprecision(1)
         << "Success Rate: " << (100.0 * passedTests / totalTests) << "%\n";

    if (passedTests == totalTests)
        cout << "\nALL TESTS PASSED! KD-tree implementation is correct.\n";
    else
        cout << "\nSome tests failed. Check output above.\n";

    return 0;
}
