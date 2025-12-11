#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>

// Epsilon comparison helpers
inline bool approxEqual(double a, double b, double eps = 1e-9) {
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff <= eps * scale;
}

bool approxEqualPoint(const std::vector<double>& a,
                      const std::vector<double>& b,
                      double eps = 1e-9);

// A balanced K-Dimensional tree using median-based splitting
// K is the number of dimensions in the space
class KDTree {
public:
    struct Node {
    private:
        std::vector<double> point;   // Point coordinates in K dimensions
        Node* left;                    // Pointer to left child (points with smaller coordinate)
        Node* right;                   // Pointer to right child (points with larger coordinate)
        mutable int cluster;                  // Cluster assignment
        // Constructs a new Node
        friend class KDTree;

    public:
        Node(const std::vector<double>& pt);
        std::vector<double> getPoint() const;
        const Node* getLeft() const;
        const Node* getRight() const;
        void setCluster(int c) const;
    };

private:
    Node* root; // Root node of the KD-tree
    size_t K;   // Number of dimensions

    // Recursively builds a balanced KD-tree using median splitting
    Node* buildTree(std::vector<std::vector<double>>& points, int start, int end, int depth);
    // Recursively searches for a point in the KD-tree
    bool searchRecursive(Node* node, const std::vector<double>& point, int depth, double eps) const;
    // Recursively prints the KD-tree structure
    void printRecursive(Node* node, int depth) const;
    // Recursively deletes all nodes in the tree
    void deleteRecursive(Node* node);
    // Recursively collects all points from the tree
    void collectPoints(Node* node, std::vector<std::vector<double>>& points) const;
    void writeCSVRecursive(Node* node, std::ofstream& out) const;

public:
    // Constructs an empty KD-tree
    KDTree(size_t dimension);
    // Constructs a balanced KD-tree from a vector of points
    KDTree(std::vector<std::vector<double>> points);
    // Destroys the KD-tree and frees all memory
    ~KDTree();
    // Deleted copy constructor to prevent shallow copies
    KDTree(const KDTree&) = delete;
    // Deleted assignment operator to prevent shallow copies
    KDTree& operator=(const KDTree&) = delete;
    // Builds a balanced KD-tree from a vector of points
    void build(std::vector<std::vector<double>> points);
    // Inserts a single point and rebuilds the tree (inefficient)
    // NOTE: This is inefficient (O(n log n)) as it rebuilds the entire tree.
    void insert(const std::vector<double>& point);
    // Searches for a point in the KD-tree
    bool search(const std::vector<double>& point) const;
    // Prints the tree structure to standard output
    void print() const;
    // Checks if the tree is empty
    bool empty() const;

    //return root node
    const Node* getRoot();
    //append a point without checking if it's the right position
    const Node* appendNode(const std::vector<double>& point);

    void writeCSV(const std::string& filename) const;
};

#endif
