#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

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
        std::vector<double> point;         // Point coordinates in K dimensions
        std::unique_ptr<Node> left;        // Pointer to left child (points with smaller coordinate)
        std::unique_ptr<Node> right;       // Pointer to right child (points with larger coordinate)
        mutable int cluster;               // Cluster assignment
        
        friend class KDTree;
        
    public:
        Node(const std::vector<double>& pt);
        std::vector<double> getPoint() const;
        const Node* getLeft() const;       // Keep const accessors for safety
        const Node* getRight() const;
        void setCluster(int c) const;
        int getCluster() const { return cluster; }
    };

private:
    static constexpr double default_eps = 1e-9; 
    std::unique_ptr<Node> root;            // Root node of the KD-tree
    size_t K;                              // Number of dimensions
    size_t node_count;                     // Track number of nodes in the tree
    
    // Validates that all points have the correct dimensionality
    void validatePoints(const std::vector<std::vector<double>>& points, size_t expectedDim) const;
    
    // Recursively builds a balanced KD-tree using median splitting
    std::unique_ptr<Node> buildTree(std::vector<std::vector<double>>& points, 
                                    int start, int end, int depth);
    
    // Recursively searches for a point in the KD-tree
    bool searchRecursive(const std::unique_ptr<Node>& node, 
                        const std::vector<double>& point, 
                        int depth, 
                        double eps = default_eps) const;
    
    // Recursively inserts a point respecting KDTree splitting rules
    // Returns true if point was inserted, false if already exists
    bool insertRecursive(std::unique_ptr<Node>& node,
                        const std::vector<double>& point,
                        int depth,
                        double eps = default_eps);
    
    // Recursively deletes a point from the tree
    // Returns true if point was found and deleted, false otherwise
    bool deleteRecursive(std::unique_ptr<Node>& node,
                        const std::vector<double>& point,
                        int depth,
                        double eps = default_eps);
    
    // Helper for deleteRecursive: finds the minimum point in a given dimension
    const Node* findMin(const std::unique_ptr<Node>& node, int dim, int depth) const;
    
    // Recursively counts nodes in the tree
    size_t countNodes(const std::unique_ptr<Node>& node) const;
    
    // Recursively calculates tree height
    int getHeight(const std::unique_ptr<Node>& node) const;
    
    // Recursively prints the KD-tree structure
    void printRecursive(const std::unique_ptr<Node>& node, int depth) const;
    
    // Recursively collects all points from the tree
    void collectPoints(const std::unique_ptr<Node>& node, 
                      std::vector<std::vector<double>>& points) const;
    
    void writeCSVRecursive(const std::unique_ptr<Node>& node, 
                          std::ofstream& out) const;

public:
    // Constructs an empty KD-tree (dimension inferred on first build/insert)
    KDTree();
    
    // Constructs an empty KD-tree with specified dimensionality
    KDTree(size_t dimension);
    
    // Constructs a balanced KD-tree from a vector of points
    KDTree(std::vector<std::vector<double>> points);
    
    ~KDTree() = default;
    
    // Deleted copy constructor to prevent shallow copies
    KDTree(const KDTree&) = delete;
    
    // Deleted assignment operator to prevent shallow copies
    KDTree& operator=(const KDTree&) = delete;
    
    // Add move operations
    KDTree(KDTree&& other) noexcept = default;
    KDTree& operator=(KDTree&& other) noexcept = default;
    
    // Builds a balanced KD-tree from a vector of points
    // Throws std::invalid_argument if points have mismatched dimensions
    void build(std::vector<std::vector<double>> points);
    
    // Inserts a single point maintaining KDTree structure (O(log n) average)
    // Returns true if inserted, false if already exists
    // Throws std::invalid_argument if point dimension doesn't match K
    bool insert(const std::vector<double>& point);
    
    // Deletes a point from the tree (O(log n) average, worst case O(n))
    // Returns true if deleted, false if not found
    bool deletePoint(const std::vector<double>& point);
    
    // Searches for a point in the KD-tree
    // Throws std::invalid_argument if point dimension doesn't match K
    bool search(const std::vector<double>& point) const;
    
    // Prints the tree structure to standard output
    void print() const;
    
    // Checks if the tree is empty
    bool empty() const;
    
    // Returns the number of nodes in the tree
    size_t size() const;
    
    // Returns the height of the tree (0 for single node, -1 for empty)
    int height() const;
    
    // Return root node
    const Node* getRoot() const;
    
    // Append a point without checking if it's the right position (just for testing)
    Node* appendNode(const std::vector<double>& point);
    
    void writeCSV(const std::string& filename) const;
};

#endif