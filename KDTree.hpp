#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>
#include <algorithm>

// A balanced K-Dimensional tree using median-based splitting
// K is the number of dimensions in the space

template <size_t K>
class KDTree {
private:
    struct Node {
        std::array<double, K> point;   // Point coordinates in K dimensions
        Node* left;                    // Pointer to left child (points with smaller coordinate)
        Node* right;                   // Pointer to right child (points with larger coordinate)
        
        // Constructs a new Node
        Node(const std::array<double, K>& pt) 
            : point(pt), left(nullptr), right(nullptr) {}
    };
    
    Node* root;  // Root node of the KD-tree
    
    // Recursively builds a balanced KD-tree using median splitting
    Node* buildTree(std::vector<std::array<double, K>>& points, 
                    int start, int end, int depth) {
        // Base case: no points to process
        if (start >= end) {
            return nullptr;
        }
        
        // Calculate current dimension (cycles through 0 to K-1)
        int cd = depth % K;
        
        // Find median index
        int mid = start + (end - start) / 2;
        
        // Partition around median using nth_element
        // This puts the median at position 'mid' and partitions around it
        std::nth_element(points.begin() + start, 
                        points.begin() + mid, 
                        points.begin() + end,
                        [cd](const std::array<double, K>& a, 
                             const std::array<double, K>& b) {
                            return a[cd] < b[cd];
                        });
        
        // Create node with median point
        Node* node = new Node(points[mid]);
        
        // Recursively build left and right subtrees
        node->left = buildTree(points, start, mid, depth + 1);
        node->right = buildTree(points, mid + 1, end, depth + 1);
        
        return node;
    }
    
    // Recursively searches for a point in the KD-tree
    bool searchRecursive(Node* node, const std::array<double, K>& point, int depth) const {
        // Base case: reached a leaf (point not found)
        if (node == nullptr) {
            return false;
        }
        
        // Check if current node matches the search point
        if (node->point == point) {
            return true;
        }
        
        // Calculate current dimension (cycles through 0 to K-1)
        int cd = depth % K;
        
        // Recurse down the appropriate subtree based on current dimension
        if (point[cd] < node->point[cd]) {
            return searchRecursive(node->left, point, depth + 1);
        } else if (point[cd] > node->point[cd]) {
            return searchRecursive(node->right, point, depth + 1);
        } else {
            // When coordinates are equal, point could be in EITHER subtree
            return searchRecursive(node->left, point, depth + 1) ||
                   searchRecursive(node->right, point, depth + 1);
        }
    }
    
    // Recursively prints the KD-tree structure
    void printRecursive(Node* node, int depth) const {
        // Base case: null node, nothing to print
        if (node == nullptr) {
            return;
        }
        
        // Print indentation based on depth
        for (int i = 0; i < depth; i++) {
            std::cout << "  ";
        }
        
        // Print the point coordinates with the splitting dimension
        std::cout << "(";
        for (size_t i = 0; i < K; i++) {
            std::cout << node->point[i];
            if (i < K - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ") [dim=" << (depth % K) << "]" << std::endl;
        
        // Recursively print left and right subtrees
        printRecursive(node->left, depth + 1);
        printRecursive(node->right, depth + 1);
    }
    
    // Recursively deletes all nodes in the tree
    void deleteRecursive(Node* node) {
        if (node == nullptr) return;
        
        // Post-order traversal: delete children first
        deleteRecursive(node->left);
        deleteRecursive(node->right);
        delete node;
    }
    
    // Recursively collects all points from the tree
    void collectPoints(Node* node, std::vector<std::array<double, K>>& points) const {
        if (node == nullptr) return;
        
        points.push_back(node->point);
        collectPoints(node->left, points);
        collectPoints(node->right, points);
    }

public:
    // Constructs an empty KD-tree
    KDTree() : root(nullptr) {}
    
    // Constructs a balanced KD-tree from a vector of points
    KDTree(std::vector<std::array<double, K>> points) : root(nullptr) {
        if (!points.empty()) {
            root = buildTree(points, 0, points.size(), 0);
        }
    }
    
    // Destroys the KD-tree and frees all memory
    ~KDTree() {
        deleteRecursive(root);
    }
    
    // Deleted copy constructor to prevent shallow copies
    KDTree(const KDTree&) = delete;
    
    // Deleted assignment operator to prevent shallow copies
    KDTree& operator=(const KDTree&) = delete;
    
    // Builds a balanced KD-tree from a vector of points
    void build(std::vector<std::array<double, K>> points) {
        // Delete existing tree
        deleteRecursive(root);
        root = nullptr;
        
        // Build new tree
        if (!points.empty()) {
            root = buildTree(points, 0, points.size(), 0);
        }
    }
    
    // Inserts a single point and rebuilds the tree (inefficient)
    // NOTE: This is inefficient (O(n log n)) as it rebuilds the entire tree.
    void insert(const std::array<double, K>& point) {
        // Collect all existing points
        std::vector<std::array<double, K>> points;
        collectPoints(root, points);
        
        // Add new point
        points.push_back(point);
        
        // Rebuild tree
        build(points);
    }
    
    // Searches for a point in the KD-tree
    bool search(const std::array<double, K>& point) const {
        return searchRecursive(root, point, 0);
    }
    
    // Prints the tree structure to standard output
    void print() const {
        printRecursive(root, 0);
    }
    
    // Checks if the tree is empty
    bool empty() const {
        return root == nullptr;
    }
};

#endif // KDTREE_HPP