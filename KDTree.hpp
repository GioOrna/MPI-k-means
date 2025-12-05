#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <array>
#include <cstddef>

// Template class for KDTree with K dimensions
template <size_t K>
class KDTree {
private:
    // Node structure representing each point in the KDTree
    struct Node {
        // Point in K dimensions
        std::array<double, K> point;
        // Pointer to left child
        Node* left;
        // Pointer to right child
        Node* right;
        
        // Constructor to initialize a Node
        Node(const std::array<double, K>& pt);
    };
    
    // Root of the KDTree
    Node* root;
    
    // Recursive function to insert a point into the KDTree
    Node* insertRecursive(Node* node, const std::array<double, K>& point, int depth);
    
    // Recursive function to search for a point in the KDTree
    bool searchRecursive(Node* node, const std::array<double, K>& point, int depth) const;
    
    // Recursive function to print the KDTree
    void printRecursive(Node* node, int depth) const;
    
    // Recursive function to delete all nodes
    void deleteRecursive(Node* node);

public:
    // Constructor to initialize the KDTree with a null root
    KDTree();
    
    // Destructor to clean up memory
    ~KDTree();
    
    // Delete copy constructor and assignment operator
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    
    // Public function to insert a point into the KDTree
    void insert(const std::array<double, K>& point);
    
    // Public function to search for a point in the KDTree
    bool search(const std::array<double, K>& point) const;
    
    // Public function to print the KDTree
    void print() const;
};

// Include implementation for template class
#include "KDTree.cpp"

#endif // KDTREE_HPP