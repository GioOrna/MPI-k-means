#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <fstream>



// Epsilon comparison helpers

inline bool approxEqual(double a, double b, double eps = 1e-9) {
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff <= eps * scale;
}

bool approxEqualPoint(const std::vector<double>& a,
                      const std::vector<double>& b,
                      double eps = 1e-9) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approxEqual(a[i], b[i], eps))
            return false;
    }
    return true;
}


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
        Node(const std::vector<double>& pt) 
            : point(pt), left(nullptr), right(nullptr), cluster(-1) {}
        
        std::vector<double> getPoint() const{
            return point;
        }

        //navigate to the left of the node
        const Node* getLeft() const{
            return left;
        }

        //navigate to the right of the node
        const Node* getRight() const{
            return right;
        }

            //set cluster field of the node
        void setCluster(int c) const{
            cluster = c;
        }
    };
    
    Node* root;  // Root node of the KD-tree
    size_t K;   // Number of dimensions
    
    // Recursively builds a balanced KD-tree using median splitting
    Node* buildTree(std::vector<std::vector<double>>& points, 
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
                        [cd](const std::vector<double>& a, 
                             const std::vector<double>& b) {
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
    bool searchRecursive(Node* node,
                        const std::vector<double>& point,
                        int depth,
                        double eps = 1e-9) const {
        // Base case: reached a leaf (point not found)
        if (node == nullptr) {
            return false;
        }
        
        // Check if points are equal within tolerance
        if (approxEqualPoint(node->point, point, eps)) {
            return true;
        }

        // Calculate current dimension (cycles through 0 to K-1)
        int cd = depth % K;

        // Recurse down the appropriate subtree based on current dimension
        if (approxEqual(point[cd], node->point[cd], eps)) {
            return searchRecursive(node->left, point, depth + 1, eps) ||
                   searchRecursive(node->right, point, depth + 1, eps);
        }
        if (point[cd] < node->point[cd]) {
            return searchRecursive(node->left, point, depth + 1, eps);
        } else {
            return searchRecursive(node->right, point, depth + 1, eps);
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
    void collectPoints(Node* node, std::vector<std::vector<double>>& points) const {
        if (node == nullptr) return;
        
        points.push_back(node->point);
        collectPoints(node->left, points);
        collectPoints(node->right, points);
    }

void writeCSVRecursive(Node* node, std::ofstream& out) const {
    if (!node) return;

    // Write point coordinates
    for (size_t i = 0; i < node->point.size(); ++i) {
        out << node->point[i];
        out << ",";
    }
    out << node->cluster;
    out << "\n";

    // Recurse
    writeCSVRecursive(node->left, out);
    writeCSVRecursive(node->right, out);
}



public:
    // Constructs an empty KD-tree
    KDTree(size_t dimension) : root(nullptr), K(dimension) {}
    
    // Constructs a balanced KD-tree from a vector of points
    KDTree(std::vector<std::vector<double>> points) : root(nullptr) {
        if (!points.empty()) {
            K = points[0].size();
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
    void build(std::vector<std::vector<double>> points) {
        // Delete existing tree
        deleteRecursive(root);
        root = nullptr;
        
        // Build new tree
        if (!points.empty()) {
            K = points[0].size();
            root = buildTree(points, 0, points.size(), 0);
        }
    }

    
    // Inserts a single point and rebuilds the tree (inefficient)
    // NOTE: This is inefficient (O(n log n)) as it rebuilds the entire tree.
    void insert(const std::vector<double>& point) {
        // Collect all existing points
        std::vector<std::vector<double>> points;
        collectPoints(root, points);
        
        // Add new point
        points.push_back(point);
        
        // Rebuild tree
        build(points);
    }
    
    // Searches for a point in the KD-tree
    bool search(const std::vector<double>& point) const {
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
    
    //return root node
    const Node* getRoot(){
        return root;
    }

    //append a point without checking if it's the right position
    const Node* appendNode(const std::vector<double>& point){
        Node* parsing_node = root;
        while(parsing_node->right != nullptr){
		    parsing_node = parsing_node->right;
	    }
        parsing_node->right = new KDTree::Node(point); //just to test the output file
        return parsing_node->right;
    }
    
    void writeCSV(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Header: x0,x1,...,x(K-1),splitDim
    for (size_t i = 0; i < K; ++i) {
        out << "x" << i;
        if (i < K - 1) out << ",";
    }
    out << ",BelongingCluster\n";

    writeCSVRecursive(root, out);

    out.close();
}
    
};
#endif // KDTREE_HPP