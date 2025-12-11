#include "../include/KDTree.hpp"

// ---- approxEqualPoint ----
bool approxEqualPoint(const std::vector<double>& a,
                      const std::vector<double>& b,
                      double eps) {
    for (size_t i = 0; i < a.size(); ++i)
        if (!approxEqual(a[i], b[i], eps))
            return false;
    return true;
}

// ---- Node methods ----

KDTree::Node::Node(const std::vector<double>& pt)
    : point(pt), left(nullptr), right(nullptr), cluster(-1) {}

std::vector<double> KDTree::Node::getPoint() const { return point; }
const KDTree::Node* KDTree::Node::getLeft() const { return left; }
const KDTree::Node* KDTree::Node::getRight() const { return right; }
void KDTree::Node::setCluster(int c) const { cluster = c; }

// ---- KDTree methods ----

KDTree::KDTree(size_t dimension) : root(nullptr), K(dimension) {}

KDTree::KDTree(std::vector<std::vector<double>> points) : root(nullptr) {
    if (!points.empty()) {
        K = points[0].size();
        root = buildTree(points, 0, points.size(), 0);
    }
}

KDTree::~KDTree() {
    deleteRecursive(root);
}

KDTree::Node* KDTree::buildTree(std::vector<std::vector<double>>& points,
                                int start, int end, int depth) {
    // Base case: no points to process
    if (start >= end){
        return nullptr;
    }

    // Calculate current dimension (cycles through 0 to K-1)
    int cd = depth % K;
    // Find median index
    int mid = start + (end - start) / 2;

    // Partition around median using nth_element
    // This puts the median at position 'mid' and partitions around it
    std::nth_element(points.begin() + start, points.begin() + mid,
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

bool KDTree::searchRecursive(Node* node, const std::vector<double>& point,
                             int depth, double eps = 1e-9) const {
    // Base case: reached a leaf (point not found)
    if (node == nullptr){
        return false;
    }
    // Check if points are equal within tolerance
    if (approxEqualPoint(node->point, point, eps))
        return true;
    // Calculate current dimension (cycles through 0 to K-1)
    int cd = depth % K;
    // Recurse down the appropriate subtree based on current dimension
    if (approxEqual(point[cd], node->point[cd], eps)) {
        return searchRecursive(node->left, point, depth + 1, eps) ||
               searchRecursive(node->right, point, depth + 1, eps);
    }
    if (point[cd] < node->point[cd]){
        return searchRecursive(node->left, point, depth + 1, eps);
    }
    return searchRecursive(node->right, point, depth + 1, eps);
}

void KDTree::printRecursive(Node* node, int depth) const {
    // Base case: null node, nothing to print
    if (node == nullptr) {
        return;
    }
    // Print indentation based on depth
    for (int i = 0; i < depth; ++i){
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

void KDTree::deleteRecursive(Node* node) {
    if (node == nullptr) return;    
    // Post-order traversal: delete children first
    deleteRecursive(node->left);
    deleteRecursive(node->right);
    delete node;
}

void KDTree::collectPoints(Node* node, std::vector<std::vector<double>>& points) const {
    if (node == nullptr) return;
    points.push_back(node->point);
    collectPoints(node->left, points);
    collectPoints(node->right, points);
}

void KDTree::writeCSVRecursive(Node* node, std::ofstream& out) const {
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

void KDTree::build(std::vector<std::vector<double>> points) {
    // Delete existing tree
    deleteRecursive(root);
    root = nullptr;
    // Build new tree
    if (!points.empty()) {
        K = points[0].size();
        root = buildTree(points, 0, points.size(), 0);
    }
}

void KDTree::insert(const std::vector<double>& point) {
    // Collect all existing points
    std::vector<std::vector<double>> points;
    collectPoints(root, points);
    // Add new point
    points.push_back(point);
    // Rebuild tree
    build(points);
}

bool KDTree::search(const std::vector<double>& point) const {
    return searchRecursive(root, point, 0);
}

void KDTree::print() const {
    printRecursive(root, 0);
}

bool KDTree::empty() const {
    return root == nullptr;
}

const KDTree::Node* KDTree::getRoot() {
    return root;
}

const KDTree::Node* KDTree::appendNode(const std::vector<double>& point) {
    Node* parsing_node = root;
    while(parsing_node->right != nullptr){
		parsing_node = parsing_node->right;
	}
    parsing_node->right = new KDTree::Node(point); //just to test the output file
    return parsing_node->right;
}

void KDTree::writeCSV(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    // Header: x0,x1,...,x(K-1),BelongingCluster
    for (size_t i = 0; i < K; ++i) {
        out << "x" << i;
        if (i < K - 1) out << ",";
    }
    out << ",BelongingCluster\n";

    writeCSVRecursive(root, out);

    out.close();
}
