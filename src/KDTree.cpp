#include "../include/KDTree.hpp"

// ---- approxEqualPoint ----
bool approxEqualPoint(const std::vector<double>& a,
                      const std::vector<double>& b,
                      double eps) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!approxEqual(a[i], b[i], eps))
            return false;
    return true;
}

// ---- Node methods ----

KDTree::Node::Node(const std::vector<double>& pt)
    : point(pt), left(nullptr), right(nullptr), cluster(-1) {}

std::vector<double> KDTree::Node::getPoint() const { return point; }
const KDTree::Node* KDTree::Node::getLeft() const { return left.get(); }
const KDTree::Node* KDTree::Node::getRight() const { return right.get(); }
void KDTree::Node::setCluster(int c) const { cluster = c; }

// ---- Validation ----

void KDTree::validatePoints(const std::vector<std::vector<double>>& points, 
                            size_t expectedDim) const {
    if (points.empty()) {
        throw std::invalid_argument("Cannot build tree with empty point set");
    }
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].size() != expectedDim) {
            throw std::invalid_argument("Point at index " + std::to_string(i) + 
                                      " has dimension " + std::to_string(points[i].size()) + 
                                      ", expected " + std::to_string(expectedDim));
        }
    }
}

// ---- KDTree methods ----

KDTree::KDTree() 
    : root(nullptr), K(0), node_count(0) {}

KDTree::KDTree(size_t dimension) 
    : root(nullptr), K(dimension), node_count(0) {
    if (dimension == 0) {
        throw std::invalid_argument("Dimension K must be at least 1");
    }
}

KDTree::KDTree(std::vector<std::vector<double>> points) 
    : root(nullptr), K(0), node_count(0) {
    if (!points.empty()) {
        K = points[0].size();
        if (K == 0) {
            throw std::invalid_argument("Points must have at least 1 dimension");
        }
        validatePoints(points, K);
        root = buildTree(points, 0, points.size(), 0);
        node_count = countNodes(root);
    }
}

std::unique_ptr<KDTree::Node> KDTree::buildTree(std::vector<std::vector<double>>& points,
                                                 int start, int end, int depth) {
    // Base case: no points to process
    if (start >= end) return nullptr;

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
    auto node = std::make_unique<Node>(points[mid]);
    // Recursively build left and right subtrees
    node->left = buildTree(points, start, mid, depth + 1);
    node->right = buildTree(points, mid + 1, end, depth + 1);
    return node;
}

bool KDTree::searchRecursive(const std::unique_ptr<Node>& node, 
                             const std::vector<double>& point,
                             int depth, double eps) const {
    // Base case: reached a leaf (point not found)
    if (node == nullptr) return false;

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
    if (point[cd] < node->point[cd]) {
        return searchRecursive(node->left, point, depth + 1, eps);
    }
    return searchRecursive(node->right, point, depth + 1, eps);
}

bool KDTree::insertRecursive(std::unique_ptr<Node>& node,
                            const std::vector<double>& point,
                            int depth,
                            double eps) {
    // Base case: empty space, insert here
    if (!node) {
        node = std::make_unique<Node>(point);
        node_count++;
        return true;
    }

    // Check if point already exists
    if (approxEqualPoint(node->point, point, eps)) {
        return false;  // Point already exists
    }

    int cd = depth % K;

    // Decide which subtree to traverse
    if (point[cd] < node->point[cd]) {
        return insertRecursive(node->left, point, depth + 1, eps);
    }
    return insertRecursive(node->right, point, depth + 1, eps);
}

const KDTree::Node* KDTree::findMin(const std::unique_ptr<Node>& node, 
                                   int dim, int depth) const {
    if (!node) return nullptr;

    int cd = depth % K;
    
    // If we're at the discriminating dimension, min is in left subtree
    if (cd == dim) {
        if (!node->left) return node.get();
        return findMin(node->left, dim, depth + 1);
    }

    // Otherwise, check all three nodes (current, left, right)
    const Node* minNode = node.get();
    
    if (node->left) {
        const Node* leftMin = findMin(node->left, dim, depth + 1);
        if (leftMin && leftMin->getPoint()[dim] < minNode->getPoint()[dim]) {
            minNode = leftMin;
        }
    }
    
    if (node->right) {
        const Node* rightMin = findMin(node->right, dim, depth + 1);
        if (rightMin && rightMin->getPoint()[dim] < minNode->getPoint()[dim]) {
            minNode = rightMin;
        }
    }
    
    return minNode;
}

// Function not tested completely yet.
// Right now, we need to improve, the use of getCluster. It is not used correctly, and 
// it is inefficient, we are copying cluster data unnecessarily. But like delete is not used 
// in our kmeans implementation, we can leave it for now and optimize later if needed.
bool KDTree::deleteRecursive(std::unique_ptr<Node>& node,
                            const std::vector<double>& point,
                            int depth,
                            double eps) {
    if (!node) return false;

    int cd = depth % K;

    // Check if this is the point to delete
    if (approxEqualPoint(node->point, point, eps)) {
        // Case 1: Node has right child
        if (node->right) {
            const Node* minNode = findMin(node->right, cd, depth + 1);
            node->point = minNode->getPoint();
            node->cluster = minNode->getCluster();
            return deleteRecursive(node->right, minNode->getPoint(), depth + 1, eps);
        }
        // Case 2: Node has only left child
        else if (node->left) {
            // Move left to right first
            node->right = std::move(node->left);
            const Node* minNode = findMin(node->right, cd, depth + 1);
            node->point = minNode->getPoint();
            node->cluster = minNode->getCluster();
            return deleteRecursive(node->right, minNode->getPoint(), depth + 1, eps);
        }
        // Case 3: Node is a leaf
        else {
            node = nullptr;
            node_count--;
            return true;
        }
    }

    // Recurse to find the point
    if (approxEqual(point[cd], node->point[cd], eps)) {
        // Could be in either subtree when coordinates are equal
        bool found = deleteRecursive(node->left, point, depth + 1, eps);
        if (!found) {
            found = deleteRecursive(node->right, point, depth + 1, eps);
        }
        return found;
    }
    if (point[cd] < node->point[cd]) {
        return deleteRecursive(node->left, point, depth + 1, eps);
    }
    return deleteRecursive(node->right, point, depth + 1, eps);
}

size_t KDTree::countNodes(const std::unique_ptr<Node>& node) const {
    if (!node) return 0;
    return 1 + countNodes(node->left) + countNodes(node->right);
}

int KDTree::getHeight(const std::unique_ptr<Node>& node) const {
    if (!node) return -1;
    int leftHeight = getHeight(node->left);
    int rightHeight = getHeight(node->right);
    return 1 + std::max(leftHeight, rightHeight);
}

void KDTree::printRecursive(const std::unique_ptr<Node>& node, int depth) const {
    // Base case: null node, nothing to print
    if (node == nullptr) return;
    
    // Print indentation based on depth
    for (int i = 0; i < depth; ++i) {
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

void KDTree::collectPoints(const std::unique_ptr<Node>& node, 
                           std::vector<std::vector<double>>& points) const {
    if (node == nullptr) return;
    points.push_back(node->point);
    collectPoints(node->left, points);
    collectPoints(node->right, points);
}

void KDTree::writeCSVRecursive(const std::unique_ptr<Node>& node, 
                               std::ofstream& out) const {
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
    if (points.empty()) {
        root.reset();  // Clears the unique_ptr
        node_count = 0;
        return;
    }
    
    // Validate all points have correct dimension
    size_t dim = points[0].size();
    if (dim == 0) {
        throw std::invalid_argument("Points must have at least 1 dimension");
    }
    validatePoints(points, dim);
    
    // If K is set and doesn't match, throw error
    if (K > 0 && K != dim) {
        throw std::invalid_argument("Point dimension " + std::to_string(dim) + 
                                  " doesn't match tree dimension " + std::to_string(K));
    }
    
    K = dim;
    root.reset();  // Clears the unique_ptr
    root = buildTree(points, 0, points.size(), 0);
    node_count = countNodes(root);
}

bool KDTree::insert(const std::vector<double>& point) {
    // Validate point is not empty
    if (point.empty()) {
        throw std::invalid_argument("Cannot insert empty point");
    }
    
    // If tree is uninitialized, set dimension from first point
    if (K == 0) {
        K = point.size();
    }
    
    // Validate dimension matches
    if (point.size() != K) {
        throw std::invalid_argument("Point dimension " + std::to_string(point.size()) + 
                                  " doesn't match tree dimension " + std::to_string(K));
    }
    
    return insertRecursive(root, point, 0, default_eps);
}

bool KDTree::deletePoint(const std::vector<double>& point) {
    if (K == 0) {
        throw std::invalid_argument("Cannot delete from tree with uninitialized dimension");
    }
    if (point.size() != K) {
        throw std::invalid_argument("Point dimension " + std::to_string(point.size()) + 
                                  " doesn't match tree dimension " + std::to_string(K));
    }
    return deleteRecursive(root, point, 0, default_eps);
}

bool KDTree::search(const std::vector<double>& point) const {
    if (K == 0) {
        throw std::invalid_argument("Cannot search in tree with uninitialized dimension");
    }
    if (point.size() != K) {
        throw std::invalid_argument("Point dimension " + std::to_string(point.size()) + 
                                  " doesn't match tree dimension " + std::to_string(K));
    }
    return searchRecursive(root, point, 0, default_eps);
}

void KDTree::print() const {
    if (empty()) {
        std::cout << "Empty tree\n";
        return;
    }
    printRecursive(root, 0);
}

bool KDTree::empty() const {
    return root == nullptr;
}

size_t KDTree::size() const {
    return node_count;
}

int KDTree::height() const {
    return getHeight(root);
}

const KDTree::Node* KDTree::getRoot() const {
    return root.get();
}

KDTree::Node* KDTree::appendNode(const std::vector<double>& point) {
    if (!root) {
        throw std::runtime_error("Cannot append to empty tree");
    }
    Node* parsing_node = root.get();
    while (parsing_node->right != nullptr) {
        parsing_node = parsing_node->right.get();
    }
    parsing_node->right = std::make_unique<Node>(point);  // just to test the output file
    node_count++;
    return parsing_node->right.get();
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
