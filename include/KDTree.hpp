#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

/**
 * @brief Compare two floating-point values with relative tolerance.
 * Useful for numerical stability and avoiding strict equality checks.
 * @param a First value
 * @param b Second value
 * @param eps Relative tolerance (default 1e-9)
 * @return True if values are approximately equal
 */
inline bool approxEqual(double a, double b, double eps = 1e-9) {
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff <= eps * scale;
}

/**
 * @brief Compare two points using element-wise approximate equality.
 * Checks if all corresponding coordinates are approximately equal within tolerance.
 * @param a First point
 * @param b Second point
 * @param eps Relative tolerance (default 1e-9)
 * @return True if all coordinates are approximately equal
 */
bool approxEqualPoint(const std::vector<double>& a,
                      const std::vector<double>& b,
                      double eps = 1e-9);

/**
 * @brief Balanced K-Dimensional tree using median-based splitting.
 * Dimension `K` is inferred from first insertion or build, or can be set explicitly.
 * Invariant: K == 0.  KDTree is uninitialized; all points in tree have dimension K.
 */
class KDTree {
public:
    /**
     * @brief KD-tree node.
     * Contains a point, pointers to left and right children, and optional cluster assignment.
     */
    struct Node {
    private:
        std::vector<double> point;         ///< Point coordinates in K dimensions
        std::unique_ptr<Node> left;        ///< Left subtree (points with smaller splitting coordinate)
        std::unique_ptr<Node> right;       ///< Right subtree (points with larger splitting coordinate)
        mutable int cluster;               ///< Optional cluster assignment

        friend class KDTree;

    public:
        /**
         * @brief Construct node from point.
         * @param pt Coordinates of the point
         */
        Node(const std::vector<double>& pt);

        /**
         * @brief Get the point stored in the node.
         * @return Copy of the point coordinates
         */
        std::vector<double> getPoint() const;

        /**
         * @brief Get the left child node.
         * @return Pointer to left child (const)
         */
        const Node* getLeft() const;

        /**
         * @brief Get the right child node.
         * @return Pointer to right child (const)
         */
        const Node* getRight() const;

        /**
         * @brief Set cluster assignment for this node.
         * @param c Cluster ID
         */
        void setCluster(int c) const;

        /**
         * @brief Get cluster assignment.
         * @return Cluster ID (default -1)
         */
        int getCluster() const { return cluster; }
    };

private:
    static constexpr double default_eps = 1e-9; ///< Default epsilon for approximate comparisons
    std::unique_ptr<Node> root;                 ///< Root node of the tree
    size_t K;                                   ///< Dimensionality (0 if uninitialized)
    size_t node_count;                          ///< Total number of nodes

    /**
     * @brief Validate that all points have the expected dimensionality.
     *
     * @param points Vector of points
     * @param expectedDim Expected dimension
     * @throw std::invalid_argument if a point does not match expected dimension
     */
    void validatePoints(const std::vector<std::vector<double>>& points,
                        size_t expectedDim) const;

    /**
     * @brief Build balanced KD-tree recursively from points.
     * @param points Vector of points (mutable for median partition)
     * @param start Start index
     * @param end End index (exclusive)
     * @param depth Current recursion depth
     * @return Pointer to subtree root
     */
    std::unique_ptr<Node> buildTree(std::vector<std::vector<double>>& points,
                                    int start, int end, int depth);

    /**
     * @brief Recursively search for a point in the subtree.
     * @param node Current subtree root
     * @param point Point to search
     * @param depth Current depth
     * @param eps Approximate equality tolerance
     * @return True if point exists in subtree
     */
    bool searchRecursive(const std::unique_ptr<Node>& node,
                         const std::vector<double>& point,
                         int depth,
                         double eps = default_eps) const;

    /**
     * @brief Recursively insert a point into the subtree.
     * @param node Current subtree root
     * @param point Point to insert
     * @param depth Current depth
     * @param eps Approximate equality tolerance
     * @return True if point was inserted; false if it already exists
     */
    bool insertRecursive(std::unique_ptr<Node>& node,
                         const std::vector<double>& point,
                         int depth,
                         double eps = default_eps);

    /**
     * @brief Count nodes in a subtree recursively.
     * @param node Subtree root
     * @return Number of nodes
     */
    size_t countNodes(const std::unique_ptr<Node>& node) const;

    /**
     * @brief Compute subtree height recursively.
     * @param node Subtree root
     * @return Height of subtree (-1 if empty)
     */
    int getHeight(const std::unique_ptr<Node>& node) const;

    /**
     * @brief Print subtree recursively to standard output.
     * @param node Subtree root
     * @param depth Current depth (for indentation)
     */
    void printRecursive(const std::unique_ptr<Node>& node, int depth) const;

    /**
     * @brief Collect all points in the subtree.
     * @param node Subtree root
     * @param points Output vector of points
     */
    void collectPoints(const std::unique_ptr<Node>& node,
                       std::vector<std::vector<double>>& points) const;

    /**
     * @brief Write all subtree points to CSV recursively.
     * @param node Subtree root
     * @param out Opened output file stream
     */
    void writeCSVRecursive(const std::unique_ptr<Node>& node,
                           std::ofstream& out) const;

public:
    /**
     * @brief Construct an empty KD-tree with uninitialized dimension.
     */
    KDTree();

    /**
     * @brief Construct an empty KD-tree with a fixed dimension.
     * @param dimension Number of dimensions (must be > 0)
     * @throw std::invalid_argument if dimension is 0
     */
    KDTree(size_t dimension);

    /**
     * @brief Construct a KD-tree from a set of points.
     * @param points Vector of points
     * @throw std::invalid_argument if points have inconsistent dimensions
     */
    KDTree(std::vector<std::vector<double>> points);

    ~KDTree() = default;

    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;

    KDTree(KDTree&& other) noexcept = default;
    KDTree& operator=(KDTree&& other) noexcept = default;

    /**
     * @brief Build a balanced KD-tree from a set of points.
     * If `points` is empty, the tree is fully reset and `K` is cleared.
     * @param points Vector of points
     * @throw std::invalid_argument if points have inconsistent dimensions
     */
    void build(std::vector<std::vector<double>> points);

    /**
     * @brief Insert a single point into the KD-tree.
     * Initializes `K` if the tree is uninitialized.
     * @param point Point to insert
     * @return True if inserted, false if it already exists
     * @throw std::invalid_argument if point dimension doesn't match `K`
     */
    bool insert(const std::vector<double>& point);

    /**
     * @brief Search for a point in the KD-tree.
     * @param point Query point
     * @return True if found
     * @throw std::logic_error if tree dimension is uninitialized
     * @throw std::invalid_argument if point dimension doesn't match `K`
     */
    bool search(const std::vector<double>& point) const;

    /**
     * @brief Print the tree structure to standard output.
     */
    void print() const;

    /**
     * @brief Check whether the tree is empty.
     * @return True if empty, false otherwise
     */
    bool empty() const;

    /**
     * @brief Get the number of nodes in the tree.
     * @return Node count
     */
    size_t size() const;

    /**
     * @brief Get the height of the tree.
     * @return Height (-1 if empty)
     */
    int height() const;

    /**
     * @brief Get the root node.
     * @return Pointer to root (const)
     */
    const Node* getRoot() const;

    /**
     * @brief Append a point without maintaining KD-tree invariants (for testing).
     * Used to output CSV for debugging.
     * @param point Point to append
     * @return Pointer to newly appended node
     * @throw std::runtime_error if tree is empty
     */
    Node* appendNode(const std::vector<double>& point);

    /**
     * @brief Write all tree points to a CSV file.
     * Output format: x0,x1,...,x(K-1),BelongingCluster
     * @param filename Name of output CSV file
     * @throw std::runtime_error if file cannot be opened
     */
    void writeCSV(const std::string& filename) const;
};

#endif
