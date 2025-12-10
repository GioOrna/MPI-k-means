#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>

#include "../include/csv_utils.hpp"
#include "../include/KDTree.hpp"

using namespace std;

/**
 * @brief Macro to execute a block of code only on a specific MPI rank.
 * @param x The rank on which to execute the code block.
 * \internal
 * Similar to OpenMP's single directive.
 */
#define MPI_single(x) if(rank == x)

/**
 * @brief Macro to execute a block of code only on the master MPI rank (rank 0).
 * \internal
 * Similar to OpenMP's master directive.
 */
#define MPI_master() MPI_single(0)

/**
 * @brief Evenly scatter a 2D vector of doubles among all MPI processes.
 * @param data The 2D vector to scatter (only needed on master process).
 * @param mpi_comm The MPI communicator.
 * @return The portion of the data assigned to the calling process.
 */
vector<vector<double>> MPI_evenlyScatterData(const vector<vector<double>>&,
										 MPI_Comm);

/**
 * @brief Generates random centroids within given bounds.
 * @param num_centroids Number of centroids to generate.
 * @param dim Data dimension.
 * @param max_values Maximum values for each dimension.
 * @param min_values Minimum values for each dimension.
 * @return A vector containing the generated centroids.
 */
vector<vector<double>> generate_centroids(const int num_centroids,
										 const size_t dim,
										 const vector<double>& max_values,
										 const vector<double>& min_values);

/**
 * @brief Flattens a 2D vector into a 1D vector.
 * @param mat The 2D vector to flatten.
 * @return The flattened 1D vector.
 */
vector<double> flatten(const vector<vector<double>>& mat);

/**
 * @brief Unflattens a 1D vector into a 2D vector.
 * @param flat The 1D vector to unflatten.
 * @param rows Number of rows in the resulting 2D vector.
 * @param cols Number of columns in the resulting 2D vector.
 * @return The unflattened 2D vector.
 */
vector<vector<double>> unflatten(const vector<double>& flat, int rows, int cols);

/**
 * @brief Broadcasts centroids from the sender rank to all other ranks.
 * @param centroids The centroids to broadcast.
 * @param rank The rank of the calling process.
 * @param sender_rank The rank that holds the centroids to broadcast.
 * @param mpi_comm The MPI communicator.
 * @return void
 */
void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank, MPI_Comm mpi_comm);

/**
 * @brief Compute the closest centroid to the point.
 * @param point The point to be analyzed.
 * @param centroids The centroids to be analyzed.
 * @return The index of the closest centroid.
 */
int closest_centroid(const vector<double>& point,
					 const vector<vector<double>>& centroids);

/**
 * @brief Puts the cluster field of all nodes in the specified subtree equal to the specified value.
 * @param node The root of the subtree.
 * @param cluster The index of the cluster we want to assign.
 * @return void
 */
void assign_to_cluster(const KDTree::Node* node, const int cluster);

/**
 * @brief Calculates the midpoint of the subtree specified.
 * @param node The root of the subtree.
 * @return A vector of double containing the mipoint coordinates
 */
vector<double> midpoint(const KDTree::Node* node);
bool is_farther(const KDTree::Node* node,
				const vector<vector<double>>& centroids,
				const int c1,
				const int c2);

/**
 * @brief An algorithm to calculate new centroids coordinated based on KD-tree.
 * @param u root of the subtree we want to analyze.
 * @param candidates Vector of possible clusters to assign to the subtree.
 * @param ccentroids 2D vector of all centroids.
 * @param wgtCent 2D vector of sum of coordinates of points assigned to each centroid.
 * @param counts Vector of number of points assigned to each centroid.
 * @return void
 */
void filter(const KDTree::Node* u, vector<int> candidates, vector<vector<double>>& centroids, 
		    vector<vector<double>>& wgtCent, vector<int>& counts);

/**
 * @brief Calculate squared Euclidean distance between two points.
 * @param a First point.
 * @param b Second point.
 * @return Squared Euclidean distance.
 */
double distance(vector<double> a, vector<double> b);

/**
 * @brief Sums coordinates of all nodes in the subtree and keeps count of how many have been summed.
 * @param wgtCent Vector to which coordinates will be summed (it won't be resetted).
 * @param counts Int where the number of nodes will be summed (it won't be resetted).
 * @param node Root of the subtree.
 * @return void
 */
void assign_subtree_to_cluster(vector<double>& wgtCent, int& counts,const KDTree::Node* node);

/**
 * @brief Adds the centroids as points in the tree with new values in the cluster field to distinguish them from the other points.
 * @param tree The tree where we want to add the centroids
 * @param centroids Centroids we want to add.
 * @return void
 */
void insert_centroid_in_tree(KDTree& tree, vector<vector<double>>& centroids);

/**
 * @brief Sequential K-means algorithm, will update centroids with the new calculated centroids
 * @param centroids Centroids we want to update.
 * @param node The root of the tree we want to use. 
 */
void Kmeans_sequential(vector<vector<double>>& centroids,
			const KDTree::Node* node);
int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm mpi_comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	vector<vector<double>> data;
	vector<vector<double>> centroids;
	MPI_master() {
		if(argc != 3) {
			cerr << "Usage: " << argv[0] << " <path_to_file> <number_of_centroids>" << endl;
			MPI_Abort(mpi_comm, 1);
		}
		int num_centroids;
		try {
			num_centroids = std::stoul(argv[2]);
			if(num_centroids < 1){
				throw std::invalid_argument("Number of centroids must be at least 1.");
			}
		} catch(const std::exception& e) {
			cerr << "Invalid number of centroids: " << argv[2] << endl;
			MPI_Abort(mpi_comm, 1);
		}

		// File loading and parsing.
		// It isn't worth to concurrently load and parse the file; the required
		// operations for each process would be:
		// - open and read the file (a csv file is not splittable a priori), so
		//   we would lose the advantage of parallelism.
		// - parse the data
		// - send/receive the information about what slice of data each process
		//   should work on.
		// Finally, it's easier for code readability and ease of coding to just
		// do it on the master process and then scatter the data to the other
		// processes.
		vector<double> max_values;
		vector<double> min_values;
		try {
			data = readCSV(argv[1], true, max_values, min_values);
		} catch(const exception& e) {
			cerr << "Error reading CSV file: " << e.what() << endl;
			MPI_Abort(mpi_comm, 1);
		}
			centroids = generate_centroids(
			num_centroids, data[0].size(), max_values, min_values);		
	}
	MPI_Barrier(mpi_comm); // synchronize all ranks
	broadcast_centroids(centroids, rank, 0, mpi_comm); 	// Broadcast centroids to all processes.
	
	//rank0 iterate over points in the tree and check closest centroid
	//if two points have different closest centroids, split the tree and send to the other ranks
	//(if there are ranks available)
	//if all points have the same closest centroid, send back summation andd count to rank0
	//recurseively do the same on each subtree until no ranks are available or no more splits are possible
	if(rank == 0){
		KDTree tree(data);
		const KDTree::Node* root = tree.getRoot();
		Kmeans_sequential(centroids, root);
		insert_centroid_in_tree(tree, centroids); // to test the output file
		tree.writeCSV("output_cluster.csv");
	}
	MPI_Barrier(mpi_comm); // synchronize all ranks
	
	// Scatter the data evenly among all processes.
	//vector<vector<double>> local_data = MPI_evenlyScatterData(data, mpi_comm);

	// All processes can now work on their portion of the data
	
	MPI_Finalize();
}

void Kmeans_sequential(vector<vector<double>>& centroids,
			const KDTree::Node* node){
	vector<int> candidates;
		candidates.resize(centroids.size());
		vector<vector<double>> wgtCent;
		wgtCent.resize(centroids.size());
		for(auto& cent:wgtCent){
			cent.resize(node->getPoint().size(), 0.0);
		}
		vector<int> counts;
		counts.resize(centroids.size());
		vector<vector<double>> old_centroids = centroids;
		bool stabilized = false;
		while(!stabilized){
			stabilized = true;
			for(int i=0; i<candidates.size(); ++i){ // reset candidates
				candidates[i] = i;
			}
			for(auto& cent: wgtCent){ //reset weighted centroids
				for(auto& val: cent){
					val = 0.0;
				}
			}
			for(int& cnt: counts){ // reset counts
				cnt = 0;
			}
			filter(node, candidates, centroids, wgtCent, counts);
			for(int i=0; i<centroids.size(); ++i){
				if(counts[i]>0){
					for(int j=0; j<centroids[i].size(); ++j){
						centroids[i][j] = wgtCent[i][j] / counts[i];
						if(abs(centroids[i][j] - old_centroids[i][j]) > 1e-9){ // check for convergence
							stabilized = false;
							break;
						}
					}
				}
			}
			old_centroids = centroids;		
		}
}

void insert_centroid_in_tree(KDTree& tree, vector<vector<double>>& centroids){
	for(int i=0; i<centroids.size(); i++){
		const KDTree::Node* node = tree.appendNode(centroids[i]);
		node->setCluster(centroids[i].size()+1+i);
	}
}

void filter(const KDTree::Node* u, vector<int> candidates, vector<vector<double>>& centroids, 
		    vector<vector<double>>& wgtCent, vector<int>& counts){
	if(u == nullptr){
		return;
	}
	if(u->getLeft() == nullptr && u->getRight() == nullptr){
		int c = closest_centroid(u->getPoint(), centroids);
		candidates.clear();
		candidates.push_back(c);
		for(int i=0; i<wgtCent[c].size(); ++i){ // for each dimension sum the coordinate
			wgtCent[c][i] += u->getPoint()[i];
		}
		counts[c] += 1;
		assign_to_cluster(u, c);
		return;
	}else{
		int c = closest_centroid(midpoint(u), centroids);
		for(int i = candidates.size() - 1; i >= 0; --i){
    		if(c != candidates[i] && is_farther(u, centroids, candidates[i], c)){
        		candidates.erase(candidates.begin() + i);
    		}
		}

		if(candidates.size()==1){
			assign_subtree_to_cluster(wgtCent[c], counts[c], u); //sum coordinates and update counts
			assign_to_cluster(u, c);
			return;
		}
		else{
			filter(u->getLeft(), candidates, centroids, wgtCent, counts);
			filter(u->getRight(), candidates, centroids, wgtCent, counts);
			u->setCluster(closest_centroid(u->getPoint(), centroids)); //assign to closest centroid
		}
	}
	return;
}

void assign_subtree_to_cluster(vector<double>& wgtCent, int& counts, const KDTree::Node* node){
	for(int i=0; i<wgtCent.size(); i++){
		wgtCent[i] += node->getPoint()[i];
	}
	counts++;
	if(node->getLeft() != nullptr){
		assign_subtree_to_cluster(wgtCent, counts, node->getLeft());
	}
	if(node->getRight() != nullptr){
		assign_subtree_to_cluster(wgtCent, counts, node->getRight());
	}	
	return;
}


void midpoint_sum(const KDTree::Node* node, vector<double>& sum, int& count){
    if (node==nullptr){
		return;
	}
	for (size_t i = 0; i < sum.size(); i++){
		sum[i] += node->getPoint()[i];
	}
    count += 1;

    if (node->getLeft()) { // if left child exists
        midpoint_sum(node->getLeft(), sum, count);
    }
    if (node->getRight()) { // if right child exists
        midpoint_sum(node->getRight(), sum, count);
    }
    return;
}

vector<double> midpoint(const KDTree::Node* node) {
	vector<double> sum(node->getPoint().size(), 0.0);
	int count = 0;
	midpoint_sum(node, sum, count);
    for(int i=0;i<sum.size();++i){
		sum[i] /= count;
	}
    return sum;
}


//recursive function to check if all points in the subtree are closer to centroid c1 than c2
bool is_farther(const KDTree::Node* node,
				const vector<vector<double>>& centroids,
				const int c1,
				const int c2){
	if(node == nullptr)
		return true;
	
	double dist1 = distance(node->getPoint(), centroids[c1]);
	double dist2 = distance(node->getPoint(), centroids[c2]);
	if (dist1 > dist2 + 1e-9){
		return false;
	}
	bool left = is_farther(node->getLeft(), centroids, c1, c2);
	bool right = is_farther(node->getRight(), centroids, c1, c2);

	return left && right;
}

double distance(vector<double> a, vector<double> b){
	double dist = 0.0;
	for(size_t j = 0; j < a.size(); ++j){ //iterate over dimensions
		dist += (a[j] - b[j]) * (a[j] - b[j]); //sum (over all dimensions) of distance squared
	}
	return dist;
}

void assign_to_cluster(const KDTree::Node* node, const int cluster){
		node->setCluster(cluster); //add cluster info to point
		if(node->getLeft() != nullptr){
			assign_to_cluster(node->getLeft(), cluster);
		}
		if(node->getRight() != nullptr){
			assign_to_cluster(node->getRight(), cluster);
		}	
}

int closest_centroid(const vector<double>& point,
					 const vector<vector<double>>& centroids){
	double min_dist = std::numeric_limits<double>::max(); //set distance to max possible
	int closest = -1;
	for(size_t i = 0; i < centroids.size(); ++i){ //iterate over centroids
		double dist = 0.0;
		for(size_t j = 0; j < point.size(); ++j){ //iterate over dimensions
			dist += (point[j] - centroids[i][j]) * (point[j] - centroids[i][j]); //sum (over all dimensions) of distance squared
		}
		if(dist < min_dist){
			min_dist = dist;
			closest = i;
		}
	}
	return closest;
}


void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank, MPI_Comm mpi_comm) {
	// Broadcast centroids to all processes.
	vector<double> flat_centroids;
	int rows, cols;
	if(rank==sender_rank){
		flat_centroids = flatten(centroids);
		rows = centroids.size();
		cols = centroids[0].size();
	}
	MPI_Bcast(&rows, 1, MPI_INT, sender_rank, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, sender_rank, MPI_COMM_WORLD);
	if(rank!=sender_rank){
		flat_centroids.resize(rows * cols);
	}
    MPI_Bcast(flat_centroids.data(), flat_centroids.size(), MPI_DOUBLE, sender_rank, MPI_COMM_WORLD);
	centroids = unflatten(flat_centroids, rows, cols);
	return;
}

vector<vector<double>> generate_centroids(const int num_centroids,
										 const size_t dim,
										 const vector<double>& max_values,
										 const vector<double>& min_values) {
	random_device rd;                // non-deterministic seed
    mt19937 gen(rd());               // Mersenne Twister engine
    vector<uniform_real_distribution<double>> dist;  // ranges
	dist.resize(dim);
	for(size_t i = 0; i < dim; ++i) {  // for each dimension define max and min
		dist[i] = std::uniform_real_distribution<double>(min_values[i], max_values[i]);
	}
	vector<vector<double>> centroids;
	centroids.resize(num_centroids); 
	for(size_t i = 0; i < num_centroids; ++i) { // for each centroid
		centroids[i].resize(dim);
		for(size_t j = 0; j < dim; ++j){  // for each dimension
			centroids[i][j] = dist[j](gen);
		}
	}
	return centroids;
}	

// Flatten a 2D vector into a 1D vector
vector<double> flatten(const vector<vector<double>>& mat) {
    vector<double> flat;
    for (const auto& row : mat) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Unflatten a 1D vector into a 2D vector
vector<vector<double>> unflatten(const vector<double>& flat, int rows, int cols) {
    if (flat.size() != rows * cols) {
        throw invalid_argument("Size of flat vector does not match rows*cols");
    }
    vector<vector<double>> mat(rows, vector<double>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = flat[i * cols + j];
    return mat;
}

vector<vector<double>> MPI_evenlyScatterData(const vector<vector<double>>& data,
										 MPI_Comm mpi_comm) {
	int rank, size;
	MPI_Comm_rank(mpi_comm, &rank);
	MPI_Comm_size(mpi_comm, &size);

	int total_cols, per_process_rows, remainder;
	vector<double> flat_data;

	MPI_master() {
		int total_rows = data.size();
		total_cols = data[0].size();

		// Calculate the number of rows per process.
		per_process_rows = total_rows /size;
		remainder = total_rows % per_process_rows;

		// Flatten the data.
		flat_data.reserve(data.size() * total_cols);
		for(const auto& row : data)
			flat_data.insert(flat_data.end(), row.begin(), row.end());
	}

	// Send to all processes the data dimensions.
	MPI_Bcast(&per_process_rows, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&remainder, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&total_cols, 1, MPI_INT, 0, mpi_comm);

	// Buffer send and displacements.
	// Note: the master process gets more rows beacause it's a fake comunication
	// the one between master and itself (especially in case of clusters), so it
	// should be faster than the others (this is a naive optimization).
	int send_counts[size], displacements[size];

	for(int i = 0; i < size; ++i) {
		send_counts[i] = per_process_rows * total_cols;
		displacements[i] = i * per_process_rows * total_cols;

		// Master gets the remainder.
		if(i == 0)
			send_counts[i] += remainder * total_cols;
		else
			displacements[i] += remainder * total_cols;
	}

	MPI_master() {
		per_process_rows += remainder;
	}

	// Each process allocates its local data.
	vector<double> local_data(per_process_rows * total_cols);

	MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr,
				 send_counts,
				 displacements,
				 MPI_DOUBLE,
				 local_data.data(),
				 per_process_rows * total_cols,
				 MPI_DOUBLE,
				 0,
				 mpi_comm);

	// Reshape the local data into a 2D vector.
	vector<vector<double>> reshaped(per_process_rows,
									vector<double>(total_cols));
	for(int i = 0; i < per_process_rows; ++i)
		copy(local_data.begin()+i*total_cols,
			 local_data.begin()+(i+1)*total_cols,
			 reshaped[i].begin());

	return reshaped;
}
