#include "../include/kmeans.hpp"
#include "../include/MPI_utils.hpp"

vector<vector<double>> kmeans_parallel(int rank, int size, vector<vector<double>>& centroids,
			std::vector<std::vector<double>> data_to_work){
	KDTree tree(data_to_work);
	const KDTree::Node* node = tree.getRoot();
	bool stabilized = false;
	vector<vector<double>> old_centroids = centroids;
	broadcast_centroids(centroids, rank, 0); // Broadcast centroids to all processes.
	if(centroids.empty()){
		return old_centroids;
	}
	while(!stabilized){
		if(rank==0){
			stabilized = true;
		}
		vector<int> candidates(centroids.size());
		vector<vector<double>> wgtCent(centroids.size(), vector<double>(node->getPoint().size(), 0.0));
        vector<int> counts(centroids.size(), 0);
		filter(node, candidates, centroids, wgtCent, counts);
		gather_results(wgtCent, counts, 0, rank, size); //now rank0 has the sum
		if(rank==0){
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
			if(stabilized == true){
				centroids.clear();
			}
		}
		broadcast_centroids(centroids, rank, 0); // Broadcast centroids to all processes.
		if(centroids.empty()){
			return old_centroids;
		}
	}
	return old_centroids;
}

void kmeans_sequential(vector<vector<double>>& centroids,
			std::vector<std::vector<double>> data_to_work){
	KDTree tree(data_to_work);
	const KDTree::Node* node = tree.getRoot();
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

void filter(const KDTree::Node* u, vector<int> candidates, vector<vector<double>>& centroids, 
		    vector<vector<double>>& wgtCent, vector<int>& counts){
	if(u == nullptr){
		return;
	}
	if(u->getLeft() == nullptr && u->getRight() == nullptr){
		int c = closest_centroid(u->getPoint(), centroids);
		for(int i=0; i<wgtCent[c].size(); ++i){ // for each dimension sum the coordinate
			wgtCent[c][i] += u->getPoint()[i];
		}
		counts[c] += 1;
		candidates.clear();
		candidates.push_back(c);
		return;
	}else{
		vector<double> midpoint = u->getSum();
		for(int i=0; i<midpoint.size(); i++){
			midpoint[i]/u->getCount();
		}
		int c = closest_centroid(midpoint, centroids);
		for(int i = candidates.size() - 1; i >= 0; --i){
    		if(c != candidates[i] && is_farther(u, centroids, candidates[i], c)){
        		candidates.erase(candidates.begin() + i);
    		}
		}

		if(candidates.size()==1){
			wgtCent[c] = u->getSum();
			counts[c] = u->getCount();
			return;
		}
		else{
			filter(u->getLeft(), candidates, centroids, wgtCent, counts);
			filter(u->getRight(), candidates, centroids, wgtCent, counts);
			c = closest_centroid(u->getPoint(), centroids);
			for(int i=0; i<wgtCent[c].size(); ++i){ // for each dimension sum the coordinate
				wgtCent[c][i] += u->getPoint()[i];
			}
			counts[c] += 1;
		}
	}
	return;
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

