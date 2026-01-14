#include "../include/MPI_utils.hpp"
#include "../include/kmeans.hpp"

void broadcast_centroids(vector<vector<double>>& centroids, int rank, int sender_rank) {
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

void gather_results(vector<vector<double>>& wgtCent, vector<int>& counts, int to_rank, int rank, int size){
	if(rank == to_rank){
		for(int i=0;i<size;i++){
			if(i!=rank){
				vector<vector<double>> wgtCent2;
				vector<int> counts2;
				receive_result(wgtCent2, counts2, i);
				for(int i=0;i<wgtCent.size();i++){
                    for(int j=0;j<wgtCent[i].size();j++){
                        wgtCent[i][j] = wgtCent[i][j]+wgtCent2[i][j];
                    }
                    counts[i] = counts[i] + counts2[i];
                }
			}
		}
	}else{
		send_result(wgtCent, counts, to_rank);
	}
}

void send_result(vector<vector<double>>& wgtCent, vector<int> counts, int to_rank){
    if(wgtCent.empty() || wgtCent[0].empty()){
        int rows = 0 ;
        int cols = 0;
        MPI_Send(&rows, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
        return;
    }
    vector<double> flat_wgtCent = flatten(wgtCent);
    int rows = wgtCent.size();
    int cols = wgtCent[0].size();
    int num = counts.size();
    MPI_Send(&rows, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&cols, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
    MPI_Send(flat_wgtCent.data(), flat_wgtCent.size(), MPI_DOUBLE, to_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&num, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
    MPI_Send(counts.data(), counts.size(), MPI_INT, to_rank, 0, MPI_COMM_WORLD);
}

void receive_result(vector<vector<double>>& wgtCent, vector<int>& counts, int sender_rank){
    vector<double> flat_wgtCent;
    int rows;
    int cols;
    int num;
    MPI_Recv(&rows, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&cols, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(rows <= 0 || cols <= 0){
        wgtCent.clear();
        counts.clear();
        return;
    }
    flat_wgtCent.resize(rows*cols);
    MPI_Recv(flat_wgtCent.data(), rows*cols, MPI_DOUBLE, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&num, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    counts.resize(num);
    MPI_Recv(counts.data(), num, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    wgtCent = unflatten(flat_wgtCent, rows, cols);
    return;
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
		flat_data = flatten(data);
	}

	// Send to all processes the data dimensions.
	MPI_Bcast(&per_process_rows, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&remainder, 1, MPI_INT, 0, mpi_comm);
	MPI_Bcast(&total_cols, 1, MPI_INT, 0, mpi_comm);

	// Buffer send and displacements.
	// Note: the master process gets more rows beacause it's a fake comunication
	// the one between master and itself (especially in case of clusters), so it
	// should be faster than the others (this is a naive optimization). This
	// implementation choice simplifies also the communication logic, indeed the
	// sent `remainder` variable is mainly used by the master process.
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

	return unflatten(local_data, per_process_rows, total_cols);
}

vector<int> MPI_gatherUnbalancedData(const vector<int>& local_clustering,
									 MPI_Comm comm) {
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	// Gather sizes of local clusterings.
	int local_size = local_clustering.size();
	vector<int> all_sizes(size);
	MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, comm);

	vector<int> displacements(size), full_clustering;
	MPI_master() {
		// Calculate displacements for gathering.
		displacements[0] = 0;
		for(int i = 1; i < size; ++i)
			displacements[i] = displacements[i - 1] + all_sizes[i - 1];

		// Gather all local clusterings to the master process.
		int total_size = 0;
		for(int sz : all_sizes)
			total_size += sz;
		full_clustering.resize(total_size);
	}

	MPI_Gatherv(local_clustering.data(),
				local_size,
				MPI_INT,
				rank == 0 ? full_clustering.data() : nullptr,
				all_sizes.data(),
				displacements.data(),
				MPI_INT,
				0,
				comm);

	return full_clustering;
}

vector<int> MPI_computeClustering(const vector<vector<double>>& data,
								  const vector<vector<double>>& centroids) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Broadcast centroids to all processes.
	vector<vector<double>> local_centroids = centroids;
	broadcast_centroids(local_centroids, rank, 0);

	// Scatter data among processes.
	vector<vector<double>> local_data = MPI_evenlyScatterData(data,
															  MPI_COMM_WORLD);

	// Each process computes its local clustering.
	vector<int> local_clustering(local_data.size());
	
	vector<int> all_candidates(centroids.size());
	for(int i=0;i<all_candidates.size();i++){
		all_candidates[i] = i;
	}
	for(size_t i = 0; i < local_data.size(); ++i)
		local_clustering[i] = closest_centroid(local_data[i], local_centroids, all_candidates);

	// Gather the local clusterings to the master process.
	return MPI_gatherUnbalancedData(local_clustering, MPI_COMM_WORLD);
}
