#include "../include/MPI_utils.hpp"

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
