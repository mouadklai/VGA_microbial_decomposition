#ifndef _UTILS_H_
#define _UTILS_H_


#include <stdbool.h>

typedef struct sym_csr{
    int n_rows;
    int n_nz;
    int* row_ptrs;
    int* col_indices;
    double* values;
} sym_csr; 

typedef struct Sparse_CSR{
    int n_rows;
    int n_nz;
    int* row_ptrs;
    int* col_indices;
    double* values;
} Sparse_CSR; 


void bubbleSort(int arr[], int n);
bool isNeighbor6(int i, int j, int k, int new_i, int new_j, int new_k);
void create_implicit_diffusion_matrix_from_adjacency(
    const double Dc,
    const double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    int n_nz,
    Sparse_CSR* A_csr,
    Sparse_CSR* M_inv
);

void sparseProduct_parallel(
    Sparse_CSR* A,
    double* vec,
    double* res
);
double scalar_product_parallel(double* vec1, double* vec2, int nb_col);

void iterer_GCP_parallel(
    Sparse_CSR* A,
    Sparse_CSR* Minv,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    );

double* gradient_conjugue_preconditionne_parallel(Sparse_CSR* A,
                                                 Sparse_CSR* Minv,
                                                 double* b, double* Auk, double* uk, double* rk, double* zk, double* pk,double* Apk, double tolerance, int max_itr);


void construct_diff_sym(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    int n_nz,
    sym_csr* A,
    sym_csr* M
    );

void multiply_sym_csr_Vector(sym_csr* A, double* vector, double* result);

void iterer_GCP_sym_csr(
    sym_csr* A,
    sym_csr* Minv,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    );

double* GCP_sym_csr_parallel(
    sym_csr* A,
    sym_csr* Minv,
    double* b, 
    double* Auk, 
    double* uk, 
    double* rk, 
    double* zk, 
    double* pk,
    double* Apk, 
    double tolerance, 
    int max_itr);

void perform_bacterial_actions(
    double **M,
    int nb_nodes,
    double voxelVolume,
    double deltat,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab);


void Asynchronous_Transformation(
    double **M,
    int nb_nodes,
    double voxelVolume,
    double deltat, 
    double rho, 
    double mu, 
    double rho_m, 
    double vfom, 
    double vsom, 
    double vdom, 
    double kab);

void mass_sum(double **M,int nb_nodes,double history[5]);


int load3DImage(char* path,int*** threeDArray,int length,int width,int height);
int get_coordinates(int*** threeDArray,int** nodesCoordinates,int* nodes_degree,int** nodes_Adjacency, int length,int width,int height);

void filling_pore_space(int** nodesCoordinates, double **M, int nb_nodes,double DOM_totalMass, char* biomass_file, double microorganismWeight);
void simulate_biologie(
    double **B,
    int** nodes_Adjacency,
    int* nodes_degree,
    int nb_nodes, 
    int n_nz, 
    double Dc, 
    double tolerance,
    int max_itr,
    double DT, 
    double dt,
    double voxelVolume,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab,
    int typetr,
    char* outputFile);

void simulate_biologie_MK(
    double **B,
    int** nodes_Adjacency,
    int* nodes_degree,
    int nb_nodes, 
    int n_nz, 
    double Dc, 
    double tolerance,
    int max_itr,
    double DT, 
    double dt,
    double dt_tr,
    double voxelVolume,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab,
    int typetr,
    char* outputFile);

void simulate_transformation_diffusion_sym_csr(
    double **B,
    int** nodes_Adjacency,
    int* nodes_degree,
    int nb_nodes, 
    int n_nz, 
    double Dc, 
    double tolerance,
    int max_itr,
    double DT, 
    double dt,
    double voxelVolume,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab,
    int typetr,
    char* outputFile);

void multiply_exp_diff_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result);

void simulate_biologie_exp_MK(
    double **B,
    int** nodes_Adjacency,
    int* nodes_degree,
    int nb_nodes, 
    double Dc, 
    double tolerance,
    int max_itr,
    double DT, 
    double dt,
    double voxelVolume,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab,
    int typetr,
    char* outputFile);
    
void Asynchronous_Transformation_diff_MK(
    int nb_nodes,
    double **M,
    int **nodes_Adjacency,
    int *nodes_degree,
    double voxelVolume,
    double Dc,
    double rho, 
    double mu, 
    double rho_m, 
    double vfom, 
    double vsom, 
    double vdom, 
    double kab,
    double deltat);


void perform_bacterial_actions_diff_MK(
    int nb_nodes,
    double **M,
    int **nodes_Adjacency,
    int *nodes_degree,
    double voxelVolume,
    double Dc,
    double rho, 
    double mu, 
    double rho_m, 
    double vfom, 
    double vsom, 
    double vdom, 
    double kab,
    double deltat);


void simulate_biologie_MK_prime(
    double **B,
    int** nodes_Adjacency,
    int* nodes_degree,
    int nb_nodes, 
    int n_nz, 
    double Dc, 
    double tolerance,
    int max_itr,
    double DT, 
    double dt,
    double dt_tr,
    double voxelVolume,
    double rho,
    double mu,
    double rho_m,
    double vfom,
    double vsom,
    double vdom,
    double kab,
    int typetr,
    char* outputFile);





void multiply_invCond_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result);

void multiply_diff_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result);

void iterer_GCP_MK(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    );



double* GCP_MK(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* b, 
    double* Auk, 
    double* uk, 
    double* rk, 
    double* zk, 
    double* pk,
    double* Apk, 
    double tolerance, 
    int max_itr);

#endif