#include "utils.h"
#include <locale.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>


void bubbleSort(int arr[], int n) {
    int swapped;
    do {
        swapped = 0;
        for (int i = 1; i < n; i++) {
            if (arr[i - 1] > arr[i]) {
                // Swap arr[i-1] and arr[i]
                int temp = arr[i - 1];
                arr[i - 1] = arr[i];
                arr[i] = temp;
                swapped = 1;
            }
        }
    } while (swapped);
}


bool isNeighbor6(int i, int j, int k, int new_i, int new_j, int new_k) {
    int dx = i - new_i;
    int dy = j - new_j;
    int dz = k - new_k;
    // Check for 6-connectivity
    return ((dx == 1 || dx == -1) && dy == 0 && dz == 0) ||
           (dx == 0 && (dy == 1 || dy == -1) && dz == 0) ||
           (dx == 0 && dy == 0 && (dz == 1 || dz == -1));
}



void create_implicit_diffusion_matrix_from_adjacency(
    const double Dc,
    const double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    int n_nz,
    Sparse_CSR* A_csr,
    Sparse_CSR* M_inv
){
    
    double theta_ij = Dc*dt;
    A_csr->n_rows = nb_nodes;
    A_csr->n_nz = n_nz;
    A_csr->col_indices = (int *) calloc(A_csr->n_nz,sizeof(int));
    A_csr->values = (double *) calloc(A_csr->n_nz,sizeof(double));
    A_csr->row_ptrs = (int *) calloc(A_csr->n_rows+1,sizeof(int));

    int nz_id=0;
    for (int i = 0 ; i<A_csr->n_rows;i++){
        A_csr->row_ptrs[i]=nz_id;
        for (int m=0;m<7;m++){
            int j = nodes_Adjacency[i][m];
            if (j!=-1){
                if (j!=i){
                    A_csr->col_indices[nz_id] = j;
                    A_csr->values[nz_id] = -theta_ij;
                    nz_id++;
                }else if (j==i)
                {
                    A_csr->col_indices[nz_id] = i;
                    A_csr->values[nz_id] = 1+nodes_degree[i]*theta_ij;
                    nz_id++;
                }
            }
        }
    }
    A_csr->row_ptrs[A_csr->n_rows] = nz_id;
    if (nz_id!=A_csr->n_nz){
        printf("there is a problem!");
    }

    M_inv->n_rows = nb_nodes;
    M_inv->n_nz = nb_nodes;
    M_inv->col_indices = calloc(M_inv->n_nz,sizeof(int));
    M_inv->values = calloc(M_inv->n_nz,sizeof(double));
    M_inv->row_ptrs = calloc(M_inv->n_rows+1,sizeof(int));

    nz_id = 0;
    for (int i=0;i<M_inv->n_rows;i++){
        M_inv->row_ptrs[i]=nz_id;
        M_inv->col_indices[nz_id] = i;
        M_inv->values[nz_id] = 1.0/(1+nodes_degree[i]*theta_ij);
        nz_id++;
    }
    M_inv->row_ptrs[M_inv->n_rows] = nz_id;
};






void sparseProduct_parallel(
    Sparse_CSR* A,
    double* vec,
    double* res
){
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            double private_accumulator = 0.0;
            int nz_start = A->row_ptrs[i];
            int nz_end = A->row_ptrs[i + 1];
            #pragma omp simd reduction(+:private_accumulator)
            for (int nz_id = nz_start; nz_id < nz_end; nz_id++) {
                int j = A->col_indices[nz_id];
                double val = A->values[nz_id];
                private_accumulator += val * vec[j];
            }
            res[i] = private_accumulator;
        }
    }
}



double scalar_product_parallel(double* vec1, double* vec2, int nb_col) {
    double res = 0.0;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for reduction(+:res)
        for (int i = 0; i < nb_col; i++) {
            res += vec1[i] * vec2[i];
        }
    }
    return res;
}

void iterer_GCP_parallel(
    Sparse_CSR* A,
    Sparse_CSR* Minv,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    ) {
    
    sparseProduct_parallel(A, pk, Apk);
    double alpha_k_num = scalar_product_parallel(rk, zk, A->n_rows);
    double alpha_k_den =  scalar_product_parallel(Apk, pk, A->n_rows);
    double alpha_k = alpha_k_num / alpha_k_den;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            uk[i] += alpha_k * pk[i];
            rk[i] -=  alpha_k * Apk[i];
        }
    }

    sparseProduct_parallel(Minv, rk, zk);
    double beta_k = scalar_product_parallel(rk, zk, A->n_rows);
    beta_k = beta_k / alpha_k_num;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            pk[i] = zk[i] + beta_k * pk[i];
        }
    }
}

double* gradient_conjugue_preconditionne_parallel(Sparse_CSR* A,
                                                 Sparse_CSR* Minv,
                                                 double* b, double* Auk, double* uk, double* rk, double* zk, double* pk,double* Apk, double tolerance, int max_itr) {


    sparseProduct_parallel(A, uk, Auk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            rk[i] = b[i] - Auk[i];
        }
    }

    sparseProduct_parallel(Minv, rk, zk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            pk[i] = zk[i];
        }
    }

    int k = 0;
    while (k < max_itr) {
        printf("%d\n",k);
        // Calculate rk, zk, pk, and uk using the iterer_GCP function
        iterer_GCP_parallel(A, Minv, rk, zk, pk, uk,Apk);

        double sum_rk_squared = 0.0;
        #pragma omp parallel num_threads(6)
        {
            #pragma omp for reduction(+:sum_rk_squared)
            for (int i = 0; i < A->n_rows; i++) {
                sum_rk_squared += rk[i] * rk[i];
            }
        }
        if (sum_rk_squared < tolerance*tolerance) {
            return uk;
        }

        k++;
    }
    printf("not converged ! ");
    return b;
}

/*------- Using only the triangular part of the matrix ----- */

void construct_diff_sym(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    int n_nz,
    sym_csr* A,
    sym_csr* M
    )
{
    A->n_nz = (n_nz-nb_nodes)/2 + nb_nodes;
    A->n_rows = nb_nodes;
    A->values = (double *)calloc(A->n_nz,sizeof(double));
    A->col_indices = (int *)calloc(A->n_nz,sizeof(int));
    A->row_ptrs = (int *)calloc(A->n_rows+1,sizeof(int));
    M->n_nz = nb_nodes;
    M->n_rows = nb_nodes;
    M->values = (double *)calloc(M->n_nz,sizeof(double));
    M->col_indices = (int *)calloc(M->n_nz,sizeof(int));
    M->row_ptrs = (int *)calloc(M->n_rows+1,sizeof(int));
    double theta_ij = Dc*dt;
    int nnz=0;
    A->row_ptrs[0] = 0; // Initialize the first element of row_ptr
    for (int i = 0; i < A->n_rows; ++i) {
        for (int m=0;m<7;m++){
            int j = nodes_Adjacency[i][m];
            if (j!=-1) {
                if (j < i){                
                    A->values[nnz] = -theta_ij;
                    A->col_indices[nnz] = j;
                    ++nnz;
                }
                else if (j==i){
                    A->values[nnz] = 1+nodes_degree[i]*theta_ij;
                    A->col_indices[nnz] = j;
                    ++nnz;
                }
            }
        }
        A->row_ptrs[i+1]=nnz;
    }
    if (A->n_nz != nnz){
        printf("nnz = %d, A.nnz = %d\n",nnz,A->n_nz);
        printf("There is a problem!\n");
    }
    nnz=0;
    M->row_ptrs[0] = 0; // Initialize the first element of row_ptr
    for (int i = 0; i < M->n_rows; ++i) {       
        M->values[nnz] = 1.0/(1+nodes_degree[i]*theta_ij);
        M->col_indices[nnz] = i;
        ++nnz;
        M->row_ptrs[i+1]=nnz;
    }
}

void multiply_diff_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result)
{
    double theta_ij = Dc*dt;
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; ++i) {
            double private_accumulator = 0.0;
            #pragma omp simd reduction(+:private_accumulator)
            for (int m=0;m<7;m++){
                int j = nodes_Adjacency[i][m];
                if (j!=-1) {
                    if (j!=i){                
                        private_accumulator+= -theta_ij*vector[j];
                    }
                    else{
                        private_accumulator+= (1+nodes_degree[i]*theta_ij)*vector[j];
                    }
                }
            }
            result[i]=private_accumulator;
        }
    }
}



void multiply_invCond_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result)
{
    double theta_ij = Dc*dt;
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; ++i) {
            result[i]=vector[i]/(1+nodes_degree[i]*theta_ij);
        }
    }
}



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
    ) {
    multiply_diff_matrix_vector(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,pk,Apk);
    double alpha_k_num = scalar_product_parallel(rk, zk, nb_nodes);
    double alpha_k_den =  scalar_product_parallel(Apk, pk, nb_nodes);
    double alpha_k = alpha_k_num / alpha_k_den;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            uk[i] += alpha_k * pk[i];
            rk[i] -=  alpha_k * Apk[i];
        }
    }

    multiply_invCond_matrix_vector(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,rk,zk);
    double beta_k = scalar_product_parallel(rk, zk,nb_nodes);
    beta_k = beta_k / alpha_k_num;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i <nb_nodes; i++) {
            pk[i] = zk[i] + beta_k * pk[i];
        }
    }
}

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
    int max_itr) {
    

    multiply_diff_matrix_vector(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,uk,Auk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            rk[i] = b[i] - Auk[i];
        }
    }

    multiply_invCond_matrix_vector(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,rk,zk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            pk[i] = zk[i];
        }
    }

    int k = 0;
    while (k < max_itr) {
        if (k%100==0){
            printf("k= %d\n",k);
        }
        // Calculate rk, zk, pk, and uk using the iterer_GCP function
        iterer_GCP_MK(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,rk, zk, pk, uk,Apk);
        double sum_rk_squared = 0.0;
        #pragma omp parallel num_threads(6)
        {
            #pragma omp for reduction(+:sum_rk_squared)
            for (int i = 0; i < nb_nodes; i++) {
                sum_rk_squared += rk[i] * rk[i];
            }
        }
        if (sum_rk_squared < tolerance*tolerance) {
            //printf("k= %d\n",k);
            return uk;
        }

        k++;
    }
    printf("not converged ! ");
    return b;
}





void multiply_sym_csr_Vector(sym_csr* A, double* vector, double* result) {
    for (int i = 0; i < A->n_rows; ++i) {
        result[i] = 0.0; // Initialize the result vector
        for (int j = A->row_ptrs[i]; j < A->row_ptrs[i + 1]-1; ++j) {
            result[i] += A->values[j] * vector[A->col_indices[j]];
            result[A->col_indices[j]] +=  A->values[j] * vector[i];
        }
        result[i] += A->values[A->row_ptrs[i + 1]-1] * vector[A->col_indices[A->row_ptrs[i + 1]-1]];
    }
}


void iterer_GCP_sym_csr(
    sym_csr* A,
    sym_csr* Minv,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    ) {
    
    multiply_sym_csr_Vector(A, pk, Apk);
    double alpha_k_num = scalar_product_parallel(rk, zk, A->n_rows);
    double alpha_k_den =  scalar_product_parallel(Apk, pk, A->n_rows);
    double alpha_k = alpha_k_num / alpha_k_den;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            uk[i] += alpha_k * pk[i];
            rk[i] -=  alpha_k * Apk[i];
        }
    }

    multiply_sym_csr_Vector(Minv, rk, zk);
    double beta_k = scalar_product_parallel(rk, zk, A->n_rows);
    beta_k = beta_k / alpha_k_num;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            pk[i] = zk[i] + beta_k * pk[i];
        }
    }
}


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
    int max_itr) {


    multiply_sym_csr_Vector(A, uk, Auk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            rk[i] = b[i] - Auk[i];
        }
    }

    multiply_sym_csr_Vector(Minv, rk, zk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < A->n_rows; i++) {
            pk[i] = zk[i];
        }
    }

    int k = 0;
    while (k < max_itr) {
        // Calculate rk, zk, pk, and uk using the iterer_GCP function
        iterer_GCP_sym_csr(A, Minv, rk, zk, pk, uk,Apk);

        double sum_rk_squared = 0.0;
        #pragma omp parallel num_threads(6)
        {
            #pragma omp for reduction(+:sum_rk_squared)
            for (int i = 0; i < A->n_rows; i++) {
                sum_rk_squared += rk[i] * rk[i];
            }
        }
        if (sum_rk_squared < tolerance*tolerance) {
            printf("%d\n",k);
            return uk;
        }

        k++;
    }
    printf("not converged ! ");
    return b;
}




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
    double kab){
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0; i< nb_nodes; i++){
            /*--------------------UPTAKE, RESPIRATION, MORTALITY ---------------------------------*/
            double cDOM = M[1][i]/voxelVolume;
            double uptake=(double) ((vdom*cDOM)/(cDOM+kab)) *M[0][i] * deltat; // Uptake according the Monod equation
            if (uptake>M[1][i]) uptake=M[1][i];  //case where bacteria demand > DOM offer

            /*--------------------RESPIRATION---------------------------------*/

            double resp = (double)( rho * M[0][i] ) * deltat; //respiration

            /*--------------------MORTALITY-----------------------------------*/
            double morta=(double)( mu * M[0][i] ) * deltat; //mortality

            /* Case where cell activity > mass of active cell */
            if( (resp + morta) > M[0][i] ) {
                if (morta>M[0][i]) {
                    morta = M[0][i];
                    resp = 0.0;
                }
                else {
                    resp = M[0][i] - morta;
                }
            }
            
            /*--------------------BACTERIAL LYSIS-----------------------------*/
            double morSOM = (double)(1.0-rho_m) * morta ;
            double morDOM = (double) rho_m * morta ; 



            /*------------------- TurnOver --------------------------------------*/

            double turnFOM = (double) vfom* M[3][i] * deltat ;
            double turnSOM = (double) vsom * M[2][i] * deltat ;

            if( turnFOM>M[3][i]) turnFOM = M[3][i];
            if( turnSOM>M[2][i]) turnSOM = M[2][i];

            /*--------------------UPDATING THE PROPERTIES OF THE INDIVIDUAL---------------------------------*/
            M[0][i] += (uptake - morta - resp);
            M[1][i] += (morDOM - uptake+turnFOM +turnSOM) ; 
            M[2][i] += (morSOM - turnSOM);
            M[3][i] += -turnFOM;
            M[4][i] += resp;
        }
    }
}




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
    double kab) {
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            double temp;
            if (M[0][i] > 0.){
                //Growth : first we let the microrganisms eat some from the dom in orther to grow
                if (M[1][i]>0.) { 
                    temp =  (double) (vdom*M[1][i]*M[0][i]*deltat)/(kab+M[1][i]);//(voxelVolume*kab+M[1][i]);
                    if (M[1][i] > temp){ // that the microorganisms have excess DOM 
                        M[0][i] = M[0][i] + temp; // we let the microorganisms grow
                        M[1][i] = M[1][i] - temp;
                    }
                    else{
                        M[0][i] = M[0][i] + M[1][i]; // the microorganisms don't have enough DOM in order to grow during deltat
                        M[1][i] = 0.0; // it lasts no DOM anymore in this voxel
                    }
                }
                // Mortality : the decomposition of MB after dying to DOM and FOM 
                temp = (double) mu*M[0][i]*deltat; //the portion of MB to be decomposed 
                if (M[0][i] > temp){ // there is enough MB
                    M[0][i] = M[0][i] - temp; // MB dying
                    M[1][i] = M[1][i] + rho_m * temp; //fast decomposition
                    M[2][i] = M[2][i] + (1-rho_m) * temp; //slow decomposition
                }
                else {
                    M[1][i] = M[1][i] + rho_m * M[0][i]; //fast decomposition
                    M[2][i] = M[2][i] + (1-rho_m) * M[0][i]; //slow decomposition
                    M[0][i] = 0.0; // it lasts no MB anymore in this voxel
                }
                // Respiration: CO2 emission
                if (M[0][i] > 0.){ 
                    temp = (double) rho * M[0][i] * deltat;
                    if (M[0][i] > temp) {
                        M[0][i] = M[0][i] - temp ;
                        M[4][i] = M[4][i] + temp; // CO2 emission by microorganisms
                    }
                    else {
                        M[4][i] = M[4][i] + M[0][i]; // CO2 emission by microorganisms
                        M[0][i] = 0.0;
                    }
                }
            }
            //Transformation of SOM and FOM to DOM 
            //Transformation of SOM
            if (M[2][i] > 0.) {
                temp = (double) vsom * M[2][i] * deltat; // portion of SOM that can be dissolved during deltat (SOM to DOM)
                if (M[2][i] > temp){ // there is enough SOM in the ball
                    M[1][i] = M[1][i] + temp;
                    M[2][i] = M[2][i] - temp;
                }
                else {
                    M[1][i] = M[1][i] + M[2][i];
                    M[2][i] = 0.0;
                }
            }
            //transformation of FOM 
            if (M[3][i] > 0.) {
                temp = (double) vfom * M[3][i] * deltat; // portion of FOM that can be dissolved during deltat  (FOM to DOM)
                if (M[3][i] > temp){  // there is enough FOM in the ball
                    M[1][i] = M[1][i] + temp ;
                    M[3][i] = M[3][i] - temp;
                }
                else {
                    M[1][i] = M[1][i] + M[3][i] ;
                    M[3][i] = 0.0 ;
                }
            }
        }
    }
}


void mass_sum(double **M,int nb_nodes,double history[5]){
    double sum1 ;
    double sum2 ;
    double sum3 ;
    double sum4 ;
    double sum5 ;
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for reduction(+:sum1, sum2, sum3, sum4, sum5)
        for (int i = 0; i < nb_nodes; i++) {
            sum1 += M[0][i];
            sum2 += M[1][i];
            sum3 += M[2][i];
            sum4 += M[3][i];
            sum5 += M[4][i];
        }
    }
    history[0] = sum1;
    history[1] = sum2;
    history[2] = sum3;
    history[3] = sum4;
    history[4] = sum5;
}


int load3DImage(char* path,int*** threeDArray,int length,int width,int height){
    int nb_nodes=0;
    // Loop through files and read data into the 3D array
    for (int i = 0; i < length; i++) {
        char filename[20];
        sprintf(filename, "input/%s/%d.txt",path, i);  // Generate filename (0.txt to 511.txt)

        FILE *file = fopen(filename, "r");
        if (file == NULL) {
            perror("Error opening file");
            return 1;
        }

        for (int j = 0; j < width; j++) {
            for (int k = 0; k < height; k++) {
                if (fscanf(file, "%d", &threeDArray[i][j][k]) != 1) {
                    fprintf(stderr, "Error reading data from file %s\n", filename);
                    return 1;
                }
                if (threeDArray[i][j][k]>0){
                    nb_nodes++;
                }
            }
        }

        fclose(file);
    }
    return nb_nodes;
}

int get_coordinates(int*** threeDArray,int** nodesCoordinates,int* nodes_degree,int** nodes_Adjacency, int length,int width,int height){
    int di[6] = {1, -1, 0, 0, 0, 0};
    int dj[6] = {0, 0, 1, -1, 0, 0};
    int dk[6] = {0, 0, 0, 0, 1, -1};

    int n_nz = 0;
    for (int i=0;i<length;i++){
        for (int j=0;j<width;j++){
            for (int k=0;k<height;k++){
                if (threeDArray[i][j][k]>0){
                    nodesCoordinates[threeDArray[i][j][k]-1][0] = i;
                    nodesCoordinates[threeDArray[i][j][k]-1][1] = j;
                    nodesCoordinates[threeDArray[i][j][k]-1][2] = k;
                    nodes_Adjacency[threeDArray[i][j][k]-1][6] = threeDArray[i][j][k]-1;
                    n_nz ++;
                    for (int m=0;m<6;m++){
                        int new_i = i+di[m];
                        int new_j = j+dj[m];
                        int new_k = k+dk[m];
                        if (new_i>=0 && new_j>=0 && new_k>=0 && new_i<length && new_j<width && new_k<height && isNeighbor6(i,j,k,new_i,new_j,new_k)){
                            if (threeDArray[new_i][new_j][new_k]>0){
                                nodes_Adjacency[threeDArray[i][j][k]-1][m]=threeDArray[new_i][new_j][new_k]-1;
                                nodes_degree[threeDArray[i][j][k]-1]++;
                                n_nz++;
                            }
                        }
                    }
                    bubbleSort(nodes_Adjacency[threeDArray[i][j][k]-1],7);
                }
            }
        }
    }
    return n_nz;
}


void filling_pore_space(int** nodesCoordinates, double **M, int nb_nodes,double DOM_totalMass, char* biomass_file, double microorganismWeight){
    for (int i=0;i<nb_nodes;i++){
        M[1][i] = DOM_totalMass/nb_nodes;
    }
    char filename[20];
    sprintf(filename, "input/%s",biomass_file);
    FILE *biomass = fopen(filename, "r");
    
    int nb_spots;

    if (fscanf(biomass, "%d", &nb_spots) != 1) {
        printf("Error reading the number of lines from the file\n");
        fclose(biomass);
    }
    bool found;
    int x,y,z,nb_mo,z_a,y_a,x_a;
    for (int i=0;i<nb_spots;i++){
        if (fscanf(biomass, "%d %d %d %d", &x, &y, &z, &nb_mo) != 4) {
            printf("Error reading data from the file\n");
            fclose(biomass);
        }
        found =0;
        for (int j=0;j<nb_nodes;j++){
            z_a = nodesCoordinates[j][0];
            y_a = nodesCoordinates[j][1];
            x_a = nodesCoordinates[j][2];
            if ((x == x_a) && (y == y_a) && (z == z_a)){
                M[0][j] += nb_mo * microorganismWeight;
                found=1;
                break;
            }
        }
        if (found==0){
            printf("no voxel found!");
        }
    }
}

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
    char* outputFile){

    // initiate a sparse matrix in CSR format
    Sparse_CSR* A = (Sparse_CSR*)malloc(sizeof(Sparse_CSR));
    Sparse_CSR* CM = (Sparse_CSR*)malloc(sizeof(Sparse_CSR));

    printf("constructing the diffusion matrix\n");
    create_implicit_diffusion_matrix_from_adjacency(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,n_nz,A,CM);
    
    // Open a file for writing
    char filename[40];

    sprintf(filename, "output/%s_trtype=%d_kb=%e_dt=%lf_DT=%lf",outputFile,typetr,kab,dt*60*60*24,DT);
    FILE* file_hisotry = fopen(filename, "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }

    double* Auk=(double*)calloc(nb_nodes, sizeof(double));
    double* rk=(double*)calloc(nb_nodes, sizeof(double));
    double* zk=(double*)calloc(nb_nodes, sizeof(double));
    double* pk=(double*)calloc(nb_nodes, sizeof(double));
    double* Apk=(double*)calloc(nb_nodes, sizeof(double));
    
    if (Auk == NULL || rk == NULL || zk == NULL || pk == NULL || Apk == NULL) {
        printf("problem here \n");
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(B,nb_nodes,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int nb_steps = DT/dt;
    printf("nb_steps = %d\n",nb_steps);
    
    int savingdevider = nb_steps/1;
    printf("saving devider=%d\n",savingdevider);
    switch (typetr)
    {
    case 0:
        for (int step=1;step<nb_steps+1;step++){
            Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt, rho,mu, rho_m, vfom, vsom, vdom, kab);
            B[1] = gradient_conjugue_preconditionne_parallel(A,CM,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    case 1:
        for (int step=1;step<nb_steps+1;step++){
            perform_bacterial_actions(B,nb_nodes,voxelVolume,dt,rho,mu,rho_m,vfom,vsom,vdom,kab);
            B[1] = gradient_conjugue_preconditionne_parallel(A,CM,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    }
    
    end_time = omp_get_wtime();

    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    free(Auk);
    free(rk);
    free(zk);
    free(pk);
    free(Apk);
}






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
    char* outputFile){
    // Open a file for writing
    char filename[40];
    char filenamemetadata[40];
    sprintf(filename, "output/%s",outputFile);
    sprintf(filenamemetadata, "output/%s.md",outputFile);
    FILE* file_hisotry = fopen(filename, "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }
    FILE* file_md = fopen(filenamemetadata, "w");
    if (file_md == NULL) {
        printf("Failed to open file for writing\n");
    }

    fprintf(file_md,"dt_diff = %e seconds \ndt_tr = %e seconds \ndT =%e day \nkab =%e µg/v \ntolerance = %e \ntypetr =%d \n", dt*(60*60*24),dt_tr*(60*60*24),DT,kab,tolerance,typetr);

    double* Auk=(double*)calloc(nb_nodes, sizeof(double));
    double* rk=(double*)calloc(nb_nodes, sizeof(double));
    double* zk=(double*)calloc(nb_nodes, sizeof(double));
    double* pk=(double*)calloc(nb_nodes, sizeof(double));
    double* Apk=(double*)calloc(nb_nodes, sizeof(double));
    
    if (Auk == NULL || rk == NULL || zk == NULL || pk == NULL || Apk == NULL) {
        printf("problem here \n");
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(B,nb_nodes,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int nb_steps = DT/dt;
    printf("nb_steps = %d\n",nb_steps);
    int nb_steps_tr= dt/dt_tr;
    printf("nb_steps_tr = %d\n",nb_steps_tr);
    int savingdevider = nb_steps/120;
    printf("saving devider=%d\n",savingdevider);
    switch (typetr)
    {
    case 0:
        for (int step=1;step<nb_steps+1;step++){
            for (int tr_s=0;tr_s<nb_steps_tr;tr_s++){
                Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt_tr, rho,mu, rho_m, vfom, vsom, vdom, kab);
            }
            //Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt, rho,mu, rho_m, vfom, vsom, vdom, kab);
            B[1] = GCP_MK(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
            if ((step%3600 == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                }
        }
        break;
    case 1:
        for (int step=1;step<nb_steps+1;step++){
            for (int tr_s=0;tr_s<nb_steps_tr;tr_s++){
                perform_bacterial_actions(B,nb_nodes,voxelVolume,dt_tr,rho,mu,rho_m,vfom,vsom,vdom,kab);
            }
            //perform_bacterial_actions(B,nb_nodes,voxelVolume,dt,rho,mu,rho_m,vfom,vsom,vdom,kab);
            B[1] = GCP_MK(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    }
    
    end_time = omp_get_wtime();
    fprintf(file_md,"Execution time: %lf seconds\n", end_time - start_time);

    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    free(Auk);
    free(rk);
    free(zk);
    free(pk);
    free(Apk);
}




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
    char* outputFile){
    // Open a file for writing
    char filename[40];
    char filenamemetadata[40];
    sprintf(filename, "output/%s",outputFile);
    sprintf(filenamemetadata, "output/%s.md",outputFile);
    FILE* file_hisotry = fopen(filename, "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }
    FILE* file_md = fopen(filenamemetadata, "w");
    if (file_md == NULL) {
        printf("Failed to open file for writing\n");
    }

    fprintf(file_md,"dt_diff = %e seconds \ndt_tr = %e seconds \ndT =%e day \nkab =%e µg/v \ntolerance = %e \ntypetr =%d \n", dt*(60*60*24),dt_tr*(60*60*24),DT,kab,tolerance,typetr);

    double* Auk=(double*)calloc(nb_nodes, sizeof(double));
    double* rk=(double*)calloc(nb_nodes, sizeof(double));
    double* zk=(double*)calloc(nb_nodes, sizeof(double));
    double* pk=(double*)calloc(nb_nodes, sizeof(double));
    double* Apk=(double*)calloc(nb_nodes, sizeof(double));
    
    if (Auk == NULL || rk == NULL || zk == NULL || pk == NULL || Apk == NULL) {
        printf("problem here \n");
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(B,nb_nodes,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int nb_steps = DT/dt;
    printf("nb_steps = %d\n",nb_steps);
    int nb_steps_tr= dt/dt_tr;
    printf("nb_steps_tr = %d\n",nb_steps_tr);
    int savingdevider = nb_steps/120;
    printf("saving devider=%d\n",savingdevider);
    switch (typetr)
    {
    case 0:
        for (int step=1;step<nb_steps+1;step++){
            Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt_tr, rho,mu, rho_m, vfom, vsom, vdom, kab);
            //Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt, rho,mu, rho_m, vfom, vsom, vdom, kab);
            B[1] = GCP_MK(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
            if ((step%3600 == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                }
        }
        break;
    case 1:
        for (int step=1;step<nb_steps+1;step++){
            perform_bacterial_actions(B,nb_nodes,voxelVolume,dt_tr,rho,mu,rho_m,vfom,vsom,vdom,kab);
            //perform_bacterial_actions(B,nb_nodes,voxelVolume,dt,rho,mu,rho_m,vfom,vsom,vdom,kab);
            B[1] = GCP_MK(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    }
    
    end_time = omp_get_wtime();
    fprintf(file_md,"Execution time: %lf seconds\n", end_time - start_time);

    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    free(Auk);
    free(rk);
    free(zk);
    free(pk);
    free(Apk);
}









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
    char* outputFile){

    // initiate a sparse matrix in CSR format
    sym_csr* A = (sym_csr*)malloc(sizeof(sym_csr));
    sym_csr* CM = (sym_csr*)malloc(sizeof(sym_csr));

    printf("constructing the diffusion matrix\n");
    construct_diff_sym(Dc,dt,nodes_Adjacency,nodes_degree,nb_nodes,n_nz,A,CM);
    
    
    // Open a file for writing
    char filename[40];

    sprintf(filename, "output/%s_trtype=%d_kb=%e_dt=%lf_DT=%lf",outputFile,typetr,kab,dt*60*60*24,DT);
    FILE* file_hisotry = fopen(filename, "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }

    double* Auk=(double*)calloc(nb_nodes, sizeof(double));
    double* rk=(double*)calloc(nb_nodes, sizeof(double));
    double* zk=(double*)calloc(nb_nodes, sizeof(double));
    double* pk=(double*)calloc(nb_nodes, sizeof(double));
    double* Apk=(double*)calloc(nb_nodes, sizeof(double));
    
    if (Auk == NULL || rk == NULL || zk == NULL || pk == NULL || Apk == NULL) {
        printf("problem here \n");
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(B,nb_nodes,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int nb_steps = DT/dt;
    printf("nb_steps = %d\n",nb_steps);
    
    int savingdevider = nb_steps/1;
    printf("saving devider=%d\n",savingdevider);
    switch (typetr)
    {
    case 0:
        for (int step=1;step<nb_steps+1;step++){
            Asynchronous_Transformation(B,nb_nodes,voxelVolume,dt, rho,mu, rho_m, vfom, vsom, vdom, kab);
            B[1] = GCP_sym_csr_parallel(A,CM,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    case 1:
        for (int step=1;step<nb_steps+1;step++){
            perform_bacterial_actions(B,nb_nodes,voxelVolume,dt,rho,mu,rho_m,vfom,vsom,vdom,kab);
            B[1] = GCP_sym_csr_parallel(A,CM,B[1],Auk,B[1],rk,zk,pk,Apk,tolerance,max_itr);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    }
    
    end_time = omp_get_wtime();

    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    free(Auk);
    free(rk);
    free(zk);
    free(pk);
    free(Apk);
}

void multiply_exp_diff_matrix_vector(
    double Dc,
    double dt,
    int **nodes_Adjacency,
    int *nodes_degree,
    int nb_nodes,
    double* vector,
    double* result)
{
    double theta_ij = -Dc*dt;
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; ++i) {
            result[i]=0.0;
            double private_accumulator = 0.0;
            #pragma omp simd reduction(+:private_accumulator)
            for (int m=0;m<7;m++){
                int j = nodes_Adjacency[i][m];
                if (j!=-1) {
                    if (j!=i){                
                        private_accumulator+= -theta_ij*vector[j];
                    }
                    else{
                        private_accumulator+= (1+nodes_degree[i]*theta_ij)*vector[j];
                    }
                }
            }
            result[i]=private_accumulator;
        }
    }
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; ++i) {
            if (result[i]<0){
                printf("diminuer dt\n");
            }
            vector[i]=result[i];
        }
    }
}






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
    char* outputFile){
    // Open a file for writing
    char filename[40];
    char filenamemetadata[40];
    sprintf(filename, "output/%s",outputFile);
    sprintf(filenamemetadata, "output/%s.md",outputFile);
    FILE* file_hisotry = fopen(filename, "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }
    FILE* file_md = fopen(filenamemetadata, "w");
    if (file_md == NULL) {
        printf("Failed to open file for writing\n");
    }

    fprintf(file_md,"dt = %e seconds \ndT =%e day \nkab =%e µg/v \nvdom=%e \ntolerance = %e \ntypetr =%d \n", dt*(60*60*24),DT,kab,vdom,tolerance,typetr);

    double* Auk=(double*)calloc(nb_nodes, sizeof(double));
    double* rk=(double*)calloc(nb_nodes, sizeof(double));
    double* zk=(double*)calloc(nb_nodes, sizeof(double));
    double* pk=(double*)calloc(nb_nodes, sizeof(double));
    double* Apk=(double*)calloc(nb_nodes, sizeof(double));
    
    if (Auk == NULL || rk == NULL || zk == NULL || pk == NULL || Apk == NULL) {
        printf("problem here \n");
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(B,nb_nodes,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int nb_steps = DT/dt;
    printf("nb_steps = %d\n",nb_steps);
    
    int savingdevider = nb_steps/120;
    printf("saving devider=%d\n",savingdevider);
    double* temp = (double*)calloc(nb_nodes,sizeof(double));
    switch (typetr)
    {
    case 0:
        for (int step=1;step<nb_steps+1;step++){
            Asynchronous_Transformation_diff_MK(nb_nodes,B,nodes_Adjacency,nodes_degree,voxelVolume,Dc,rho,mu,rho_m,vfom,vsom,vdom,kab,dt);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    case 1:
        for (int step=1;step<nb_steps+1;step++){
            perform_bacterial_actions_diff_MK(nb_nodes,B,nodes_Adjacency,nodes_degree,voxelVolume,Dc,rho,mu,rho_m,vfom,vsom,vdom,kab,dt);
            if ((step%savingdevider == 0)){
                mass_sum(B,nb_nodes,history);
                printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
                fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
            }
        }
        break;
    }
    
    end_time = omp_get_wtime();
    fprintf(file_md,"Execution time: %lf seconds\n", end_time - start_time);
    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    free(Auk);
    free(rk);
    free(zk);
    free(pk);
    free(Apk);
}








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
    double deltat) {


    double theta_ij = -Dc*deltat;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            double temp;
            if (M[0][i] > 0.){
                //Growth : first we let the microrganisms eat some from the dom in orther to grow
                if (M[1][i]>0.) { 
                    temp =  (double) (vdom*M[1][i]*M[0][i]*deltat)/(kab+M[1][i]);//(voxelVolume*kab+M[1][i]);
                    if (M[1][i] > temp){ // that the microorganisms have excess DOM 
                        M[0][i] = M[0][i] + temp; // we let the microorganisms grow
                        M[1][i] = M[1][i] - temp;
                    }
                    else{
                        M[0][i] = M[0][i] + M[1][i]; // the microorganisms don't have enough DOM in order to grow during deltat
                        M[1][i] = 0.0; // it lasts no DOM anymore in this voxel
                    }
                }
                // Mortality : the decomposition of MB after dying to DOM and FOM 
                temp = (double) mu*M[0][i]*deltat; //the portion of MB to be decomposed 
                if (M[0][i] > temp){ // there is enough MB
                    M[0][i] = M[0][i] - temp; // MB dying
                    M[1][i] = M[1][i] + rho_m * temp; //fast decomposition
                    M[2][i] = M[2][i] + (1-rho_m) * temp; //slow decomposition
                }
                else {
                    M[1][i] = M[1][i] + rho_m * M[0][i]; //fast decomposition
                    M[2][i] = M[2][i] + (1-rho_m) * M[0][i]; //slow decomposition
                    M[0][i] = 0.0; // it lasts no MB anymore in this voxel
                }
                // Respiration: CO2 emission
                if (M[0][i] > 0.){ 
                    temp = (double) rho * M[0][i] * deltat;
                    if (M[0][i] > temp) {
                        M[0][i] = M[0][i] - temp ;
                        M[4][i] = M[4][i] + temp; // CO2 emission by microorganisms
                    }
                    else {
                        M[4][i] = M[4][i] + M[0][i]; // CO2 emission by microorganisms
                        M[0][i] = 0.0;
                    }
                }
            }
            //Transformation of SOM and FOM to DOM 
            //Transformation of SOM
            if (M[2][i] > 0.) {
                temp = (double) vsom * M[2][i] * deltat; // portion of SOM that can be dissolved during deltat (SOM to DOM)
                if (M[2][i] > temp){ // there is enough SOM in the ball
                    M[1][i] = M[1][i] + temp;
                    M[2][i] = M[2][i] - temp;
                }
                else {
                    M[1][i] = M[1][i] + M[2][i];
                    M[2][i] = 0.0;
                }
            }
            //transformation of FOM 
            if (M[3][i] > 0.) {
                temp = (double) vfom * M[3][i] * deltat; // portion of FOM that can be dissolved during deltat  (FOM to DOM)
                if (M[3][i] > temp){  // there is enough FOM in the ball
                    M[1][i] = M[1][i] + temp ;
                    M[3][i] = M[3][i] - temp;
                }
                else {
                    M[1][i] = M[1][i] + M[3][i] ;
                    M[3][i] = 0.0 ;
                }
            }
        }
        #pragma omp for
        for (int i=0;i<nb_nodes;i++){
            double temp = 0.0;
            #pragma omp simd reduction(+:temp)
            for (int m=0;m<7;m++){
                int j = nodes_Adjacency[i][m];
                if (j!=-1) {
                    if (j!=i){                
                        temp+= -theta_ij*M[1][j];
                    }
                    else{
                        temp+= (1+nodes_degree[i]*theta_ij)*M[1][j];
                    }
                }
            }
            M[1][i]=temp;
        }
    }
}


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
    double deltat)
    {
    
    double theta_ij = -Dc*deltat;
    
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0; i< nb_nodes; i++){
            /*--------------------UPTAKE, RESPIRATION, MORTALITY ---------------------------------*/
            double cDOM = M[1][i]/voxelVolume;
            double uptake=(double) ((vdom*cDOM)/(cDOM+kab)) *M[0][i] * deltat; // Uptake according the Monod equation
            if (uptake>M[1][i]) uptake=M[1][i];  //case where bacteria demand > DOM offer

            /*--------------------RESPIRATION---------------------------------*/

            double resp = (double)( rho * M[0][i] ) * deltat; //respiration

            /*--------------------MORTALITY-----------------------------------*/
            double morta=(double)( mu * M[0][i] ) * deltat; //mortality

            /* Case where cell activity > mass of active cell */
            if( (resp + morta) > M[0][i] ) {
                if (morta>M[0][i]) {
                    morta = M[0][i];
                    resp = 0.0;
                }
                else {
                    resp = M[0][i] - morta;
                }
            }
            
            /*--------------------BACTERIAL LYSIS-----------------------------*/
            double morSOM = (double)(1.0-rho_m) * morta ;
            double morDOM = (double) rho_m * morta ; 



            /*------------------- TurnOver --------------------------------------*/

            double turnFOM = (double) vfom* M[3][i] * deltat ;
            double turnSOM = (double) vsom * M[2][i] * deltat ;

            if( turnFOM>M[3][i]) turnFOM = M[3][i];
            if( turnSOM>M[2][i]) turnSOM = M[2][i];

            /*--------------------UPDATING THE PROPERTIES OF THE INDIVIDUAL---------------------------------*/
            M[0][i] += (uptake - morta - resp);
            M[1][i] += (morDOM - uptake+turnFOM +turnSOM) ; 
            M[2][i] += (morSOM - turnSOM);
            M[3][i] += -turnFOM;
            M[4][i] += resp;
        }
        #pragma omp for
        for (int i=0;i<nb_nodes;i++){
            double temp = 0.0;
            #pragma omp simd reduction(+:temp)
            for (int m=0;m<7;m++){
                int j = nodes_Adjacency[i][m];
                if (j!=-1) {
                    if (j!=i){                
                        temp+= -theta_ij*M[1][j];
                    }
                    else{
                        temp+= (1+nodes_degree[i]*theta_ij)*M[1][j];
                    }
                }
            }
            M[1][i]=temp;
        }
    }
}

