
#include "DC.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"


int getRandomInt(int n) {
    return rand() % (n + 1);  // Modulo n+1 ensures values between 0 and n
}

double getRandomDouble(double f) { 
    return ((double)rand() / RAND_MAX) * f + 1e-9;
}


// Function to check if two spheres intersect
int spheresIntersect(double x1,double y1, double z1,double r1,double x2,double y2, double z2,double r2) {
    double distance = sqrt((x1 - x2) * (x1 - x2) +
                     (y1 - y2) * (y1 - y2) +
                     (z1 - z2) * (z1 - z2));
    float sumRadii = r1 + r2 ;

    if (distance - sumRadii<=0.2 ) {
        return 1; // Intersecting
    } else {
        return 0; // Not intersecting
    }
}
// Comparison function for qsort
int compare(const void *a, const void *b) {
    const int *ia = *(const int **)a;
    const int *ib = *(const int **)b;
    return ia[0] - ib[0]; // Compare based on the first column
}

void Adjacency_balls(int nb_balls, double** ballSet,int** adjacance,int nb_edges, NeuralNetwork *nn){
    for (int eij=0;eij<nb_edges;eij++){
        nn->adjacency[eij][0]=adjacance[eij][0];
        nn->adjacency[eij][1]=adjacance[eij][1];
        nn->adjacency[eij][2]=eij;
        nn->adjacency[eij+nb_edges][0]=adjacance[eij][1];
        nn->adjacency[eij+nb_edges][1]=adjacance[eij][0];
        nn->adjacency[eij+nb_edges][2]=eij;
    }
    qsort(nn->adjacency, nb_edges*2, sizeof(nn->adjacency[0]), compare);

    int eij =0;
    int i,ni;
    i=nn->adjacency[eij][0];
    nn->Adj_ptrs[i][0] = eij;
    while(eij+1<nb_edges*2){
        i=nn->adjacency[eij][0];
        ni =nn->adjacency[eij+1][0];
        if (ni!=i){
            nn->Adj_ptrs[i][1] = eij;
            nn->Adj_ptrs[ni][0] = eij+1;
        }
        eij++;
    }
    nn->Adj_ptrs[ni][1] = eij;
 
}

// Function to initialize weights randomly
void initializeWeights(NeuralNetwork *nn, int** adjacance,int nb_edges,double** ballsSet, int nb_balls) {
    
    nn->Adj_ptrs = (int**)calloc(nb_balls,sizeof(int*));
    for (int i=0;i<nb_balls;i++){
        nn->Adj_ptrs[i]=(int*)calloc(2,sizeof(int));
    }
    
    nn->nb_adjacencies = nb_edges*2;
    nn->adjacency = (int**)calloc(nn->nb_adjacencies,sizeof(int*));
    for (int i=0;i<nn->nb_adjacencies;i++){
        nn->adjacency[i]=(int*)calloc(3,sizeof(int));
    }
    Adjacency_balls(nb_balls, ballsSet,adjacance,nb_edges,nn);
    printf("nb of adjacencies (doubled) is : %d \n" ,nn->nb_adjacencies);

    nn->weights = (double*)calloc(nb_edges,sizeof(double)); 

    for (int adj=0;adj<nb_edges;adj++){
        int i = adjacance[adj][0];
        int j = adjacance[adj][1];
        double dij = sqrt((ballsSet[i][0] - ballsSet[j][0])*(ballsSet[i][0] - ballsSet[j][0]) + (ballsSet[i][1] - ballsSet[j][1])*(ballsSet[i][1] - ballsSet[j][1])+(ballsSet[i][2] - ballsSet[j][2])*(ballsSet[i][2] - ballsSet[j][2]));
        double rij = 2*ballsSet[i][3]*ballsSet[j][3]/(ballsSet[i][3]+ballsSet[j][3]);
        double sij = 3.14159265358979323846 * rij*rij;
        nn->weights[adj]= sij/dij ;//((float)rand() / RAND_MAX) * 10.0;//
    }
}

void save_Weights(double* weights,int nb_weights, const char *filename) {
// Open a file for writing
    FILE *fp = fopen(filename, "w");

    // Check if file opened successfully
    if (fp == NULL) {
        printf("Error opening file.\n");
    }

    // Write array elements to the file
    for (int i = 0; i < nb_weights; i++) {
        fprintf(fp, "%.17g\n", weights[i]); // Write each element followed by a newline
    }

    // Close the file
    fclose(fp);

}

void load_Weights(double* weights,const char *filename) {
    int size = 0;
    // Open the file for reading
    FILE *fp = fopen(filename, "r");

    // Check if file opened successfully
    if (fp == NULL) {
        printf("Error opening file.\n");
    }

    // Read array elements from the file
    while (fscanf(fp, "%lf", &weights[size]) == 1) {
        size++; // Increment size after reading each element
    }

    // Close the file
    fclose(fp);
}


void Train_step(NeuralNetwork *nn, double** X , int batch_size, int total_nb_data, int nb_balls,double** ballsSet,double** Y_true,double diffusion_coefficient,double time_step,double LEARNING_RATE){
    int half_nb_edges = nn->nb_adjacencies/2;
    int* indices = (int*)calloc(batch_size,sizeof(int));
    for (int i=0;i<batch_size;i++){
        indices[i] = rand() % total_nb_data;
        
    }
    double* dloss_dalphaij = (double*)calloc(half_nb_edges,sizeof(double));
    for (int edge_index=0;edge_index<half_nb_edges;edge_index++){
        dloss_dalphaij[edge_index] = 0.0;
    }
    for (int edge_index=0;edge_index<nn->nb_adjacencies;edge_index++){
        for (int sample=0;sample<batch_size;sample++){
            int i=nn->adjacency[edge_index][0];
            int j=nn->adjacency[edge_index][1];
            double ci =X[indices[sample]][i]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[i][3]*ballsSet[i][3]*ballsSet[i][3]);
            double cj = X[indices[sample]][j]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[j][3]*ballsSet[j][3]*ballsSet[j][3]);
            /*if ((ci!=0.0)||(cj!=0.0)){
                printf("ci = %lf , cj=%lf\n",ci,cj);
            }*/
            double temp=0.0;
            for (int linked_edge=nn->Adj_ptrs[i][0];linked_edge<nn->Adj_ptrs[i][1];linked_edge++){
                int k=nn->adjacency[linked_edge][1];
                double ck = X[indices[sample]][k]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[k][3]*ballsSet[k][3]*ballsSet[k][3]);
                temp += nn->weights[nn->adjacency[edge_index][2]]*(ci-ck);
            }

            dloss_dalphaij[nn->adjacency[edge_index][2]] += diffusion_coefficient*time_step*(ci-cj)*(Y_true[indices[sample]][i]-X[indices[sample]][i]+diffusion_coefficient*time_step*temp);
        }
    }
    for (int edge_index=0;edge_index<half_nb_edges;edge_index++){
        if (dloss_dalphaij[edge_index]>0 || dloss_dalphaij[edge_index]<0){
            //printf("dloss_dalphaij[edge_index] = %e\n",(LEARNING_RATE*dloss_dalphaij[edge_index]*2.0)/batch_size);
        }
        printf("before nn->weights[edge_index] = %e \n",nn->weights[edge_index]);
        nn->weights[edge_index]-=(LEARNING_RATE*dloss_dalphaij[edge_index]*2.0)/batch_size;
        printf("after nn->weights[edge_index] = %e \n",nn->weights[edge_index]);
    }
    free(indices);
}



void Train_step_implicit(NeuralNetwork *nn,double* dloss_dalphaij, double** X , int batch_size, int total_nb_data, int nb_balls,double** ballsSet,double** Y_true,double diffusion_coefficient,double time_step,double LEARNING_RATE,double alpha){
    int half_nb_edges = nn->nb_adjacencies/2;
    int* indices = (int*)calloc(batch_size,sizeof(int));
    for (int i=0;i<batch_size;i++){
        indices[i] = rand() % total_nb_data;
        
    }


    for (int edge_index=0;edge_index<nn->nb_adjacencies;edge_index++){
        int i=nn->adjacency[edge_index][0];
        int j=nn->adjacency[edge_index][1];
        for (int sample=0;sample<batch_size;sample++){
            double ci =Y_true[indices[sample]][i]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[i][3]*ballsSet[i][3]*ballsSet[i][3]);
            double cj = Y_true[indices[sample]][j]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[j][3]*ballsSet[j][3]*ballsSet[j][3]);
            /*if ((ci!=0.0)||(cj!=0.0)){
                printf("ci = %lf , cj=%lf\n",ci,cj);
            }*/
            double temp=0.0;
            for (int linked_edge=nn->Adj_ptrs[i][0];linked_edge<nn->Adj_ptrs[i][1];linked_edge++){
                int k=nn->adjacency[linked_edge][1];
                double ck = Y_true[indices[sample]][k]/((4.0/3.0) * 3.14159265358979323846 *ballsSet[k][3]*ballsSet[k][3]*ballsSet[k][3]);
                temp += nn->weights[nn->adjacency[edge_index][2]]*(ci-ck);
            }

            dloss_dalphaij[nn->adjacency[edge_index][2]] = -2.0*time_step*diffusion_coefficient* (ci-cj)*(X[indices[sample]][i]-Y_true[indices[sample]][i]-diffusion_coefficient*time_step*temp);
            dloss_dalphaij[nn->adjacency[edge_index][2]] /= fabs(dloss_dalphaij[nn->adjacency[edge_index][2]]);
            double new_alpha_ij = nn->weights[nn->adjacency[edge_index][2]]-LEARNING_RATE*dloss_dalphaij[nn->adjacency[edge_index][2]];
            if (new_alpha_ij>1e-16 && (new_alpha_ij<30.0)){
                 nn->weights[nn->adjacency[edge_index][2]] = new_alpha_ij;
            }           
        }
    }

    free(indices);
}


void get_output(NeuralNetwork *nn,int batch_size,int* indices, double** X, double** Y_pred,double** ballsSet,int nb_balls,double diffusion_coefficient,double time_step){
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int sample=0;sample<batch_size;sample++){
            int indx= indices[sample];
            for (int ball_index=0;ball_index<nb_balls;ball_index++){
                double temp=0.0;
                for (int linked_edge=nn->Adj_ptrs[ball_index][0];linked_edge<nn->Adj_ptrs[ball_index][1];linked_edge++){
                    int k=nn->adjacency[linked_edge][1];
                    double dcij=(X[indx][ball_index]/((4.0/3.0) * 3.14159265358979323846 * pow(ballsSet[ball_index][3], 3)))-(X[indx][k]/((4.0/3.0) * 3.14159265358979323846 * pow(ballsSet[k][3], 3)));
                    temp+=nn->weights[nn->adjacency[linked_edge][2]]*dcij;
                }
                Y_pred[sample][ball_index] = X[indx][ball_index] - diffusion_coefficient*time_step*temp;
            }
        }
    }
}

void get_output_longdt(NeuralNetwork *nn,int batch_size,int* indices,double** output,double** output_next,double** ballsSet,int nb_balls,double diffusion_coefficient,double time_step,int nb_steps){
    for (int sample=0;sample<batch_size;sample++){
        for (int step=0;step<nb_steps;step++){
            for (int ball_index=0;ball_index<nb_balls;ball_index++){
                double ci = (output[sample][ball_index]/((4.0/3.0) * 3.14159265358979323846 * pow(ballsSet[ball_index][3], 3)));
                double temp=0.0; 
                for (int linked_edge=nn->Adj_ptrs[ball_index][0];linked_edge<nn->Adj_ptrs[ball_index][1];linked_edge++){
                    int k=nn->adjacency[linked_edge][1];
                    double dcij=ci-(output[sample][k]/((4.0/3.0) * 3.14159265358979323846 * pow(ballsSet[k][3], 3)));
                    temp+=nn->weights[nn->adjacency[linked_edge][2]]*dcij;
                }
                output_next[sample][ball_index] = output[sample][ball_index] - diffusion_coefficient*time_step*temp;
            }
            for (int ball_index=0;ball_index<nb_balls;ball_index++){
                output[sample][ball_index] = output_next[sample][ball_index];
            }
        }
    }
}

double error(int batch_size, int* indices, double** Y_true,double** Y_pred, int nb_balls){
    double err=0.0;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for reduction(+:err)
        for (int sample=0;sample<batch_size;sample++){
            int indx =indices[sample];
            for (int i=0;i<nb_balls;i++){
                err += (Y_true[indx][i] - Y_pred[sample][i])*(Y_true[indx][i] - Y_pred[sample][i]);
            }
        }
    }
    return err/(batch_size*nb_balls);
}


int isPointInsideSphere(int x, int y, int z, double centerX, double centerY, double centerZ, double radius) {
    double distance = sqrt((int) (x - centerX) * (int) (x - centerX) +
                     (int) (y - centerY) * (int) (y - centerY) +
                    (int) (z - centerZ) * (int)(z - centerZ)); 
    return distance -radius<=1;
}


int construct_3D_image(double** ballsSet, int nb_balls,int*** threeDArray_voxels,int* nb_voxels_in_balls,int length,int width,int height){
    int nb_voxels=1;
    for(int i=0;i<nb_balls;i++){
        double centerX=ballsSet[i][0];
        double centerY=ballsSet[i][1];
        double centerZ=ballsSet[i][2];
        double radius=ballsSet[i][3];
        int nb_voxels_in_ball=1;
        for (int x = (int) (centerX - radius); x <= (int) (centerX + radius+1); x++) {
            for (int y = (int) (centerY - radius); y <= (int) (centerY + radius+1); y++) {
                for (int z = (int) (centerZ - radius); z <= (int) (centerZ + radius+1); z++) {
                    if (isPointInsideSphere(x, y, z, centerX, centerY, centerZ, radius+1) && (x<length)&&(y<width)&&(z<height)&&(x>=0)&&(y>=0)&&(z>=0)) {
                        if (threeDArray_voxels[x][y][z]==0){
                            threeDArray_voxels[x][y][z]=nb_voxels;
                            nb_voxels_in_ball++;
                            nb_voxels++;
                        }
                    }
                }
            }
        }
        nb_voxels_in_balls[i]=nb_voxels_in_ball;
    }
    
    return nb_voxels-1;
}

void balls_voxels(double** ballsSet, int nb_balls,int*** threeDArray_voxels,int* voxels_balls,int length,int width,int height){
    for(int i=0;i<nb_balls;i++){
        double centerX=ballsSet[i][0];
        double centerY=ballsSet[i][1];
        double centerZ=ballsSet[i][2];
        double radius=ballsSet[i][3];
        for (int x = (int) (centerX - radius); x <= (int) (centerX + radius+1); x++) {
            for (int y = (int) (centerY - radius); y <= (int) (centerY + radius+1); y++) {
                for (int z = (int) (centerZ - radius); z <= (int) (centerZ + radius+1); z++) {
                    if (isPointInsideSphere(x, y, z, centerX, centerY, centerZ, radius) && (x<length)&&(y<width)&&(z<height)&&(x>=0)&&(y>=0)&&(z>=0)) {
                        if (threeDArray_voxels[x][y][z]>0){
                            if (voxels_balls[threeDArray_voxels[x][y][z]-1]>=0){
                                int candidate_i=voxels_balls[threeDArray_voxels[x][y][z]-1];
                                double centerX_i=ballsSet[candidate_i][0];
                                double centerY_i=ballsSet[candidate_i][1];
                                double centerZ_i=ballsSet[candidate_i][2];
                                double radius_i=ballsSet[candidate_i][3];
                                double distanceSq = (x - centerX) * (x - centerX) +
                                                    (y - centerY) * (y - centerY) +
                                                    (z - centerZ) * (z - centerZ);
                                double distance_iSq = (x - centerX_i) * (x - centerX_i) +
                                                    (y - centerY_i) * (y - centerY_i) +
                                                    (z - centerZ_i) * (z - centerZ_i);
                                if ((distance_iSq - radius_i*radius_i) <= (distanceSq - radius*radius)) {
                                    voxels_balls[threeDArray_voxels[x][y][z]-1]=i;
                                }
                            }
                            else{
                                voxels_balls[threeDArray_voxels[x][y][z]-1]=i;
                            }
                        }
                    }
                }
            }
        }
    }
}

void fill_balls_from_voxelsDistribution(
    int* voxels_balls,
    int nb_voxels,
    int nb_balls,
    double* M_v,
    double* M_b)
{
    for (int i=0;i<nb_balls;i++){
        M_b[i]=0.0;
    }
    for (int i=0;i<nb_voxels;i++){
        M_b[voxels_balls[i]] += M_v[i];
    }
}

void fill_voxels_from_ballsDistribution(
    int* voxels_balls,
    int* nb_voxels_in_balls,
    int nb_voxels,
    double* M_b,
    double* M_v)
{
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0;i<nb_voxels;i++){
            M_v[i] = M_b[voxels_balls[i]] / nb_voxels_in_balls[voxels_balls[i]];
        }
    }
}






int GCP_MK_new(
    double DC,
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
    

    multiply_diff_matrix_vector(DC,dt,nodes_Adjacency,nodes_degree,nb_nodes,uk,Auk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            rk[i] = b[i] - Auk[i];
        }
    }

    multiply_invCond_matrix_vector(DC,dt,nodes_Adjacency,nodes_degree,nb_nodes,rk,zk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nb_nodes; i++) {
            pk[i] = zk[i];
        }
    }
    int c=0;

    int k = 0;
    while (k < max_itr) {
        // Calculate rk, zk, pk, and uk using the iterer_GCP function
        iterer_GCP_MK(DC,dt,nodes_Adjacency,nodes_degree,nb_nodes,rk, zk, pk, uk,Apk);
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
            c=1;
            return c;
        }

        k++;
    }
    return c;
}



void multiply_diff_matrix_vector_balls(
    double Dc,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* vector,
    double* result)
{
    double theta_ij = Dc*dt;
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nn->nb_balls; ++i) {
            double vi = (4.0/3.0) * 3.14159265358979323846 * ballset[i][3]*ballset[i][3]*ballset[i][3];
            double ci = vector[i]/vi;
            double private_accumulator = 0.0;
            #pragma omp simd reduction(+:private_accumulator)
            for (int eij=nn->Adj_ptrs[i][0];eij<nn->Adj_ptrs[i][0];eij++){
                int j = nn->adjacency[eij][1];
                double cj = vector[j]/((4.0/3.0) * 3.14159265358979323846 *ballset[j][3]*ballset[j][3]*ballset[j][3]);
                private_accumulator+= theta_ij*nn->weights[nn->adjacency[eij][2]]*(ci - cj);

            }
            result[i]=vector[i] + private_accumulator;
        }
    }
}

void multiply_invCond_matrix_vector_balls(
    double Dc,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* vector,
    double* result
    ){
        double theta_ij = Dc*dt;
        #pragma omp  parallel num_threads(6)
        {
            #pragma omp for
            for (int i=0;i<nn->nb_balls;i++){
                double vi = (4.0/3.0) * 3.14159265358979323846 * ballset[i][3]*ballset[i][3]*ballset[i][3];
                double private_accumulator = 0.0;
                #pragma omp simd reduction(+:private_accumulator)
                for (int eij=nn->Adj_ptrs[i][0];eij<nn->Adj_ptrs[i][1];eij++){
                    int j =  nn->adjacency[eij][1];
                    private_accumulator+= theta_ij * nn->weights[nn->adjacency[eij][2]];
                }
                private_accumulator = 1 + private_accumulator/vi ;
                result[i] = vector[i]/private_accumulator;
            }
        }   
    }


void iterer_GCP_MK_balls(
    double Dc,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* rk,
    double* zk,
    double* pk,
    double* uk,
    double* Apk
    ) {
    multiply_diff_matrix_vector_balls(Dc,dt,ballset,nn,pk,Apk);
    double alpha_k_num = scalar_product_parallel(rk, zk, nn->nb_balls);
    double alpha_k_den =  scalar_product_parallel(Apk, pk, nn->nb_balls);
    double alpha_k = alpha_k_num / alpha_k_den;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nn->nb_balls; i++) {
            uk[i] += alpha_k * pk[i];
            rk[i] -=  alpha_k * Apk[i];
        }
    }

    multiply_invCond_matrix_vector_balls(Dc,dt,ballset,nn,rk,zk);
    double beta_k = scalar_product_parallel(rk, zk,nn->nb_balls);
    beta_k = beta_k / alpha_k_num;

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i <nn->nb_balls; i++) {
            pk[i] = zk[i] + beta_k * pk[i];
        }
    }
}




double* GCP_MK_balls(
    double DC,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* b, 
    double* Auk, 
    double* uk, 
    double* rk, 
    double* zk, 
    double* pk,
    double* Apk, 
    double tolerance, 
    int max_itr) {
    

    multiply_diff_matrix_vector_balls(DC,dt,ballset,nn,uk,Auk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nn->nb_balls; i++) {
            rk[i] = b[i] - Auk[i];
        }
    }

    multiply_invCond_matrix_vector_balls(DC,dt,ballset,nn,rk,zk);

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i = 0; i < nn->nb_balls; i++) {
            pk[i] = zk[i];
        }
    }
    int c=0;

    int k = 0;
    while (k < max_itr) {
        printf("iteration %d \n",k);
        // Calculate rk, zk, pk, and uk using the iterer_GCP function
        iterer_GCP_MK_balls(DC,dt,ballset,nn,rk, zk, pk, uk,Apk);
        double sum_rk_squared = 0.0;
        #pragma omp parallel num_threads(6)
        {
            #pragma omp for reduction(+:sum_rk_squared)
            for (int i = 0; i < nn->nb_balls; i++) {
                sum_rk_squared += rk[i] * rk[i];
            }
        }
        if (sum_rk_squared < tolerance*tolerance) {
            //printf("k= %d\n",k);
            return uk;
        }

        k++;
    }
    return b;
}

