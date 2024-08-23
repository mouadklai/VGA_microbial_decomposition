#ifndef _DC_H_
#define _DC_H_

// Define neural network structure
typedef struct {
    int nb_balls;
    int** adjacency;
    int nb_adjacencies;
    double* weights;
    int** Adj_ptrs;
} NeuralNetwork;

int getRandomInt(int n);
double getRandomDouble(double f);
int spheresIntersect(double x1,double y1, double z1,double r1,double x2,double y2, double z2,double r2);
void Adjacency_balls(int nb_balls, double** ballSet,int** adjacance,int nb_edges, NeuralNetwork *nn);
void initializeWeights(NeuralNetwork *nn, int** adjacance,int nb_edges,double** ballsSet, int nb_balls) ;
void save_Weights(double* weights,int nb_weights, const char *filename) ;

void load_Weights(double* weights,const char *filename) ;
void Train_step(NeuralNetwork *nn, double** X , int batch_size, int total_nb_data, int nb_balls,double** ballsSet,double** Y_true,double diffusion_coefficient,double time_step,double LEARNING_RATE);
void get_output(NeuralNetwork *nn,int batch_size,int* indices, double** X, double** Y_pred,double** ballsSet,int nb_balls,double diffusion_coefficient,double time_step);
double error(int batch_size, int* indices, double** Y_true,double** Y_pred, int nb_balls);
int isPointInsideSphere(int x, int y, int z, double centerX, double centerY, double centerZ, double radius);

int construct_3D_image(double** ballsSet, int nb_balls,int*** threeDArray_voxels,int* nb_voxels_in_balls,int length,int width,int height);
void balls_voxels(double** ballsSet, int nb_balls,int*** threeDArray_voxels,int* voxels_balls,int length,int width,int height);
void fill_balls_from_voxelsDistribution(
    int* voxels_balls,
    int nb_voxels,
    int nb_balls,
    double* M_v,
    double* M_b);
void fill_voxels_from_ballsDistribution(
    int* voxels_balls,
    int* nb_voxels_in_balls,
    int nb_voxels,
    double* M_b,
    double* M_v);
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
    int max_itr);
    

void multiply_diff_matrix_vector_balls(
    double Dc,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* vector,
    double* result);
void multiply_invCond_matrix_vector_balls(
    double Dc,
    double dt,
    double** ballset,
    NeuralNetwork *nn,
    double* vector,
    double* result
    );
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
    );
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
    int max_itr);
void get_output_longdt(NeuralNetwork *nn,int batch_size,int* indices, double** X, double** Y_pred,double** ballsSet,int nb_balls,double diffusion_coefficient,double time_step,int nb_steps);
void Train_step_implicit(NeuralNetwork *nn,double* dloss_dalphaij, double** X , int batch_size, int total_nb_data, int nb_balls,double** ballsSet,double** Y_true,double diffusion_coefficient,double time_step,double LEARNING_RATE,double alpha);

#endif
