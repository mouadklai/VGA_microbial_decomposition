#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "DC.h"



void multiply_exp_diff_matrix_vector_balls(
    NeuralNetwork *nn,
    int** adjacency,
    int nb_edges,
    int nb_balls,
    double Dc,
    double dt,
    double* vector,
    double** ballset,
    double* result)
{
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0;i<nb_balls;i++){
            result[i]=vector[i];
        }
    }
    double pi3_4 = (4.0/3.0) * 3.14159265358979323846;
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int eij=0;eij<nb_edges;eij++){
            int i = adjacency[eij][0];
            int j= adjacency[eij][1];
            double ci = vector[i]/(pi3_4 * ballset[i][3]*ballset[i][3]*ballset[i][3]);
            double cj = vector[j]/(pi3_4*ballset[j][3]*ballset[j][3]*ballset[j][3]);
            //double D_ij = sqrt((ballset[i][0] - ballset[j][0])*(ballset[i][0] - ballset[j][0]) + (ballset[i][1] - ballset[j][1])*(ballset[i][1] - ballset[j][1])+(ballset[i][2] - ballset[j][2])*(ballset[i][2] - ballset[j][2]));
            //double R_ij = 2*ballset[i][3]*ballset[j][3]/(ballset[i][3]+ballset[j][3]);
            //double S_ij = 3.14159265358979323846 * R_ij*R_ij;
            //double alpha_ij = Dc*0.6*dt*S_ij/D_ij;
            result[i] -= nn->weights[eij]*Dc*dt*(ci-cj);
            result[j] -= nn->weights[eij]*Dc*dt*(cj-ci);
        }
    }
    #pragma omp  parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0;i<nb_balls;i++){
            vector[i] = result[i];
        }
    }
}



double* BallsToPlans(double** BallsSet, double* DOM_distribution, int numBalls,double* output) {

    for (int i = 0; i < numBalls; i++) {
        float z = BallsSet[i][2];
        float ball_radius = BallsSet[i][3];
        int start_plan = (int)(z - ball_radius);
        int end_plan = (int)(z + ball_radius);
        double sum_surf = 0.0;
        for (int plan = start_plan; plan <= end_plan; plan++) {
            sum_surf += sqrt(ball_radius * ball_radius + (z - plan) * (z - plan));
        }
        if (sum_surf!=0.0){
            for (int plan = start_plan; plan <= end_plan; plan++) {     
                output[plan] += DOM_distribution[i] *sqrt(ball_radius * ball_radius + (z - plan) * (z - plan)) / sum_surf;
            }
        }
        
    }

    return output;
}

int main() {
    printf("#---- loading the set of balls ----#\n");
    FILE *fp = fopen("input/boules-100_.bmax", "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    int nb_balls;


    // Read the size of the array
    fscanf(fp, "%d", &nb_balls);
    printf("%d\n", nb_balls);
    
    double** ballsSet=(double**)calloc(nb_balls,sizeof(double *));
    for (int i=0;i<nb_balls;i++){
        ballsSet[i] = (double *) calloc(4,sizeof(double));
    }

    for (int i=0;i<nb_balls;i++){
        fscanf(fp, "%lf %lf %lf %lf\n", &ballsSet[i][0],&ballsSet[i][1],&ballsSet[i][2],&ballsSet[i][3]);
    }
    fclose(fp);
    FILE *fa = fopen("input/boules-100_.adj", "r");
    if (fa == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    int nb_edge = 647409;
    printf("%d\n",nb_edge);
    int** adjacance=(int**)calloc(nb_edge,sizeof(int *));
    for (int i=0;i<nb_edge;i++){
        adjacance[i] = (int *) calloc(2,sizeof(int));
    }

    for (int i=0;i<nb_edge;i++){
        fscanf(fa, "%d %d\n", &adjacance[i][0],&adjacance[i][1]);
    }
    fclose(fa);
    


    printf("#--- Calculating the ball adjacency and intialize the neural network ---#\n");

    NeuralNetwork* alphas = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    alphas->nb_balls = nb_balls;
    initializeWeights(alphas,adjacance,nb_edge, ballsSet, nb_balls);


    char* weights_filename="input/DC/weights_999.txt";
    //load_Weights(alphas->weights,"input/DC/weights_i.txt");
    
    int load;
    printf("\nTo continue training ----> type 1    &   To restart training -----> type 0  : ");
    scanf("%d",&load);

    if(load){
        printf("loading weights...\n");
        load_Weights(alphas->weights,weights_filename);
    }




/*


    printf("#---- Constructing a 3D image from the balls ----#\n");
    int length = 512; 
    int width = 512;
    int height = 512;
    // Construct a 3D image from the balls
    int ***threeDArray_voxels = (int ***) calloc(length,sizeof(int **));
    for (int i = 0; i < height; i++) {
        threeDArray_voxels[i] = (int **)calloc(width,sizeof(int *));
        for (int j = 0; j < width; j++) {
            threeDArray_voxels[i][j] = (int *)calloc(width,sizeof(int));
        }
    }
    int* nb_voxels_in_balls = (int*)calloc(nb_balls,sizeof(int));
    int nb_voxels = construct_3D_image(ballsSet, nb_balls, threeDArray_voxels, nb_voxels_in_balls,length,width,height);

    int nb_v;
    for (int i=0;i<nb_balls;i++){
        nb_v +=nb_voxels_in_balls[i];
        
    }

    printf("nb_voxels = %d\n",nb_voxels);

    printf("nb voxels in balls = %d\n",nb_v);
    int* voxels_balls = (int*)calloc(nb_voxels,sizeof(int));

    for (int i=0;i<nb_voxels;i++){
        voxels_balls[i] = -1;
    }

    balls_voxels(ballsSet, nb_balls, threeDArray_voxels, voxels_balls,length,width,height);



    printf("#---- constructing the graph of voxels ----#\n");
    
    int n_nz_v;
    int **voxels_Coordinates = (int **) calloc(nb_voxels , sizeof(int *)); // 2D array for saving the corrdinates of the valid voxels in the 3D image
    for (int n=0; n<nb_voxels;n++){
        voxels_Coordinates[n] = (int *) calloc(3,sizeof(int));
    }
    int **voxels_Adjacency = (int **) calloc(nb_voxels,sizeof(int *));
    for (int i=0;i<nb_voxels;i++){
        voxels_Adjacency[i] = (int *) calloc(7,sizeof(int));
        for (int m=0;m<7;m++){
            voxels_Adjacency[i][m]=-1;
        }
    }

    int* voxels_degree = (int *) calloc(nb_voxels,sizeof(int));
    for (int i=0;i<nb_voxels;i++){
        voxels_degree[i]=0;
    }

    n_nz_v = get_coordinates(threeDArray_voxels,voxels_Coordinates,voxels_degree,voxels_Adjacency,length,width,height);


    for (int i=0;i<nb_voxels;i++){
        printf("%d, %d, %d, %d, %d, %d, %d\n",voxels_Adjacency[i][0],voxels_Adjacency[i][1],voxels_Adjacency[i][2],voxels_Adjacency[i][3],voxels_Adjacency[i][4],voxels_Adjacency[i][5],voxels_Adjacency[i][6]);
    }

    printf("n_nz_v = %d\n ",n_nz_v);
    printf("#----Generating data --#\n");
    

*/
    

    double* Masses = (double*)calloc(nb_balls,sizeof(double));
    double* temp_M = (double*)calloc(nb_balls,sizeof(double));
    double DELTAT = 1.783/24;
    double DC = 100950.0;

    double M0 = 592.7593;
    double dt = 0.1/(60*60*24);

    double totalVolume = 0.0;
    for (int i=0;i<nb_balls;i++){
        if ((ballsSet[i][2] - ballsSet[i][3])>0 && (ballsSet[i][2] - ballsSet[i][3])<2){
            totalVolume += ballsSet[i][3]*ballsSet[i][3]*ballsSet[i][3];
        }
    }


    double tm = 0.0;
    
    for (int i=0;i<nb_balls;i++){
        if ((ballsSet[i][2] - ballsSet[i][3])>0 && (ballsSet[i][2] - ballsSet[i][3])<2){
            Masses[i] += ballsSet[i][3]*ballsSet[i][3]*ballsSet[i][3] * M0/totalVolume;
        }
    }
    

    for (int i=0;i<nb_balls;i++){
        temp_M[i] = 0.0;
    }
    

    double* Auk=(double*)calloc(nb_balls, sizeof(double));
    double* rk=(double*)calloc(nb_balls, sizeof(double));
    double* zk=(double*)calloc(nb_balls, sizeof(double));
    double* pk=(double*)calloc(nb_balls, sizeof(double));
    double* Apk=(double*)calloc(nb_balls, sizeof(double));


    int nb_iterations = DELTAT/dt;
    double time_e = 0.0;
    for (int iter=0;iter<nb_iterations;iter++){
        tm=0.0;
        for (int i=0;i<nb_balls;i++){
            tm+=Masses[i];
        }
        if (iter%100 == 0){
            time_e = iter*dt*24;
            printf("time %lf h total mass = %lf \n ",time_e,tm);
        }
        multiply_exp_diff_matrix_vector_balls(alphas,adjacance,nb_edge,nb_balls,DC,dt,Masses,ballsSet,temp_M);
        //Masses = GCP_MK_balls(DC,dt,ballsSet,alphas,Masses,Auk,Masses,rk,zk,pk,Apk,0.00001,1000);
    }


    double* plans_masses = (double *) calloc(512,sizeof(double));
    for (int i=0;i<512;i++){
        plans_masses[i] = 0.0;
    }
    plans_masses = BallsToPlans(ballsSet,Masses,nb_balls,plans_masses);

    tm=0.0;
    for (int i=0;i<512;i++){
        tm+=plans_masses[i];
    }

    printf("tm=%lf\n",tm);


    FILE *file = fopen("output/PNM_plans.txt", "w");

    if (file == NULL) {
        perror("Error opening the file");
        return 1;
    }

    for (int i = 0; i < 512; i++) {
        fprintf(file, "%lf\n", plans_masses[i]);
    }

    fclose(file);


/*

    double z,ball_radius,sum_surf,surface_radius;
    int start_plan,end_plan;
    for (int i = 0; i < nb_balls; i++) {
        z = ballsSet[i][2];
        ball_radius = ballsSet[i][3];
        start_plan = (int)(z - ball_radius);
        end_plan = (int)(z + ball_radius);
        sum_surf = 0.0;
        if (start_plan-1>=0 && end_plan+1 <512){
            for (int j = start_plan-1; j < end_plan+1; j++) {
                surface_radius = sqrt(ball_radius * ball_radius + (z - j) * (z - j));
                sum_surf += surface_radius;
            }
            for (int j = start_plan-1; j < end_plan+1; j++) {
                surface_radius = sqrt(ball_radius * ball_radius + (z - j) * (z - j));
                plans_masses[j] += Masses[i] * surface_radius /sum_surf;
            }
        }
        else{
            for (int j = start_plan; j < end_plan; j++) {
                surface_radius = sqrt(ball_radius * ball_radius + (z - j) * (z - j));
                sum_surf += surface_radius;
            }
            for (int j = start_plan; j < end_plan; j++) {
                surface_radius = sqrt(ball_radius * ball_radius + (z - j) * (z - j));
                plans_masses[j] += Masses[i] * surface_radius /sum_surf;
            }
        }
    }
    

    tm = 0.0;
    for (int i=0;i<512;i++){
        tm+=plans_masses[i];
    }

    printf("total mass in plans = %lf \n",tm);


*/


    
    printf("done!\n");

    return 0;
}
