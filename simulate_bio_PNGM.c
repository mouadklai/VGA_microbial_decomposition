#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "DC.h"

#include <omp.h>


#define PI 3.14159265358979323846

void fill_Balls_with_MB_Distributeoverthenearest_balls(double** BallsSet, int numBalls, double** voxels_spots, int numSpots, double DOM_mass,double microorganisms_mass, double** output) {
    double totalVolume = 0.0;
    // Calculate the total volume of all balls
    for (int i = 0; i < numBalls; i++) {
        totalVolume += (4.0 / 3.0) * PI * pow(BallsSet[i][3], 3);
    }

    // Distribute the spot mass over the nearest balls
    for (int i = 0; i < numSpots; i++) {
        double minDist = 1000.0;
        int nearestIndex = -1;

        for (int j = 0; j < numBalls; j++) {
            double dist = pow(voxels_spots[i][0] - BallsSet[j][0], 2) +
                          pow(voxels_spots[i][1] - BallsSet[j][1], 2) +
                          pow(voxels_spots[i][2] - BallsSet[j][2], 2) -
                          pow(BallsSet[j][3], 2);
            if (dist < minDist) {
                minDist = dist;
                nearestIndex = j;
            }
        }
        output[0][nearestIndex] += voxels_spots[i][3]*microorganisms_mass;
    }
    

    for (int i = 0; i < numBalls; i++) {
        double v = (4.0 / 3.0) * PI * pow(BallsSet[i][3], 3);
        output[1][i] = DOM_mass * v / totalVolume;
    }
}
double pi3_4 = (4.0/3.0) * 3.14159265358979323846;

void multiply_exp_diff_matrix_vector_balls(
    double* alpha,
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
            result[i] -= alpha[eij]*Dc*dt*(ci-cj);
            result[j] -= alpha[eij]*Dc*dt*(cj-ci);
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




void Asynchronous_Transformation_balls(
    double **M,
    double **ballset,
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
            double vi =pi3_4 * ballset[i][3]*ballset[i][3]*ballset[i][3]; 
            if (M[0][i] > 0.){
                //Growth : first we let the microrganisms eat some from the dom in orther to grow
                if (M[1][i]>0.) { 
                    temp =  (double) (vdom*M[1][i]*M[0][i]*deltat)/(kab*vi+M[1][i]);//(voxelVolume*kab+M[1][i]);
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
     
    printf("done!\n");
    double* weights = (double*)calloc(nb_edge,sizeof(double)); 

    char* weights_filename="input/DC/weights_999.txt";

    load_Weights(weights,weights_filename);

    //Parameters 
    double voxelHeight = 24.0; //micrometer
    double microorganismWeight = 5.4e-8; //microgramme 
    double DOM_totalMass = 289.5 ;


    printf("-------Filling pore space ----\n");

    
    double** M=(double**)calloc(5,sizeof(double *));
    for (int i=0;i<5;i++){
        M[i] = (double *) calloc(nb_balls,sizeof(double));
    }

    
    char* biomass_file = "input/biomasse.dat";
    
    FILE *biomass = fopen(biomass_file, "r");
    
    int nb_spots;

    if (fscanf(biomass, "%d", &nb_spots) != 1) {
        printf("Error reading the number of lines from the file\n");
        fclose(biomass);
    }

    double** biomass_spots = (double**) calloc(nb_spots, sizeof(double*));
    for (int i = 0; i < nb_spots; i++) {
        biomass_spots[i] = (double*)calloc(4, sizeof(double));
    }
    

    for (int i = 0; i < nb_spots; i++) {
        if (fscanf(biomass, "%lf %lf %lf %lf", &biomass_spots[i][0], &biomass_spots[i][1], &biomass_spots[i][2], &biomass_spots[i][3]) != 4) {
            printf("Error reading data from the file\n");
            fclose(biomass);
            return 1;
        }
    }
    fill_Balls_with_MB_Distributeoverthenearest_balls(ballsSet,nb_balls,biomass_spots,nb_spots,DOM_totalMass,microorganismWeight,M);

    double M01 = 0.0;
    double M02 = 0.0;
    for (int i=0;i<nb_balls;i++){
        M01 += M[0][i];
        M02 += M[1][i];
    }

    printf("%lf \t %lf \n",M01,M02);
    printf("------- Performing simulation ------ \n");
    double Dc =100950.0; // micrometer².j⁻¹
    double deltat = 10.0/(60*60*24) ; //step time in days for implicit scheme and transformation process 
    double dt = 0.1/(60*60*24) ; // step time in days for Explicit scheme
    double DeltaT= 5.0; //Time step in days of the hole simulation 

    double rho, mu, rho_m, vfom, vsom, vdom, kab;
    rho = 0.2;
    mu = 0.5;
    rho_m = 0.55;
    vfom = 0.3;
    vsom = 0.0;
    vdom = 9.6; 
    kab = 0.000000001;//4.40e-11;
    kab*=voxelHeight*voxelHeight*voxelHeight;
    
    printf("biological parameters : rho=%e, mu=%e,rho_m=%e ,vfom=%e ,vsom=%e ,vdom=%e ,kab=%e, nb_nodes = %d, n_nz=%d\n",rho, mu, rho_m, vfom, vsom, vdom, kab,nb_balls);
    
    int nb_steps = DeltaT /deltat;
    double* temp = (double*)calloc(nb_balls,sizeof(double));
    M01 = 0.0;
    for (int i=0;i<nb_balls;i++){
        M01 += M[0][i];
        M01 += M[1][i];
        M01 += M[2][i];
        M01 += M[3][i];
        M01 += M[4][i];
    }
    printf("total mass = %lf  \n",M01);

    // Capture the starting time
    double start_time, end_time;
    start_time = omp_get_wtime();

    FILE* file_hisotry = fopen("output/PNGM_simulation/history.txt", "w");
    if (file_hisotry == NULL) {
        printf("Failed to open file for writing\n");
    }
    double* history = (double *)calloc(5,sizeof(double));
    mass_sum(M,nb_balls,history);
    fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    printf("%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
    int savingdevider = nb_steps/120;
    printf("saving devider=%d\n",savingdevider);
    int nb_tr_steps = deltat/dt;
    for (int step =0;step<nb_steps;step++){
        printf("%d of %d \n",step,nb_steps);
        if (((step+1)%savingdevider == 0)){
            mass_sum(M,nb_balls,history);
            printf("step %d of %d\n%lf %lf %lf %lf %lf \n",step,nb_steps,history[0],history[1],history[2],history[3],history[4]);
            fprintf(file_hisotry,"%lf %lf %lf %lf %lf \n",history[0],history[1],history[2],history[3],history[4]);
        }
        Asynchronous_Transformation_balls(M,ballsSet,nb_balls,voxelHeight,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab);
        for (int tr_step=0;tr_step<nb_tr_steps;tr_step++){
            multiply_exp_diff_matrix_vector_balls(weights,adjacance,nb_edge,nb_balls,Dc,dt,M[1],ballsSet,temp);
        }
    }


    end_time = omp_get_wtime();

    printf("Done! \nExecution time: %lf seconds\n", end_time - start_time);
    
    // Freeing allocated memory to avoid leaks
    for (int i = 0; i < nb_balls; i++) {
        free(ballsSet[i]);
    }
    free(ballsSet);

    for (int i = 0; i < nb_edge; i++) {
        free(adjacance[i]);
    }
    free(adjacance);

    free(weights);

    for (int i = 0; i < 5; i++) {
        free(M[i]);
    }
    free(M);

    for (int i = 0; i < nb_spots; i++) {
        free(biomass_spots[i]);
    }
    free(biomass_spots);

    return 0;
}
