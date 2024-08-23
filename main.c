#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"

int main(){
    int length = 512;  // Number of files (0.txt to 511.txt)
    int width = 512;    // Assuming each file contains 512 line
    int height = 512;    // Number of elements in each line 
    double DOM_totalMass = 289.5 ;
    double Dc= 100950.0;
    double tolerance = 0.00001;
    char* biomass_file = "biomasse.dat";
    double microorganismWeight =  5.41*0.00000001 ;
    int ***threeDArray = (int ***) calloc(length,sizeof(int **));
    for (int i = 0; i < height; i++) {
        threeDArray[i] = (int **)calloc(width,sizeof(int *));
        for (int j = 0; j < width; j++) {
            threeDArray[i][j] = (int *)calloc(width,sizeof(int));
        }
    }
    int nb_nodes;
    printf("#---- loading the 3D image ----#\n");
    nb_nodes = load3DImage("images",threeDArray,length,width,height);
    int n_nz;
    int **nodesCoordinates = (int **) calloc(nb_nodes , sizeof(int *)); // 2D array for saving the corrdinates of the valid voxels in the 3D image
    for (int n=0; n<nb_nodes;n++){
        nodesCoordinates[n] = (int *) calloc(3,sizeof(int));
    }
    int **nodes_Adjacency = (int **) calloc(nb_nodes,sizeof(int *));
    for (int i=0;i<nb_nodes;i++){
        nodes_Adjacency[i] = (int *) calloc(7,sizeof(int));
        for (int m=0;m<7;m++){
            nodes_Adjacency[i][m]=-1;
        }
    }

    int* nodes_degree = (int *) calloc(nb_nodes,sizeof(int));
    for (int i=0;i<nb_nodes;i++){
        nodes_degree[i]=0;
    }
    printf("#---- constructing the graph ----#\n");
    n_nz = get_coordinates(threeDArray,nodesCoordinates,nodes_degree,nodes_Adjacency,length,width,height);
    double** M=(double**)calloc(5,sizeof(double *));
    for (int i=0;i<5;i++){
        M[i] = (double *) calloc(nb_nodes,sizeof(double));
    }
    printf("#---- filling the voxels with biomass ----#\n");
    filling_pore_space(nodesCoordinates,M, nb_nodes,DOM_totalMass,biomass_file, microorganismWeight);
    double voxelVolume = 24.0*24.0*24.0;
    double rho, mu, rho_m, vfom, vsom, vdom, kab;
    rho = 0.2;
    mu = 0.5;
    rho_m = 0.55;
    vfom = 0.3;
    vsom = 0.0;
    vdom = 9.6; 
    kab = 0.000000001;//4.40e-11;

    /*
    FILE *bio_parameters = fopen("input/boules.par", "r");
    if (bio_parameters == NULL) {
        perror("Error opening file");
        return 1;
    }
    int count = fscanf(bio_parameters, "%lf %lf %lf %lf %lf %lf %lf", &rho, &mu, &rho_m, &vfom, &vsom, &vdom, &kab);
    if (count != 7) {
        printf("Error reading values from the file\n");
        return 1;
    }
    fclose(bio_parameters);
    */
    printf("#---- perform simulation of transformation-diffusion ----#\n");
    int max_itr = 1000;
    printf("Enter simulation time in days: (double)");
    double DT ;
    scanf("%lf", &DT);
    printf("Enter time step for diffusion in seconds: dt = (double)");
    double dt;
    scanf("%lf", &dt);
    dt = dt/(60*60*24);
    printf("Enter time step for transformation in seconds: dt_tr = (double)");
    double dt_tr;
    scanf("%lf", &dt_tr);
    dt_tr = dt_tr/(60*60*24);

    printf("biological parameters : rho=%e, mu=%e,rho_m=%e ,vfom=%e ,vsom=%e ,vdom=%e ,kab=%e, nb_nodes = %d, n_nz=%d\n",rho, mu, rho_m, vfom, vsom, vdom, kab,nb_nodes,n_nz);
    printf("Enter name for output_file: ");
    char output_file[100];
    scanf("%s", output_file);
    simulate_biologie_MK_prime(M, nodes_Adjacency, nodes_degree, nb_nodes, n_nz, Dc, tolerance, max_itr, DT, dt, dt_tr,voxelVolume, rho,mu,rho_m,vfom,vsom,vdom,kab,1,output_file);
    //simulate_biologie_exp_MK(M,nodes_Adjacency,nodes_degree,nb_nodes,Dc,tolerance,max_itr,DT,dt,voxelVolume,rho,mu,rho_m,vfom,vsom,vdom,kab,0,output_file);
    printf("done!\n");
    for (int i=0;i<nb_nodes;i++){
        free(nodesCoordinates[i]);
    }
    free(nodesCoordinates);
    free(nodes_degree);
    return 1;
}