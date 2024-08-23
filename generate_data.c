#include "DC.h"
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"


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

    printf("#--- Calculating the ball adjacency and intialize the neural network ---#\n");

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
/*

    for (int i=0;i<nb_voxels;i++){
        printf("%d, %d, %d, %d, %d, %d, %d\n",voxels_Adjacency[i][0],voxels_Adjacency[i][1],voxels_Adjacency[i][2],voxels_Adjacency[i][3],voxels_Adjacency[i][4],voxels_Adjacency[i][5],voxels_Adjacency[i][6]);
    }
*/
    printf("n_nz_v = %d\n ",n_nz_v);
    printf("#----Generating data --#\n");
    
    double* Auk=(double*)calloc(nb_voxels, sizeof(double));
    double* rk=(double*)calloc(nb_voxels, sizeof(double));
    double* zk=(double*)calloc(nb_voxels, sizeof(double));
    double* pk=(double*)calloc(nb_voxels, sizeof(double));
    double* Apk=(double*)calloc(nb_voxels, sizeof(double));

    
    int start_idx = 0;
    int end_idx = 100;

    double* b_dis = (double*)calloc(nb_balls,sizeof(double));
    
    double* v_dis = (double*)calloc(nb_voxels,sizeof(double));
    
    double* temp_v = (double*)calloc(nb_voxels,sizeof(double));

    #pragma omp parallel num_threads(6)
    {
        #pragma omp for
        for (int i=0;i<nb_balls;i++){
            b_dis[i]=0.0;
        }

        #pragma omp for
        for (int i=0;i<nb_voxels;i++){
            v_dis[i]=0.0;
            temp_v[i] = 0.0;
        }
    }

    double Dc = 100950.0;

    double dt = 0.1/(60.0*60.0*24.0);
    int nb_steps = 300*10;


    for (int i=start_idx;i<end_idx;i++){
        printf("data point %d of %d\n",i,end_idx);
        char folder_name[50];
        sprintf(folder_name, "input/training_data/%d", i);
        int status = mkdir(folder_name); 

        // Check if folder creation was successful
        if (status == 0) {
            printf("Folder created successfully.\n");
        } else {
            printf("Failed to create folder.\n");
        }

        double randomMass= getRandomDouble(500.0)+20.0;

        //int direction = getRandomInt(2);
        /*
                printf("direction : %d\n",direction);
        int nbv=0;
        for (int voxel=0;voxel<nb_voxels;voxel++){
            if ((voxels_Coordinates[voxel][direction]==0) ||(voxels_Coordinates[voxel][direction]==1)){
                v_dis[voxel] = randomMass;
                nbv++;
            }
            else
            {
                v_dis[voxel]=0.0;
            }
        }
        for (int voxeli=0;voxeli<nb_voxels;voxeli++){
            v_dis[voxeli]/=nbv;
        }

        */

        for (int voxel=0;voxel<nb_voxels;voxel++){
            v_dis[voxel] = randomMass *((double)rand() / (double)(RAND_MAX)) * 2.0/nb_voxels;
        }
        fill_balls_from_voxelsDistribution(voxels_balls,nb_voxels,nb_balls,v_dis,b_dis);
        char X_path[50];
        sprintf(X_path, "input/training_data/%d/X0.txt", i);

        FILE *X_file;

        // Open the file for writing
        X_file = fopen(X_path, "w");
        if (X_file == NULL) {
            printf("Error opening file!\n");
            return 1;
        }
        for (int line=0;line<nb_balls;line++){
            fprintf(X_file,"%e \n",b_dis[line]);
        }
        fclose(X_file);
        int c;
        char Xj_path[50];

        for (int j=0;j<nb_steps;j++){
            multiply_exp_diff_matrix_vector(Dc,dt,voxels_Adjacency,voxels_degree,nb_voxels,v_dis,temp_v);

            if ((j+1)%100==0){
                fill_balls_from_voxelsDistribution(voxels_balls,nb_voxels,nb_balls,v_dis,b_dis);

                sprintf(Xj_path, "input/training_data/%d/X%d.txt", i,(int)((j+1)/100));
                FILE *Xj_file;
                // Open the file for writing
                Xj_file = fopen(Xj_path, "w");
                if (Xj_file == NULL) {
                    printf("Error opening file!\n");
                    return 1;
                }
                for (int iiiiivoxel=0;iiiiivoxel<nb_balls;iiiiivoxel++){
                    fprintf(Xj_file,"%e \n",b_dis[iiiiivoxel]);
                }
                fclose(Xj_file);
            }
            
        }
    }

    printf("done!\n");

    return 0;

}
