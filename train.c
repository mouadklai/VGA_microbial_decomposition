#include "DC.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"



// Define initial learning rate and decay factor
#define INITIAL_LR 0.1
#define DECAY_FACTOR 0.5
#define DECAY_EPOCHS 10

// Function to calculate the learning rate based on current epoch
double calculate_learning_rate(int epoch) {
    double lr = INITIAL_LR;
    int num_decays = epoch / DECAY_EPOCHS;

    // Apply decay factor for each decay
    for (int i = 0; i < num_decays; i++) {
        lr *= DECAY_FACTOR;
    }

    return lr;
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

    
    printf("training ...\n");

    char* weights_filename="input/DC/weights.txt";
    save_Weights(alphas->weights,nb_edge,"input/DC/weights_i.txt");

    
    int load;
    printf("\nTo continue training ----> type 1    &   To restart training -----> type 0  : ");
    scanf("%d",&load);

    if(load){
        printf("loading weights...\n");
        load_Weights(alphas->weights,weights_filename);
    }

    
    double LEARNING_RATE = 0.01;

    FILE *err_file; // File pointer for error file
    double diffusion_coefficient=100950.0;
    double dt = 0.1/(60*60*24);
    double dt_long = 10.0/(60*60*24);
    int nb_steps = 100;
    // Open the file for writing
    err_file = fopen("input/DC/err.txt", "w");
    if (err_file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }



    
    int nb_training_data = 30*30;
    int batch_size = 4;
    double** input = (double**)calloc(nb_training_data,sizeof(double*));
    for (int i=0;i<nb_training_data;i++){
        input[i] = (double*)calloc(nb_balls,sizeof(double));
    }
    
    double** target = (double**)calloc(nb_training_data,sizeof(double*));
    for (int i=0;i<nb_training_data;i++){
        target[i] = (double*)calloc(nb_balls,sizeof(double));
    }

    double** output = (double**)calloc(batch_size,sizeof(double*));
    for (int i=0;i<batch_size;i++){
        output[i] = (double*)calloc(nb_balls,sizeof(double));
    }
    double** output_next = (double**)calloc(batch_size,sizeof(double*));
    for (int i=0;i<batch_size;i++){
        output_next[i] = (double*)calloc(nb_balls,sizeof(double));
    }

    char inp[50];
    char tar[50];

    printf("loading training data ... \n");
    int idx = 0;
    FILE *inp_file;
    FILE *tar_file;
    int line_count;
    double tm;
    int start_idx = 0;
    int end_idx = 30;
    for (int dis=start_idx;dis<end_idx;dis++){
        // filling input and target from the folder dis
        printf("dir %d of %d \n",dis,end_idx-1);
        for (int j=0;j<30;j++){
            sprintf(inp, "input/training_data/%d/X%d.txt", dis,j);
            sprintf(tar, "input/training_data/%d/X%d.txt", dis,j+1);
            inp_file = fopen(inp, "r");

            if (inp_file == NULL) {
                perror("Error opening file");
                return -1; // Exit program on error
            }
            tm = 0.0;
            line_count = 0;
            while (fscanf(inp_file, "%lf", &input[idx][line_count]) == 1) {
                tm += input[idx][line_count];
                line_count++;
            }

            fclose(inp_file);
        /*
        
            for (int i=0;i<nb_balls;i++){
                input[idx][i] /= tm;
            }
        */
            tar_file = fopen(tar, "r");

            if (tar_file == NULL) {
                perror("Error opening file");
                return -1; // Exit program on error
            }
            tm=0.0;
            line_count = 0;
            while (fscanf(tar_file, "%lf", &target[idx][line_count]) == 1) {
                tm+=target[idx][line_count];
                line_count++;
            }
            
            fclose(tar_file);
/*

            for (int i=0;i<nb_balls;i++){
                target[idx][i] /= tm;
            }
*/
            idx++;

        }
    }
    printf("done ! \n");

    int* indices = (int*)calloc(batch_size,sizeof(int));
    double lr;

    double err;


    for (int i=0;i<batch_size;i++){
        indices[i] =  rand() % nb_training_data;
    }
    for(int batch=0;batch<batch_size;batch++){
        for (int i=0;i<nb_balls;i++){
            output[batch][i] = input[indices[batch]][i];
        }
    }
    get_output_longdt(alphas,batch_size,indices,output,output_next,ballsSet,nb_balls,diffusion_coefficient,dt,nb_steps);
    err = error(batch_size,indices,target,output,nb_balls); 
    printf("error before training = %.16f\n",err);
    // Write error to file
    fprintf(err_file, "%d  %.16f\n", 0, err);
    int epochs = 1000;
    double* dloss_dalphaij = (double*)calloc(nb_edge,sizeof(double));
    for (int edge_index=0;edge_index<nb_edge;edge_index++){
        dloss_dalphaij[edge_index] = 0.0;
    }


    for (int epoch = 0; epoch < epochs; epoch++) {
        lr = calculate_learning_rate(epoch);
        
        //Train_step_ADAM(alphas,input,nb_balls,ballsSet,target,diffusion_coefficient,dt,LEARNING_RATE,m,v,beta1,beta2,epsilon,t);
        Train_step_implicit(alphas,dloss_dalphaij,input,batch_size,nb_training_data,nb_balls,ballsSet,target,diffusion_coefficient,dt_long,lr,0.4);

        if ((epoch+1)%1==0){
            for(int batch=0;batch<batch_size;batch++){
                for (int i=0;i<nb_balls;i++){
                    output[batch][i] = input[indices[batch]][i];
                }
            }
            //get_output(alphas,batch_size,indices,output,output_next,ballsSet,nb_balls,diffusion_coefficient,dt);
            get_output_longdt(alphas,batch_size,indices,output,output_next,ballsSet,nb_balls,diffusion_coefficient,dt,nb_steps);
            err = error(batch_size,indices,target,output_next,nb_balls); 
            printf("error = %.16f\n",err);
            // Write error to file
            fprintf(err_file, "%d  %.16f\n", epoch, err);
            char filename[256];
            sprintf(filename, "input/DC/weights_%d.txt", epoch);
            save_Weights(alphas->weights,nb_edge,filename);
        }

    }
        
    

    printf("done!\n");

    return 0;
}