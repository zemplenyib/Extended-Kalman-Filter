
#include "matrix.h"
//#include "cholesky_decompose.h"
#include <stdio.h>
#include <stdint.h>

// Set length of vehicle
const float L = 0.56;

// Set time step
const float T = 0.01;

// Set measurement time
const float T_meas = 0.2;

// SET length of data
#define length 2001

void matrix_print(const matrix_t* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
    printf("\r\n");
}


void covariance(matrix_t* R, int i) {
    if (i % (int)(T_meas/T) != 0) {
        float val = matrix_get(R, 0, 0) +20*T;
        matrix_set(R, 0, 0, val);
        matrix_set(R, 1, 1, val);
    }
    else {
        matrix_set(R, 0, 0, 0.0001);
        matrix_set(R, 1, 1, 0.0001);
    }
}

void Ljapunov_F(matrix_t* F, const matrix_data_t psi, const matrix_data_t v, const matrix_data_t gamma) {    
    //F->data[2] = v * (-sin(psi - gamma)) * T;
    //F->data[5] = v * cos(psi - gamma) * T;
    F->data[2] = v * (-sin(psi)) * T;
    F->data[5] = v * cos(psi) * T;
}

void predict(matrix_t *x_pred, const matrix_t *x, const matrix_t *u, const matrix_t *P, matrix_t *P_tmp,
    matrix_t *P_pred, matrix_t *F, matrix_data_t *aux, const matrix_t *Q, matrix_t *add){
    matrix_data_t v = u->data[0];
    matrix_data_t gamma = u->data[1];
    matrix_data_t psi_k_minus_1 = x->data[2];
    //matrix_set(add, 0, 0, v * cos(psi_k_minus_1 - gamma) * T);
    //matrix_set(add, 1, 0, v * sin(psi_k_minus_1 - gamma) * T);
    matrix_set(add, 0, 0, v * cos(psi_k_minus_1) * T);
    matrix_set(add, 1, 0, v * sin(psi_k_minus_1) * T);
    matrix_set(add, 2, 0, tan(gamma) * v * 1 / L * T);

    matrix_add(x_pred, x, add);
    //printf("x_pred = \n");
    //matrix_print(x_pred);

    Ljapunov_F(F, x->data[2], u->data[0], u->data[1]);

    matrix_mult(F, P, P_tmp, aux);

    matrix_mult_transb(P_tmp, F, P_pred);
    matrix_copy(P_pred, P_tmp);

    matrix_add(P_pred, P_tmp, Q);
}

void update(matrix_t* x, const matrix_t* x_pred, matrix_t* P, const matrix_t* P_pred, const matrix_t* y, const matrix_t* z,
    matrix_t *S, const matrix_t* H, const matrix_t* R, matrix_t *K, matrix_data_t* aux,
    matrix_t *Sinv, matrix_t* temp_HP, matrix_t *temp_PHt, matrix_t *temp_KHP) {

    matrix_sub(z, x_pred, y);                           // y = z-H(x)
    
    // S = H*P*H' + R
    matrix_mult(H, P_pred, temp_HP, aux);               // temp = H*P
    matrix_mult_transb(temp_HP, H, S);                  // S = temp*H'
    matrix_add_inplace(S, R);                           // S += R

    // K = P*H*S^-1
    int check = cholesky_decompose_lower(S);
    if (check != 0) {
        printf("S_k is not positive semi-definite!\n");
    }

    matrix_invert_lower(S, Sinv);                       // Sinv = S^-1
    matrix_mult_transb(P_pred, H, temp_PHt);            // temp = P*H'
    matrix_mult(temp_PHt, Sinv, K, aux);                // K = temp*Sinv
    
    //matrix_set(K, 2, 0, 0.0001);
    //matrix_set(K, 2, 1, 0.0001);
    //matrix_set(K, 1, 0, 0);
    //matrix_set(K, 1, 1, 0);

    // x = x + K*y
    matrix_copy(x_pred, x);
    matrix_multadd_rowvector(K, y, x);

    // P = P - K*H*P_pred
    matrix_mult(H, P_pred, temp_HP, aux);               // temp_HP = H*P
    matrix_mult(K, temp_HP, temp_KHP, aux);             // temp_KHP = K*temp_HP
    matrix_sub(P_pred, temp_KHP, P);                    // P -= temp_KHP
}


void main()
{
    /* BEGIN READING DATA FROM FILE*/
    FILE* myFile;
    float fValues[2*length];
    float x_noisy[length], y_noisy[length];
    float v[length], gamma[length];
    float tmp;
    int n = 0;

    // Reading from state_noisy.txt
    fopen_s(&myFile,"D:/BME/Four_wheel_drive_e-wehicle/MatlabSimulink/state_noisy.txt", "r");
    if (myFile == NULL) {
        printf("failed to open file\n");
        return 1;
    }

    while (fscanf_s(myFile, "%f", &tmp) == 1) {
        fValues[n++] = tmp;
        fscanf_s(myFile, ",");
    }
    fclose(myFile);

    for (int i = 0; i < length; i++) {
        x_noisy[i] = fValues[i];
        y_noisy[i] = fValues[i + length];
    }

    // Reading from input.txt
    n = 0;
    fopen_s(&myFile, "D:/BME/Four_wheel_drive_e-wehicle/MatlabSimulink/input.txt", "r");
    if (myFile == NULL) {
        printf("failed to open file\n");
        return 1;
    }

    while (fscanf_s(myFile, "%f", &tmp) == 1) {
        fValues[n++] = tmp;
        fscanf_s(myFile, ",");
    }
    fclose(myFile);

    for (int i = 0; i < length; i++) {
        v[i] = fValues[i];
        gamma[i] = fValues[i + length];
    }

    int i = 0;
    /*
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    i = length-3;
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    printf("x_noisy[%d]=%f\n", i, x_noisy[i]); i++;
    i = 0;
    printf("y_noisy[%d]=%f\n", i, y_noisy[i]); i++;
    printf("y_noisy[%d]=%f\n", i, y_noisy[i]); i++;
    printf("y_noisy[%d]=%.15f\n", i, y_noisy[i]); i++;
    i = length-3;
    printf("y_noisy[%d]=%f\n", i, y_noisy[i]); i++;
    printf("y_noisy[%d]=%f\n", i, y_noisy[i]); i++;
    printf("y_noisy[%d]=%.15f\n", i, y_noisy[i]); i++;
    i = 0;
    printf("v[%d]=%f\n", i, v[i]); i++;
    printf("v[%d]=%f\n", i, v[i]); i++;
    printf("v[%d]=%f\n", i, v[i]); i++;
    i = length-3;
    printf("v[%d]=%f\n", i, v[i]); i++;
    printf("v[%d]=%f\n", i, v[i]); i++;
    printf("v[%d]=%f\n", i, v[i]); i++;
    i = 0;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    i = length - 3;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    printf("gamma[%d]=%f\n", i, gamma[i]); i++;
    */
    
    /* END READING DATA FROM FILE*/

    float x_t[length], y_t[length], psi_t[length], P_t[length * 4];
    matrix_t x, x_pred, u, P, P_pred, P_tmp, F, Q, y, z, S, H, R, K, add, Sinv, temp_HP, temp_PHt, temp_KHP, temp_KF;
    matrix_data_t x_data[3 * 1] = {0,0,0};
    matrix_data_t x_pred_data[3 * 1] = { 0,0,0 };
    matrix_data_t u_data[2 * 1] = { 0,0};
    matrix_data_t H_data[2 * 3] = { 1,0,0,0,1,0 };
    matrix_data_t F_data[3 * 3] = { 1,0,0,0,1,0,0,0,1 };
    //matrix_data_t Q_data[3 * 3] = { 0.00001,0,0,0,0.00001,0,0,0,0.00001 };
    matrix_data_t Q_data[3 * 3] = { 0.001,0,0,0,0.001,0,0,0,0.001 };
    matrix_data_t P_data[3 * 3] = { 0,0,0,0,0,0,0,0,0 };
    matrix_data_t P_pred_data[3 * 3] = { 0,0,0,0,0,0,0,0,0 };
    matrix_data_t P_tmp_data[3 * 3] = { 0,0,0,0,0,0,0,0,0 };
    matrix_data_t aux[3 * 1] = { 0,0,0 };
    matrix_data_t add_data[3 * 1] = { 0,0,0 };
    matrix_data_t y_data[2 * 1] = { 0,0 };
    matrix_data_t z_data[2 * 1] = { 0,0 };
    matrix_data_t S_data[2 * 2] = { 0,0,0,0 };
    //matrix_data_t R_data[2 * 2] = { 1,0,0,1 };
    matrix_data_t R_data[2 * 2] = { 0.0001,0,0,0.0001 };       //R(1,1) = 0.1^2; R(2,2) = (10*pi/180)^2
    matrix_data_t K_data[3 * 2] = { 0,0,0,0,0,0 };
    matrix_data_t Sinv_data[2 * 2] = { 0,0,0,0 };
    matrix_data_t temp_HP_data[2 * 3] = { 0,0,0,0,0,0 };
    matrix_data_t temp_PHt_data[3 * 2] = { 0,0,0,0,0,0 };
    matrix_data_t temp_KHP_data[3 * 3] = { 0,0,0,0,0,0,0,0,0 };
    matrix_data_t temp_KF_data[3 * 3] = { 0,0,0,0,0,0,0,0,0 };

    matrix_init(&x, 3, 1, x_data);
    matrix_init(&x_pred, 3, 1, x_pred_data);
    matrix_init(&u, 2, 1, u_data);
    matrix_init(&H, 2, 3, H_data);
    matrix_init(&F, 3, 3, F_data);
    matrix_init(&Q, 3, 3, Q_data);
    matrix_init(&P, 3, 3, P_data);
    matrix_init(&P_pred, 3, 3, P_pred_data);
    matrix_init(&P_tmp, 3, 3, P_tmp_data);
    matrix_init(&y, 2, 1, y_data);
    matrix_init(&z, 2, 1, z_data);
    matrix_init(&S, 2, 2, S_data);
    matrix_init(&R, 2, 2, R_data);
    matrix_init(&K, 3, 2, K_data);
    matrix_init(&add, 3, 1, add_data);
    matrix_init(&Sinv, 2, 2, Sinv_data);
    matrix_init(&temp_HP, 2, 3, temp_HP_data);
    matrix_init(&temp_PHt, 3, 2, temp_PHt_data);
    matrix_init(&temp_KHP, 3, 3, temp_KHP_data);
    matrix_init(&temp_KF, 3, 3, temp_KF_data);


    for (int i = 0; i < length; i++) {
        matrix_set(&u, 0, 0, v[i]);
        matrix_set(&u, 1, 0, gamma[i]);

        predict(&x_pred, &x, &u, &P, &P_tmp, &P_pred, &F, &aux, &Q, &add);

        matrix_set(&z, 0, 0, x_noisy[i]);
        matrix_set(&z, 1, 0, y_noisy[i]);

        update(&x, &x_pred, &P, &P_pred, &y, &z, &S, &H, &R, &K, &aux, &Sinv, &temp_HP, &temp_PHt, &temp_KHP);
        x_t[i] = x.data[0];
        y_t[i] = x.data[1];
        psi_t[i] = x.data[2];
        P_t[i * 4] = P.data[0];
        P_t[i * 4 + 1] = P.data[1];
        P_t[i * 4 + 2] = P.data[3];
        P_t[i * 4 + 3] = P.data[4];

        //matrix_print(&P);

        covariance(&R,i);
    }
   
    //printf("x = %f\n", x.data[0]);
    //printf("y = %f\n", x.data[1]);
    //printf("gamma = %f\n", x.data[2]);


    // Writing state data to file
    fopen_s(&myFile, "D:/BME/Four_wheel_drive_e-wehicle/MatlabSimulink/state_predicted.txt", "w");
    if (myFile == NULL) {
        printf("failed to open file\n");
        return 1;
    }

    for (int i = 0; i < length; i++) {
        fprintf(myFile, "%f\n", x_t[i]);
    }
    for (int i = 0; i < length; i++) {
        fprintf(myFile, "%f\n", y_t[i]);
    }
    for (int i = 0; i < length; i++) {
        fprintf(myFile, "%f\n", psi_t[i]);
    }
    fclose(myFile);

    // Writing covariance matrix to file
    fopen_s(&myFile, "D:/BME/Four_wheel_drive_e-wehicle/MatlabSimulink/covariance.txt", "w");
    if (myFile == NULL) {
        printf("failed to open file\n");
        return 1;
    }

    for (int i = 0; i < length*4; i++) {
        fprintf(myFile, "%f\n", P_t[i]);
    }
}