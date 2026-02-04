#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Struttura che rappresenta una matrice:
    Prende come parametro il numero di righe e colonne e
    un puntatore a puntatore di tipo float (costrutto necessario per strutture di dimensione dinamiche come le matrici). */
typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

/* Funzione: create_Matrix:
    prende come parametro il numero di riche e colonne e
    inizializza una matrice rows x cols. */
Matrix* create_Matrix(int rows, int cols); 

/* Funzione: fill_Matrix:
    prende come parametro una matrice e un array di valori,
    e riempie la matrice con i valori dell'array. */
int fill_Matrix(Matrix* matrix, float* values);

/* Funzione: free_Matrix:
    libera la memeoria allocata per una matrice. */
void free_Matrix(Matrix* matrix);

/* Funzione: print_Matrix
    stampa a schermo una matrice. */
void print_Matrix(Matrix* matrix);

/* Creazione di modelli di matrici fondamentali */

/* Funzione: identity_Matrix:
    restituisce una matrice identità di dimensione size x size. */
Matrix* identity_Matrix(int size);

/* Funzione zeros:
    restituisce una matrice di zeri di dimensione size x size*/
Matrix* zeros(int size);

/* Funzione ones:
    restituisce una matrice di uno di dimensione size x size*/
Matrix* ones(int rows, int cols);

/* Funzione diag:
    restituisce una matrice diagonale con i valori dell'array values contenuti sulla diagonale. */
Matrix* diag(int size, float* values);

/* Funzione M2diag:
    crea una diagonale con gli elementi della diagonale della matrice data. */
Matrix* M2Diag(Matrix *matrix);

/* Funzione: transpose_Matrix:
    restituisce una matrice trasposta di una matrice data. */
Matrix* transpose_Matrix(Matrix* matrix);

/* Funzione: scalar_multiply_matrix
    moltiplico una matrice per uno scalare. */
void scalar_multiply_matrix(float scalar, Matrix* matrix);

/* Funzione Matrix_add_matrix
    somma due matrici. */
Matrix* matrix_add_matrix(Matrix* A, Matrix* B);

/* Funzione Matrix_subtract_matrix
    sottrae due matrici. */
Matrix* matrix_subtract_matrix(Matrix* A, Matrix* B);

/* Funzione: matrix_multiply_matrix
    moltiplico una matrici con un'altra matrice o un vettore. */
Matrix* matrix_multiply_matrix(Matrix *A, Matrix *B);

/* Funzione: determinant_Matrix:
    calcola il determinante di una matrice con il metodo classico.*/
float determinant_Matrix(Matrix* matrix);

/* Funzione: inverse_Matrix:
    calcola la matrice inverse di una matrice data con il metodo di Gauss Jordan.*/
Matrix* inverse_Matrix(Matrix* matrix);

/* Funzione tr:
    Traccia di una matrice*/
float tr(Matrix* m);

/* Norma2 vettore*/
float norm2_vector(Matrix* v);

/* Funzione eigenvalues:
    calcola gli autovalori di una matrice A.*/
float* eigenvalues(Matrix* A,int maxIterations);

/* Funzione norma1:
    calcola la norma uno per una Matrice m*/
float norm1(Matrix* m);

/* Funzione normaInf:
    calcola la norma infinito per una Matrice m*/
float normInf(Matrix* m);

/* Funzione norm2: 
    calcola la norma 2 della matrice data*/
float norm2(Matrix* m);

/* Funzione frobenius_norm:
    calcola la norma di Frobenius per una Matrice m*/
float frobenius_norm(Matrix* m);

/* Funzione solve_linear_system:
    risolve un sistema lineare Ax = b come x = A^(-1)*b.*/
Matrix* solve_linear_system_backslash(Matrix* A, Matrix* b);

/* Funzione solve_linear_system_cramer:
    risolve un sistema lineare Ax=b con il metodo di Cramer*/
Matrix* solve_linear_system_cramer(Matrix* A, Matrix* b);

/* Funzione solve_linear_system_gaussE:
    risolve un sistema lineare Ax = b usando il metodo di eliminazione di Gauss.*/
Matrix* solve_linear_system_gaussE(Matrix *A, Matrix *b);

/* Funzione LUDecomposition:
    esegue la decomposizione LU di una matrice e/o la decomposizione LU con pivoting.*/
void LUDecomposition(Matrix* A, Matrix** L, Matrix** U, int* P);

/* Funzione solve_LUP:
    risolve un sistema lineare Ax = b con la decomposizione LU con pivoting. */
Matrix* solve_LUP(Matrix* L, Matrix* U, int* P, Matrix* b);

// Applica H * A (riflessione a sinistra)
void apply_householder_left(Matrix* A, float* v, int n_v, int start_row, int start_col);

// Applica A * H (riflessione a destra per aggiornare Q o riassemblare RQ)
void apply_householder_right(Matrix* A, float* v, int n_v, int start_row, int start_col);
/* Funzione QRMethod:
    Implementazione del metodo QR per calcolare gli autovalori */
void QRMethod(Matrix* A, int maxIterations) ;

/* Funzione QRDecomposition:
    decompone una matrice in una matrice triangolare superiore e una matrice ortogonale. */
void QRDecomposition(Matrix* A, Matrix* Q, Matrix* R);

/* Funzione apply_householder_vector:
    Applica la riflessione sul vettore dei termini noti, sostitutivo di apply_right per sistemi a 32bit.*/
void apply_householder_vector(float* target, float* v, int n_v, int start_idx);

/* Funzione: solveQR:
    risolve un sistema lineare con la fattorizzazione QR.*/
Matrix* solveQR(Matrix* A, Matrix* b);

/* Funzione solveQtbR:
    risolve un sistema lineare con la fattorizzazione QR senza creare esplicitamente la matrice Q,
    ma applicando le riflessioni direttamente al vettore dei termini noti.*/
void solveQtbR(Matrix* A, float* b, float* x);

/* Funzione translationMatrix:
    restituisce una matrice di traslazione omogenea in 2D/3D*/
Matrix *traslationMatrix(int size, Matrix *tvector);

/* Funzioni zRotationMatrix:
    creano una matrice di rotazione non omogenea intorno all'asse z. */
Matrix* zRotationMatrix(float angle);

/* Funzioni xRotationMatrix:
    creano una matrice di rotazione non omogenea intorno all'asse x. */
Matrix* xRotationMatrix(float angle);

/* Funzioni yRotationMatrix:
    creano una matrice di rotazione non omogenea intorno all'assey. */
Matrix* yRotationMatrix(float angle);

/* Funzioni zRotationMatrix3D:
    crea una matrice di rotazione omogenea in 3D. */
Matrix* zRotationMatrix3D(float angle);

/* Funzioni xRotationMatrix3D:
    crea una matrice di rotazione omogenea in 3D. */
Matrix* xRotationMatrix3D(float angle); 

/* Funzioni yRotationMatrix3D:
    crea una matrice di rotazione omogenea in 3D. */
Matrix* yRotationMatrix3D(float angle);

/* Funzione eulerRotationMatrix3D:
    Compone una matrice di rotazione secondo la convenzione degli angoli di eulero RPY*/
Matrix* eulerRotationMatrix3D(float alpha, float beta, float gamma);

/* Funzione ZYZRotationMatrix3D:
    Compone una matrice di rotazione secondo la convenzione degli angoli di ZYZ*/
Matrix* ZYZRotationMatrix3D(float alpha, float beta, float gamma);

/* Funzione norm:
    Calcola la norma di un vettore. */
float norm(Matrix *v);

/* Funzione dot_product:
    calcola il prodotto scalare tra due vettori 3x1*/
float dot_product(Matrix* a, Matrix* b);

/* Funzione cross_product:
    calcola il prodotto vettoriale tra due vettori 3x1.*/
Matrix* cross_product(Matrix* a, Matrix* b);

/* Funzione cross_product2D:
    calcola il prodotto vettoriale nel piano. */
float cross_product2D(Matrix* a, Matrix* b);

#endif // MATRIX_H