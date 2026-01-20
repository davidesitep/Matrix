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
Matrix* create_Matrix(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

/* Funzione: fill_Matrix:
    prende come parametro una matrice e un array di valori,
    e riempie la matrice con i valori dell'array. */
int fill_Matrix(Matrix* matrix, float* values) {
    int i, j;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = values[i * matrix->cols + j];
        }
    }
    return i*matrix->cols + j;
}

/* Funzione: free_Matrix:
    libera la memeoria allocata per una matrice. */
void free_Matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

/* Funzione: identity_Matrix:
    restituisce una matrice identità di dimensione size x size. */
Matrix* identity_Matrix(int size) {
    Matrix* identity = (Matrix*)malloc(sizeof(Matrix));
    identity->rows = size;
    identity->cols = size;
    identity->data = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++) {
        identity->data[i] = (float*)malloc(size * sizeof(float));
        for (int j = 0; j < size; j++) {
            identity->data[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    return identity;
}

/* Funzione zeros:
    restituisce una matrice di zeri di dimensione size x size*/
Matrix* zeros(int size) {
    Matrix *zeros = create_Matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            zeros->data[i][j] = 0.0f;
        }
    }
}

/* Funzione ones:
    restituisce una matrice di zeri di dimensione size x size*/
Matrix* ones(int size) {
    Matrix *zeros = create_Matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            zeros->data[i][j] = 1.0f;
        }
    }
}

/* Funzione diag:
    restituisce una matrice diagonale con i valori dell'array. */
Matrix* diag(int size, float* values) {
    Matrix *diag = create_Matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                diag->data[i][j] = *values++;
            }
            else {
                diag->data[i][j] = 0.0f;
            }
        }
    }
    return diag;
}

/* Funzione diag:
    estrae la diagonale da una matrice. */
Matrix* M2Diag(Matrix *matrix) {

    int n = matrix->rows < matrix->cols ? matrix->rows : matrix->cols;
    Matrix *diag = create_Matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                diag->data[i][j] = matrix->data[i][j];
            }
            else {
                diag->data[i][j] = 0.0f;
            }
        }
    }
    return diag;
}

/* Funzione: transpose_Matrix:
    restituisce una matrice trasposta di una matrice data. */
Matrix* transpose_Matrix(Matrix* matrix) {
    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = matrix->cols;
    result->cols = matrix->rows;
    result->data = (float**)malloc(result->rows * sizeof(float*));
    for (int i = 0; i < result->rows; i++) {
        result->data[i] = (float*)malloc(result->cols * sizeof(float));
        for (int j = 0; j < result->cols; j++) {
            result->data[i][j] = matrix->data[j][i];
        }
    }
    return result;
}

/* Funzione: determinant_Matrix:
    calcola il determinante di una matrice.*/
float determinant_Matrix(Matrix* matrix) {
    if (matrix->rows != matrix->cols) {
        // Matrice non quadrata, impossibile calcolare il determinante
        return 0.0f;
    }
    int n = matrix->rows;

    if (n == 1) {
        return matrix->data[0][0];
    }
    if (n == 2) {
        return matrix->data[0][0] * matrix->data[1][1] - matrix->data[0][1] * matrix->data[1][0];
    }

    float det = 0.0f;
    for (int p = 0; p < n; p++) {
        Matrix submatrix;
        submatrix.rows = n - 1;
        submatrix.cols = n - 1;
        submatrix.data = (float**)malloc(submatrix.rows * sizeof(float*));
        for (int i = 0; i < submatrix.rows; i++) {
            submatrix.data[i] = (float*)malloc(submatrix.cols * sizeof(float));
        }

        for (int i = 1; i < n; i++) {
            int colIndex = 0;
            for (int j = 0; j < n; j++) {
                if (j == p) continue;
                submatrix.data[i - 1][colIndex] = matrix->data[i][j];
                colIndex++;
            }
        }

        det += (p % 2 == 0 ? 1 : -1) * matrix->data[0][p] * determinant_Matrix(&submatrix);

        for (int i = 0; i < submatrix.rows; i++) {
            free(submatrix.data[i]);
        }
        free(submatrix.data);
    }
    return det;
}

/* Funzione: inverse_Matrix:
    calcola la matrice inverse di una matrice data.*/
Matrix* inverse_Matrix(Matrix* matrix) {
    // Metodo di Gauss-Jordan
    if (matrix->rows != matrix->cols) {
        // Matrice non quadrata, impossibile calcolare l'inversa
        //Matrix empty = {0, 0, NULL};
        return NULL;
    }
    if (determinant_Matrix(matrix) == 0.0f) {
        // Matrice singolare, impossibile calcolare l'inversa
        //Matrix empty = {0, 0, NULL};
        return NULL;
    }
    int n = matrix->rows;

    // Realizzo la matrice aumentata [A | I]
    Matrix* identity = identity_Matrix(n);
    Matrix augmented;
    augmented.rows = n;
    augmented.cols = 2 * n;
    augmented.data = (float**)malloc(augmented.rows * sizeof(float*));

    for (int i = 0; i < augmented.rows; i++) {
        augmented.data[i] = (float*)malloc(augmented.cols * sizeof(float));
        for (int j = 0; j < augmented.cols; j++) {
            if (j < n)
                augmented.data[i][j] = matrix->data[i][j];
            else 
                augmented.data[i][j] = identity->data[i][j-n];
        }
    }
    
    // Selezione del pivot nessuno deve essere zero
    for (int i = 0; i < n; i++) {
        if (augmented.data[i][i] == 0.0f) {
            // Scambio con una riga sottostante
            for (int k = i + 1; k < n; k++) {
                if (augmented.data[k][i] != 0.0f) {
                    float* temp = augmented.data[i];
                    augmented.data[i] = augmented.data[k];
                    augmented.data[k] = temp;
                    break;
                }
            }
        }

        // Normalizzo la riga del pivot
        float pivot = augmented.data[i][i];
        for (int j = 0; j < augmented.cols; j++) {
            augmented.data[i][j] /= pivot;
        }

        // Elimino le altre righe
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = augmented.data[k][i];
                for (int j = 0; j < augmented.cols; j++) {
                    augmented.data[k][j] -= factor * augmented.data[i][j];
                }
            }
        }
    }

    // Estraggo la matrice inversa dalla matrice aumentata
    Matrix* inverse = (Matrix*)malloc(sizeof(Matrix));
    inverse->rows = n;
    inverse->cols = n;
    inverse->data = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        inverse->data[i] = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            inverse->data[i][j] = augmented.data[i][j + n];
        }
    }

    // Libera la memoria allocata per la matrice aumentata
    for (int i = 0; i < augmented.rows; i++) {
        free(augmented.data[i]);
    }
    free(augmented.data);

    return inverse;
}

/* Funzione: matrix_multiply_matrix
    moltiplico una matrici con un'altra matrice o un vettore. */
Matrix *matrix_multiply_matrix(Matrix *A, Matrix *B)
{

    if (A->cols != B->rows)
    {
        // Dimensione disallineate, impossibile eseguire la moltiplicazione
        return NULL;
    }

    Matrix *result = (Matrix *)malloc(sizeof(Matrix));
    result->rows = A->rows;
    result->cols = ((Matrix *)B)->cols;
    result->data = (float **)malloc(result->rows * sizeof(float *));
    for (int i = 0; i < A->rows; i++)
    {

        result->data[i] = (float *)malloc(result->cols * sizeof(float));
        for (int j = 0; j < ((Matrix *)B)->cols; j++)
        {

            result->data[i][j] = 0;
            for (int k = 0; k < A->cols; k++)
            {
                result->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return result;
}

/* Funzione: scalar_multiply_matrix
    moltiplico una matrice per uno scalare. */
void scalar_multiply_matrix(float scalar, Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] *= scalar;
        }
    }
}

/* Funzione Matrix_add_matrix
    somma due matrici. */
Matrix* matrix_add_matrix(Matrix* A, Matrix* B) {
    if(A->rows != B->rows || A->cols != B->cols) {
        // Dimensione disallineate, impossibile eseguire l'addizione
        return NULL;
    }

    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = A->rows;
    result->cols = A->cols;
    result->data = (float**)malloc(result->rows * sizeof(float*));
    for (int i = 0; i < result->rows; i++) {
        result->data[i] = (float*)malloc(result->cols * sizeof(float));
        for (int j = 0; j < result->cols; j++) {
            result->data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
    return result;
}

/* Funzione Matrix_subtract_matrix
    sottrae due matrici. */
Matrix* matrix_subtract_matrix(Matrix* A, Matrix* B) {
    if(A->rows != B->rows || A->cols != B->cols) {
        // Dimensione disallineate, impossibile eseguire la sotttrazione
        return NULL;
    }

    Matrix* result = (Matrix*)malloc(sizeof(Matrix));
    result->rows = A->rows;
    result->cols = A->cols;
    result->data = (float**)malloc(result->rows * sizeof(float*));
    for (int i = 0; i < result->rows; i++) {
        result->data[i] = (float*)malloc(result->cols * sizeof(float));
        for (int j = 0; j < result->cols; j++) {
            result->data[i][j] = A->data[i][j] - B->data[i][j];
        }
    }
    return result;
}

/* Funzione solve_linear_system:
    risolve un sistema lineare Ax = b come x = A^(-1)*b.*/
Matrix* solve_linear_system_backslash(Matrix* A, Matrix* b) {
    if (b->cols != 1 || A->rows != b->rows) {
        return NULL;
    }
    Matrix* invA = inverse_Matrix(A);
    if (invA == NULL) {
        return NULL;
    }
    Matrix* x = matrix_multiply_matrix(invA, b);
    free_Matrix(invA);
    return x;
}

/* Funzione solve_linear_system_gaussE:
    risolve un sistema lineare Ax = b usando il metodo di eliminazione di Gauss.*/
Matrix* solve_linear_system_gaussE(Matrix *A, Matrix *b) {
    if (A->rows != b->rows || b->cols != 1) {
        return NULL;
    }

    int n = A->rows;
    // 1. Crea la matrice aumentata [A | b]
    Matrix *aug = create_Matrix(n, A->cols + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < A->cols; j++) {
            aug->data[i][j] = A->data[i][j];
        }
        aug->data[i][A->cols] = b->data[i][0];
    }

    // 2. Applica l'eliminazione di Gauss
    for (int i = 0; i < n; i++) {
        // Pivot check e swap (Pivoting parziale)
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(aug->data[k][i]) > fabs(aug->data[maxRow][i])) maxRow = k;
        }
        // Se la riga con il massimo elemento della colonna i-esima non è quella corrente, scambio le righe.
        if (maxRow != i) {
            float* temp = aug->data[i];
            aug->data[i] = aug->data[maxRow];
            aug->data[maxRow] = temp;
        }

        // Elimina (rende zero) gli elementi sotto il pivot
        for (int k = i + 1; k < n; k++) {
            float factor = aug->data[k][i] / aug->data[i][i];
            for (int j = i; j < A->cols + 1; j++) {
                aug->data[k][j] -= factor * aug->data[i][j];
            }
        }
    }

    // 3. Back substitution (Sostituzione all'indietro)
    
    Matrix *x = create_Matrix(n, 1);
    for (int i = n - 1; i >= 0; i--) {
        x->data[i][0] = aug->data[i][aug->cols - 1];
        for (int j = i + 1; j < n; j++) {
            x->data[i][0] -= aug->data[i][j] * x->data[j][0];
        }
        x->data[i][0] /= aug->data[i][i];
    }
    free_Matrix(aug);
    return x;
}

/* Funzione LUDecomposition:
    esegue la decomposizione LU di una matrice e/o la decomposizione LU con pivoting.*/
void LUDecomposition(Matrix* A, Matrix** L, Matrix** U, int* P) {
    
    if (A == NULL || L == NULL || U == NULL || P == NULL) {
        return;
    }

    if (A->rows != A->cols) {
        // Matrice non quadrata, impossibile eseguire la decomposizione LU
        return;
    }

    int n = A->rows;

    // Inizializzo L come matrice identità e U come copia di A
    *L = identity_Matrix(A->rows);
    *U = create_Matrix(A->rows, A->cols);

    // COPIA i valori (non solo il puntatore!)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*U)->data[i][j] = A->data[i][j];
        }
    }

    for (int i = 0; i < n; i++) P[i] = i;

    for (int i = 0; i < n; i++) {
        float maxA = 0.0;
        int maxRow = i;

        // Pivoting parziale
        for (int k = i; k < n; k++) {
            if (fabs((*U)->data[k][i]) > maxA) {
                maxA = fabs((*U)->data[k][i]);
                maxRow = k;
            }
        }
        // Matrice singolare
        if (maxA < 1e-12) continue;
                
        // Se la riga con il massimo elemento della colonna i-esima non è quella corrente, scambio le righe.
        int* tempP = P[i];
        P[i] = P[maxRow];
        P[maxRow] = tempP;

        float* tempU = (*U)->data[i];
        (*U)->data[i] = (*U)->data[maxRow];
        (*U)->data[maxRow] = tempU;
        
        // Scambio le righe di L per mantenere la coerenza
        for (int j = 0; j < i; j++) {
            float tempL = (*L)->data[i][j];
            (*L)->data[i][j] = (*L)->data[maxRow][j];
            (*L)->data[maxRow][j] = tempL;
        }
        // Azzeramento sotto il pivot
        for (int j = i + 1; j < n; j++) {
            float factor = (*U)->data[j][i] / (*U)->data[i][i];
            (*L)->data[j][i] = factor;
            for (int k = i; k < n; k++) {
                (*U)->data[j][k] -= factor * (*U)->data[i][k];
            }
        }    
    }
}

/* Funzione solve_LUP:
    risolve un sistema lineare Ax = b con la decomposizione LU con pivoting. */
Matrix* solve_LUP(Matrix* L, Matrix* U, int* P, Matrix* b) {

    int n = L->rows;
    Matrix* x = create_Matrix(n, 1); // Lo chiamiamo x per chiarezza

    // 1. Forward substitution Ly = Pb
    // Risolviamo direttamente usando b[P[i]] per evitare di creare Pb
    for (int i = 0; i < n; i++) {
        x->data[i][0] = b->data[P[i]][0];
        for (int k = 0; k < i; k++) {
            x->data[i][0] -= L->data[i][k] * x->data[k][0];
        }
        // Se L non ha 1 sulla diagonale, aggiungi: 
        // x->data[i][0] /= L->data[i][i];
    }

    // 2. Back substitution Ux = y (x sovrascrive y)
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            x->data[i][0] -= U->data[i][j] * x->data[j][0];
        }
        
        // Controllo divisione per zero (matrice singolare)
        if (fabs(U->data[i][i]) < 1e-9) {
            fprintf(stderr, "Errore: Matrice singolare (U[%d][%d] è zero)\n", i, i);
            return x; 
        }
        x->data[i][0] /= U->data[i][i];
    }

    return x;
}

/* Funzione solve_least_mean_squares:
    risolve un sistema lineare Ax = b usando il metodo dei minimi quadrati,
    A^t*A*x = A^t*b => x=(A^t*A)^(-1)*A^t*b*/
Matrix* solve_least_meansquare(Matrix *A, Matrix* b) {
    if (A->rows != b->rows || b->cols != 1) {
        return NULL;
    }

    Matrix *At = transpose_Matrix(A);
    Matrix *AtA = matrix_multiply_matrix(At, A);
    Matrix *invAtA = inverse_Matrix(AtA);
    Matrix *Atb = matrix_multiply_matrix(At, b);
    Matrix *x = matrix_multiply_matrix(invAtA, Atb);
    return x;
}

/* Funzione translationMatrix:
    restituisce una matrice di traslazione omogenea in 2D/3D*/

Matrix *traslationMatrix(int size, Matrix *tvector)
{

    Matrix *T = identity_Matrix(size + 1);
    switch (size)
    {
    case 2:
    case 3:
        for (int i = 0; i < size; i++)
        {
            T->data[i][size] = tvector->data[i][0];
        }
        break;
    default:
        T->data = NULL;
        break;
    }

    return T;
}

// Funzioni che creano matrici di rotazione omogenee in 2D
Matrix* zRotationMatrix2D(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[0][0] = cos(angle);
    R->data[0][1] = -sin(angle);
    R->data[1][0] = sin(angle);
    R->data[1][1] = cos(angle);
    return R;
} 

Matrix* xRotationMatrix2D(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[1][1] = cos(angle);
    R->data[1][2] = -sin(angle);
    R->data[2][1] = sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

Matrix* yRotationMatrix2D(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[0][0] = cos(angle);
    R->data[0][2] = sin(angle);
    R->data[2][0] = -sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

// Funzioni che creano matrici di rotazione omogenee in 2D/3D
Matrix* zRotationMatrix2D(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[0][0] = cos(angle);
    R->data[0][1] = -sin(angle);
    R->data[1][0] = sin(angle);
    R->data[1][1] = cos(angle);
    return R;
} 

Matrix* zRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[0][0] = cos(angle);
    R->data[0][1] = -sin(angle);
    R->data[1][0] = sin(angle);
    R->data[1][1] = cos(angle);
    return R;
} 

Matrix* xRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[1][1] = cos(angle);
    R->data[1][2] = -sin(angle);
    R->data[2][1] = sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

Matrix* yRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[0][0] = cos(angle);
    R->data[0][2] = sin(angle);
    R->data[2][0] = -sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

// Composizioni di matrici di rotazione in 3D
/* Composizione con angoli di eulero RPY*/
Matrix* eulerRotationMatrix3D(float alpha, float beta, float gamma)
{
    Matrix* Rz = zRotationMatrix3D(gamma);
    Matrix* Ry = yRotationMatrix3D(beta);
    Matrix* Rx = xRotationMatrix3D(alpha);

    Matrix* Rzy = matrix_multiply_matrix(Rz, Ry);
    Matrix* R = matrix_multiply_matrix(Rzy, Rx);

    free_Matrix(Rz);
    free_Matrix(Ry);
    free_Matrix(Rx);
    free_Matrix(Rzy);

    return R;
}

/* Composizione con angoli ZYZ*/
Matrix* ZYZRotationMatrix3D(float alpha, float beta, float gamma)
{
    Matrix* Rz1 = zRotationMatrix3D(alpha);
    Matrix* Ry = yRotationMatrix3D(beta);
    Matrix* Rz2 = zRotationMatrix3D(gamma);

    Matrix* Rz1y = matrix_multiply_matrix(Rz1, Ry);
    Matrix* R = matrix_multiply_matrix(Rz1y, Rz2);

    free_Matrix(Rz1);
    free_Matrix(Ry);
    free_Matrix(Rz2);
    free_Matrix(Rz1y);

    return R;
}

/* Norma di un vettore*/
float norm(Matrix *v)
{
    if (v->cols != 1)
    {
        return -1;
    }
    float norm = 0.0f;
    for (int i = 0; i < v->rows; i++)
    {
        norm += v->data[i][0] * v->data[i][0];
    }
    return sqrt(norm);
}

/* Prodotto scalare */
float dot_product(Matrix* a, Matrix* b) {
    if (a->rows != 3 || a->cols != 1 || b->rows != 3 || b->cols != 1) {
        return 0;
    }

    float result = 0;
    for (int i = 0; i < 3; i++) {
        result += a->data[i][0] * b->data[i][0];
    }
    return result;
}

/* Prodotto vettoriale */
Matrix* cross_product(Matrix* a, Matrix* b) {
    if (a->rows != 3 || a->cols != 1 || b->rows != 3 || b->cols != 1) {
        return NULL;
    }

    Matrix* result = create_Matrix(3, 1);
    result->data[0][0] = a->data[1][0] * b->data[2][0] - a->data[2][0] * b->data[1][0];
    result->data[1][0] = a->data[2][0] * b->data[0][0] - a->data[0][0] * b->data[2][0];
    result->data[2][0] = a->data[0][0] * b->data[1][0] - a->data[1][0] * b->data[0][0];
    return result;
}

/* Funzione: print_Matrix
    stampa a schermo una matrice. */
void print_Matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}
#endif // MATRIX_H