#include "Matrix.h"


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

int fill_Matrix(Matrix* matrix, float* values) {
    int i, j;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = values[i * matrix->cols + j];
        }
    }
    return i*matrix->cols + j;
}

void free_Matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

void print_Matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

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

Matrix* zeros(int size) {
    Matrix *zeros = create_Matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            zeros->data[i][j] = 0.0f;
        }
    }
    return zeros;
}

Matrix* ones(int rows, int cols) {
    Matrix *ones = create_Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ones->data[i][j] = 1.0f;
        }
    }
    return ones;
}

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

void scalar_multiply_matrix(float scalar, Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] *= scalar;
        }
    }
}

Matrix* matrix_multiply_matrix(Matrix *A, Matrix *B)
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

float tr(Matrix* m) {
    if (m->cols != m->rows) return 0.0f; // Impossibile calcolare la tracci di una matrice non quadrata
    
    float trace = 0.0f;
    for (int i  = 0; i < m->cols; i++) {
        trace += m->data[i][i];
    }
    return trace;
}

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


float* eigenvalues(Matrix* A,int maxIterations) {
    
    float* eigvalues = (float*)malloc(A->rows * sizeof(float));

    Matrix* A_copy = create_Matrix(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A_copy->data[i][j] = A->data[i][j];
        }
    }

    QRMethod(A_copy, maxIterations);
    for (int i = 0; i < A->rows; i++) {
        if (A_copy->data[i][i] < 1e-6f && A_copy->data[i][i] > -1e-6f)
            eigvalues[i] = 0.0f;
        else
            eigvalues[i] = A_copy->data[i][i];
        
    }
    return eigvalues; // Gli autovalori sono sulla diagonale di A_copy
}

float norm1(Matrix* m) {
    float maxColSum = 0.0f;
    for (int j = 0; j < m->cols; j++) {
        float colSum = 0.0f;
        for (int i = 0; i < m->rows; i++) {
            colSum += fabs(m->data[i][j]);
        }
        if (colSum > maxColSum) {
            maxColSum = colSum;
        }
    }
    return maxColSum;
}

float normInf(Matrix* m) {
    float maxColSum = 0.0f;
    for (int i = 0; i < m->rows; i++) {
        float colSum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            colSum += fabs(m->data[i][j]);
        }
        if (colSum > maxColSum) {
            maxColSum = colSum;
        }
    }
    return maxColSum;
}

float norm2(Matrix* m) {
    Matrix* Mt = transpose_Matrix(m);
    Matrix* MtM = matrix_multiply_matrix(Mt, m);
    float eig_max = -1.0f;
    float* eigvalues = eigenvalues(MtM, 100);
    for (int i = 0; i < MtM->rows; i++) {
        if (eigvalues[i] > eig_max) {
            eig_max = eigvalues[i];
        }
    }
    free_Matrix(Mt);
    free_Matrix(MtM);
    free(eigvalues);
    return sqrtf(eig_max);
}

float frobenius_norm(Matrix* m) {
    float sum = 0.0f;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            sum += m->data[i][j] * m->data[i][j];
        }
    }
    return sqrt(sum);
}

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

Matrix* solve_linear_system_cramer(Matrix* A, Matrix* b) {
    if (A->rows != A->cols || A->rows != b->rows || b->cols != 1) {
        return NULL;
    }

    float detA = determinant_Matrix(A);
    if (fabs(detA) < 1e-12) {
        // Matrice singolare
        return NULL;
    }

    Matrix* x = create_Matrix(A->rows, 1);
    for (int i = 0; i < A->rows; i++) {
        Matrix* Ai = create_Matrix(A->rows, A->cols);
        for (int j = 0; j < A->rows; j++) {
            for (int k = 0; k < A->cols; k++) {
                if (k == i) {
                    Ai->data[j][k] = b->data[j][0];
                } else {
                    Ai->data[j][k] = A->data[j][k];
                }
            }
        }
        x->data[i][0] = determinant_Matrix(Ai) / detA;
        free_Matrix(Ai);
    }

    return x;
}

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

        if (fabs(aug->data[maxRow][i]) < 1e-12) {
            // Matrice singolare
            free_Matrix(aug);
            return NULL;
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
        int tempP = P[i];
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

void apply_householder_left(Matrix* A, float* v, int n_v, int start_row, int start_col) {
    for (int j = start_col; j < A->cols; j++) {
        float dot = 0;
        for (int i = 0; i < n_v; i++) {
            dot += v[i] * A->data[start_row + i][j];
        }
        for (int i = 0; i < n_v; i++) {
            A->data[start_row + i][j] -= 2.0f * v[i] * dot;
        }
    }
}

void apply_householder_right(Matrix* A, float* v, int n_v, int start_row, int start_col) {
    // Scorriamo TUTTE le righe di Q (da 0 a n-1)
    for (int i = 0; i < A->rows; i++) {
        float dot = 0;
        // Il prodotto scalare deve coinvolgere solo le colonne da start_col in poi
        for (int j = 0; j < n_v; j++) {
            dot += A->data[i][start_col + j] * v[j];
        }
        
        // Applichiamo la riflessione: A = A - 2 * (A*v) * v^T
        for (int j = 0; j < n_v; j++) {
            A->data[i][start_col + j] -= 2.0f * dot * v[j];
        }
    }
}

void apply_householder_vector(float* target, float* v, int n_v, int start_idx) {
    double dot = 0;
    // Calcola il prodotto scalare v^T * target[start_idx...]
    for (int i = 0; i < n_v; i++) {
        dot += (double)v[i] * (double)target[start_idx + i];
    }

    // Applica la riflessione: b = b - 2 * (v^T * b) * v
    for (int i = 0; i < n_v; i++) {
        target[start_idx + i] -= (float)(2.0 * dot * (double)v[i]);
    }
}

void QRMethod(Matrix* A, int maxIterations) {
    int n = A->rows;
    float tol = 1e-7f;

    float* v = malloc(n * sizeof(float));
    Matrix* Q = identity_Matrix(n);
    Matrix* A_next = create_Matrix(n, n);
    for (int iter = 0; iter < maxIterations; iter++) {

        // Q = I
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Q->data[i][j] = (i == j) ? 1.0f : 0.0f;

        // === Wilkinson shift ===
        float d = (A->data[n-2][n-2] - A->data[n-1][n-1]) * 0.5f;
        float b = A->data[n-1][n-2];
        float mu = A->data[n-1][n-1]
                 - (d >= 0 ? 1.0f : -1.0f) * b*b / (fabsf(d) + sqrtf(d*d + b*b));

        for (int i = 0; i < n; i++)
            A->data[i][i] -= mu;
        // === QR tramite Householder ===
        for (int j = 0; j < n - 1; j++) {

            // norma colonna
            float norm_x = 0.0f;
            for (int i = j; i < n; i++)
                norm_x += A->data[i][j] * A->data[i][j];
            norm_x = sqrtf(norm_x);

            if (norm_x < 1e-10f)
                continue;

            float sign = (A->data[j][j] >= 0.0f) ? 1.0f : -1.0f;
            v[0] = A->data[j][j] + sign * norm_x;

            int v_size = n - j;
            for (int i = 1; i < v_size; i++)
                v[i] = A->data[j + i][j];

            // normalizzazione v
            float norm_v = 0.0f;
            for (int i = 0; i < v_size; i++)
                norm_v += v[i] * v[i];
            norm_v = sqrtf(norm_v);

            for (int i = 0; i < v_size; i++)
                v[i] /= norm_v;

            // R = H * R
            apply_householder_left(A, v, v_size, j, j);

            // azzera sotto-diagonale (OBBLIGATORIO)
            for (int i = j + 1; i < n; i++)
                A->data[i][j] = 0.0f;

            // Q = Q * H   (QUESTO ERA IL TUO BUG)
            apply_householder_right(Q, v, v_size, 0, j);
        }

        // === A = R * Q + mu I ===
        Matrix* A_next = matrix_multiply_matrix(A, Q);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A->data[i][j] = A_next->data[i][j] + ((i == j) ? mu : 0.0f);

        

        // === convergenza ===
        float max_off = 0.0f;
        for (int i = 1; i < n; i++)
            for (int j = 0; j < i; j++)
                if (fabsf(A->data[i][j]) > max_off)
                    max_off = fabsf(A->data[i][j]);

        if (max_off < tol)
            break;
    }
    free_Matrix(A_next);
    free(v);
}

void QRDecomposition(Matrix* A, Matrix* Q, Matrix* R) {
    int m = A->rows; int n = A->cols;
    float* v = malloc(m * sizeof(float));

    // Q = I
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            Q->data[i][j] = (i == j) ? 1.0f : 0.0f;

    // QR tramite Householder
    for (int j = 0; j < m - 1; j++) {

        // norma colonna
        float norm_x = 0.0f;

        for (int i = j; i < m; i++)
            norm_x += A->data[i][j] * A->data[i][j];
        norm_x = sqrtf(norm_x);

        if (norm_x < 1e-10f)
            continue;

        float sign = (A->data[j][j] >= 0.0f) ? 1.0f : -1.0f;
        v[0] = A->data[j][j] + sign * norm_x;

        int v_size = m - j;
        for (int i = 1; i < v_size; i++)
            v[i] = A->data[j + i][j];

        // normalizzazione v
        float norm_v = 0.0f;
        for (int i = 0; i < v_size; i++)
            norm_v += v[i] * v[i];
        norm_v = sqrtf(norm_v);

        for (int i = 0; i < v_size; i++)
            v[i] /= norm_v;

        // R = H * R
        apply_householder_left(A, v, v_size, j, j);

        // azzera sotto-diagonale (OBBLIGATORIO)
        for (int i = j + 1; i < m; i++)
            A->data[i][j] = 0.0f;

        // Q = Q * H
        apply_householder_right(Q, v, v_size, 0, j);
    }

    // Copia A in R
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            R->data[i][j] = A->data[i][j];
    free(v);
}

Matrix* solveQR(Matrix* A, Matrix* b) {

    int m = A->rows;
    int n = A->cols;
    printf("Matrice A:\n");
    print_Matrix(A);
    printf("Vettore b:\n");
    print_Matrix(b);
    Matrix* Q = create_Matrix(m, n);
    Matrix* R = create_Matrix(m, n);

    QRDecomposition(A, Q, R);
    printf("Matrice Q:\n");
    print_Matrix(Q);
    Matrix* Qt = transpose_Matrix(Q);
    Matrix* Qt_b = matrix_multiply_matrix(Qt, b);
    printf("Matrice Qt:\n");
    print_Matrix(Qt);
    Matrix* x = create_Matrix(n, 1);

    for (int i = n - 1; i >= 0; i--) {
        x->data[i][0] = 0.0f;
        for (int j = i + 1; j < n; j++) {
            x->data[i][0] -= R->data[i][j] * x->data[j][0];
        }
        
        // Controllo divisione per zero (matrice singolare)
        if (fabs(R->data[i][i]) < 1e-9) {
            return x; 
        }
        x->data[i][0] += Qt_b->data[i][0];
        x->data[i][0] /= R->data[i][i];
    }
    return x;
}

void solveQtbR(Matrix* A, float* b, float* x) {
    int m = A->rows;
    int n = A->cols;
    float* v = malloc(m * sizeof(float));

    for (int j = 0; j < n; j++) { // Iteriamo sulle colonne
        // 1. Calcolo norma colonna (usando double per stabilità)
        double norm_x_sq = 0;
        for (int i = j; i < m; i++) {
            norm_x_sq += (double)A->data[i][j] * (double)A->data[i][j];
        }
        float norm_x = sqrt(norm_x_sq);

        if (norm_x < 1e-10f) continue;

        // 2. Costruzione vettore v
        float sign = (A->data[j][j] >= 0.0f) ? 1.0f : -1.0f;
        v[0] = A->data[j][j] + sign * norm_x;

        int v_size = m - j;
        for (int i = 1; i < v_size; i++) {
            v[i] = A->data[j + i][j];
        }

        // 3. Normalizzazione v (fondamentale in double)
        double norm_v_sq = 0;
        for (int i = 0; i < v_size; i++) {
            norm_v_sq += (double)v[i] * (double)v[i];
        }
        float norm_v = sqrt(norm_v_sq);
        for (int i = 0; i < v_size; i++) {
            v[i] /= norm_v;
        }

        // 4. Applica a destra sulla matrice A (diventa R)
        apply_householder_left(A, v, v_size, j, j);

        // 5. Applica al vettore b (diventa Q^T * b)
        apply_householder_vector(b, v, v_size, j);

        // Azzera sotto-diagonale per pulizia
        for (int i = j + 1; i < m; i++) A->data[i][j] = 0.0f;
    }

    // 6. Back-substitution: Risolve Rx = b_trasformato
    // x_i = (b_i - sum(R_ij * x_j)) / R_ii
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += (double)A->data[i][j] * (double)x[j];
        }
        x[i] = (float)(((double)b[i] - sum) / (double)A->data[i][i]);
    }

    free(v);
}

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

Matrix* zRotationMatrix(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[0][0] = cos(angle);
    R->data[0][1] = -sin(angle);
    R->data[1][0] = sin(angle);
    R->data[1][1] = cos(angle);
    return R;
} 

Matrix* xRotationMatrix(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[1][1] = cos(angle);
    R->data[1][2] = -sin(angle);
    R->data[2][1] = sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

Matrix* yRotationMatrix(float angle)
{
    Matrix* R = identity_Matrix(3);
    R->data[0][0] = cos(angle);
    R->data[0][2] = sin(angle);
    R->data[2][0] = -sin(angle);
    R->data[2][2] = cos(angle);
    return R;
} 

Matrix* zRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[0][0] = cos(angle);
    R->data[0][1] = -sin(angle);
    R->data[1][0] = sin(angle);
    R->data[1][1] = cos(angle);
    R->data[3][3] = 1;
    return R;
} 

Matrix* xRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[1][1] = cos(angle);
    R->data[1][2] = -sin(angle);
    R->data[2][1] = sin(angle);
    R->data[2][2] = cos(angle);
    R->data[3][3] = 1;
    return R;
} 

Matrix* yRotationMatrix3D(float angle)
{
    Matrix* R = identity_Matrix(4);
    R->data[0][0] = cos(angle);
    R->data[0][2] = sin(angle);
    R->data[2][0] = -sin(angle);
    R->data[2][2] = cos(angle);
    R->data[3][3] = 1;
    return R;
} 

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

float cross_product2D(Matrix* a, Matrix* b) {
    if (a->rows != 3 || a->cols != 1 || b->rows != 3 || b->cols != 1) {
        return 0.0f;
    }

    float result = 0.0;    
    result = a->data[0][0] * b->data[1][0] - a->data[1][0] * b->data[0][0];
    return result;
}