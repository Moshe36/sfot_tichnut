#include <chrono> 
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <immintrin.h>
#include <thread>
#include <numeric>

using namespace std;
using namespace std::chrono;

#define epsilon 0.0001
ofstream f1("matrix.txt");

//declaration of the functions
void sumMatrices(vector<vector<double> >& mat1, vector<vector<double> >& mat2);
void printResult(vector<vector<double> >& matRes);
void mulMatWithFactorial(vector<vector<double> >& mat1, vector<vector<double> >& mat2, double factorialValue);
void initializeMatrix(vector<vector<double> >& mat);
void initializeIdenticalMatrix(vector<vector<double> >& mat);
bool checkIfTheMatrixIsDiagonal(vector<vector<double> >& mat);
void calculateExpoMatrixWithDiagonalMatrix(vector<vector<double> >& mat1, vector<vector<double> >& mat2, double e);
bool readMatrixFromFile(vector<vector<double> >& mat);
bool powerMat(vector<vector<double> >& mat1, vector<vector<double> >& mat2, vector<vector<vector<double>>>& list, int powNum);
double calcNormOfMatrix(vector<vector<double> >& mat);
vector<vector<double>> multMat(vector<vector<double> >& mat1, vector<vector<double> >& mat2);
void checkMatrixSize();

//functions for multiply matices with threads and blocks
float* mult_thread(int Z, const float* A, const float* B, int lda, int ldb, int ldc);
void mult(int Z, int K, int N, const float* A, const float* B, float* C, int lda, int ldb, int ldc);
void kernel(int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc);
void reorder(int K, const float* B, int ldb, float* B_tmp);
void init_c(int Z, int N, float* C, int ldc);

float M = 0; //size of matrix

int main() {
	auto start = std::chrono::high_resolution_clock::now(); // Starting the timer
	cout << "Starting the program....." << endl;
	cout << "-----------------------------------------------------" << endl;
	checkMatrixSize(); // check the matrix size first
	//declaration of the variables
	vector<vector<double>> inputMatrix(M, vector<double>(M));
	vector<vector<double>> powerMatrixResult(M, vector<double>(M));
	vector<vector<double>> mulFactorialMatrixResult(M, vector<double>(M));
	vector<vector<double>> finalMatrixResult(M, vector<double>(M));
	vector<vector<double>> identicalMatrix(M, vector<double>(M));
	vector<vector<vector<double>>> listOfMatrices;
	bool matrixIsNilpotent = false;
	double normMatrixValue = 0;
	double e = 2.71828182845904523536;
	double factorialValue = 1;
	double remainderOfTaylorSeries = 0;
	double c = 0;
	double ePowc = 0;
	int idx = 0;
	int count = 0;

	if (readMatrixFromFile(inputMatrix) == false) {// read the matrix from the file and check if its square matrix
		cout << "Matrix is not square so can not calculate matrix exponential!";
		goto endOfLoop;
	}
	listOfMatrices.push_back(inputMatrix); // add this matrix to the list of matrices
	initializeIdenticalMatrix(identicalMatrix); // create identical matrix

	//check if the matrix is diagonal - so we will have easier and faster compute
	if (checkIfTheMatrixIsDiagonal(inputMatrix)) {
		calculateExpoMatrixWithDiagonalMatrix(inputMatrix, finalMatrixResult, e);
		goto endOfLoop;
	}

	//Check how many loops is needes until remainder is smaller than epsilon
	normMatrixValue = calcNormOfMatrix(inputMatrix); // find the norm 2 of the matrix
	c = normMatrixValue + epsilon;
	ePowc = pow(e, c); // e^c
	while (1) {
		if (idx == 0) {
			remainderOfTaylorSeries = ePowc * normMatrixValue;
		}
		else {
			remainderOfTaylorSeries = remainderOfTaylorSeries * (normMatrixValue / (idx + 1));
		}
		if (remainderOfTaylorSeries < epsilon) { // if the remainder is smaller than epsilon, we know how many iterations we need for the taylor series
			break;
		}
		idx++;
	}
	
	//This loop is for taylor series to calculate the value of the matrix exponential
	for (int i = 0; i < idx; i++) {
		if (i == 0) { // first we add identical matrix when the power is 0
			sumMatrices(finalMatrixResult, identicalMatrix); // summarize between this 2 matrices
		}

		if (i == 1) { // we add the matrix itself because the power is 1
			sumMatrices(finalMatrixResult, inputMatrix);
		}
		if (i > 1) {
			matrixIsNilpotent = powerMat(powerMatrixResult, inputMatrix, listOfMatrices, i);
			if (matrixIsNilpotent) { // it means that A^i is 0 for some integer, so the series terminates after a finite number
				goto endOfLoop; 
			}
			factorialValue = factorialValue * i;
			mulMatWithFactorial(mulFactorialMatrixResult, powerMatrixResult, factorialValue); // multiply (1/i) * matrix^i - like in the algorithm
			//Check if we can short our taylor series
			for (int j = 0; j < M; j++) {
				for (int k = 0; k < M; k++) {
					if (std::abs(mulFactorialMatrixResult[i][j]) <= 0.00001) {
						count++;
					}
				}
			}
			if (count == M * M) {
				goto endOfLoop; // It means that the ((1/i) * matrix^i) is matrix of 0.00001 so we can break the loop here
			}
			else {
				count = 0;
			}
			sumMatrices(finalMatrixResult, mulFactorialMatrixResult); // summarize it with the previous result
		}
	}

endOfLoop:
	auto end = std::chrono::high_resolution_clock::now();
	float elapsed_time = std::chrono::duration<float>(end - start).count();
	std::cout << "The time it takes to finish calculate the matrix exponential is " << elapsed_time << " seconds" << std::endl; // checking run time
	cout << "The final matrix result is: [" << finalMatrixResult[0][0] << ", " << finalMatrixResult[0][1] << ",......," << finalMatrixResult[999][998] << ", " << finalMatrixResult[999][999] << "]" << endl;
	printResult(finalMatrixResult); // It takes too long but we have the option to print the result matrix
	return 0;
}

//Summarize matrices
void sumMatrices(vector<vector<double> >& mat1, vector<vector<double> >& mat2) {
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			mat1[i][j] = mat1[i][j] + mat2[i][j];
}

//Print matrix
void printResult(vector<vector<double> >& matRes) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			//printf("%f\n", matRes[i][j]);
			f1 << matRes[i][j] << endl;
			/*if (j == M - 1) {
				printf("\n");
			}*/
		}
	}
}

// mutiply the matrix with scalar
void mulMatWithFactorial(vector<vector<double> >& mat1, vector<vector<double> >& mat2, double factorialValue) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			mat1[i][j] = mat2[i][j] * 1 / factorialValue;
		}
	}
}

//initialize matrix
void initializeMatrix(vector<vector<double> >& mat) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			mat[i][j] = 0;
		}
	}
}

//Check if the matrix is diagonal
bool checkIfTheMatrixIsDiagonal(vector<vector<double> >& mat) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			if (i != j && mat[i][j] != 0)
			{
				return false;
			}
		}
	}
	return true;
}

//If the matrix diagonal, we will calculate the exponential matrix by this easy way
void calculateExpoMatrixWithDiagonalMatrix(vector<vector<double> >& mat1, vector<vector<double> >& mat2, double e) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			if (i == j)
			{
				for (int k = 0; k < mat1[i][j]; ++k)// loop to calculate the pow of e^alpha
				{
					mat2[i][j] *= e;
				}
			}
		}
	}
}

//Read the input matrix from the file
bool readMatrixFromFile(vector<vector<double> >& mat) {
	int countElementsInEachLine = 0;

	ifstream f2("inv_matrix(1000x1000).txt");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			f2 >> mat[i][j];
			countElementsInEachLine++;
			if (f2.peek() == ',') {
				f2.ignore();
			}
		}
		if (M != countElementsInEachLine) {
			return false; // matrix is not square
		}
		countElementsInEachLine = 0;
	}
}

//check the matrix size from the input matrix from the file
void checkMatrixSize() {
	string line;   // To read each line from code
	int countRows = 0;

	//Check numbers of rows in matrix
	ifstream f("inv_matrix(1000x1000).txt");
	while (f.peek() != EOF)
	{
		getline(f, line);
		countRows++;
	}
	f.close();

	M = countRows;
}

//Create identical matrix
void initializeIdenticalMatrix(vector<vector<double> >& mat) {
	for (int i = 0; i < M; i++) {
		for (int k = 0; k < M; k++) {
			if (i == k) {
				mat[i][k] = 1;
			}
			else {
				mat[i][k] = 0;
			}
		}
	}
}

//Calculate the power of the matrix and check if we get zero matrix. if yes, A is nilpotent matrix 
bool powerMat(vector<vector<double> >& mat1, vector<vector<double> >& mat2, vector<vector<vector<double>>>& list, int powNum) {

	initializeMatrix(mat1); // initialize the matrix
	mat1 = multMat(mat2, list[powNum - 2]);
	list.push_back(mat1); // add A^i to the list so we will use it in the next this functions calls

	// check if after we we did A^i , the matrix is equal to 0 and it means that the matrix is nilpotent
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			if (mat1[i][j] != 0) {
				return false;
			}
		}
	}
	return true;
}

//Calculate the norm 2 of matrix
double calcNormOfMatrix(vector<vector<double> >& mat) {
	vector<vector<double>> transposeMatrix(M, vector<double>(M));
	vector<vector<double>> resultMulMatrices(M, vector<double>(M));
	vector<double> randomVector(M, 0);
	vector<double> resultVector(M, 0);
	double maximumValue = 0;
	double oldMaximumValue = 0;
	bool firstTime = true;
	double error = epsilon;

	//Generate random vector
	generate(randomVector.begin(), randomVector.end(), rand);

	// Computing transpose of the matrix
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < M; ++j) {
			transposeMatrix[j][i] = mat[i][j];
		}

	// Multiply A^T * A
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			resultMulMatrices[i][j] = 0;
			for (int k = 0; k < M; k++) {
				resultMulMatrices[i][j] += transposeMatrix[i][k] * mat[k][j];
			}
		}
	}
	//Power iteration algorithm to find the largest eigenvalue
	while (1) {
		//Multiply the matrix with the random vector
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				resultVector[i] += (resultMulMatrices[i][j] * randomVector[j]);
			}
		}
		//find maximum value in the resultVector
		maximumValue = fabs(resultVector[0]);
		for (int i = 1; i < M; i++) {
			if (fabs(resultVector[i]) > maximumValue) {
				maximumValue = fabs(resultVector[i]);
			}
		}
		// if its just the first loop of the while
		if (firstTime) {
			firstTime = false;
		} // check if its that same eigenvalue from the last iteration and its not the first loop of the while
		else if (fabs(maximumValue - oldMaximumValue) < error) {
			return sqrt(maximumValue); // return the value of the norm 2
		}

		oldMaximumValue = maximumValue; // its not the same so we save it for the next iteration

		//divide the result vector by this maximum number - and we will use it to multiply the matrix with this updated vector
		for (int i = 0; i < M; i++) {
			randomVector[i] = resultVector[i] / maximumValue;
			resultVector[i] = 0;
		}
	}
}

//multiply matrices
 vector<vector<double>> multMat(vector<vector<double> >& mat1, vector<vector<double> >& mat2) {
	vector<vector<double>> matRes(M, vector<double>(M));
	int size = M + 152;
    float* A = new float[size * size];
    float* B = new float[size * size];
    float* C;

	//copy the matrices into vectors, and padding them with 8 columns and rows of zeros, so the multiply will work
    for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i >= M || j >= M) {
				A[int(i * size + j)] = 0;
				B[int(i * size + j)] = 0;
			}
			else {
				A[int(i * size + j)] = mat1[i][j];
				B[int(i * size + j)] = mat2[i][j];
			}
		}
	}

	C = mult_thread(size, A, B, size, size, size);

	//copy the result vector to the result matrix
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			matRes[i][j] = C[int(i * size + j)];
			if (isnan(matRes[i][j]) || isinf(matRes[i][j])) { // if we get nan or inf, preventing it by changing this element in the result matrix to the recent value
				matRes[i][j] = mat2[i][j];
			}
		}
	}

	delete[] A;
	delete[] B;
	delete[] C;

	return matRes;
}

 // Zero initialization of the block (M x N) in the matrix
 void init_c(int Z, int N, float* C, int ldc)
 {
	 for (int i = 0; i < Z; ++i, C += ldc) {
		 for (int j = 0; j < N; j += 8) {
			 _mm256_storeu_ps(C + j, _mm256_setzero_ps());
		 }
	 }
}

 // Reordering of (K x 16) block of B
void reorder(int K, const float* B, int ldb, float* B_tmp)
{
	 for (int k = 0; k < K; ++k, B += ldb, B_tmp += 16)
		  {
		  _mm256_storeu_ps(B_tmp + 0, _mm256_loadu_ps(B + 0));
		  _mm256_storeu_ps(B_tmp + 8, _mm256_loadu_ps(B + 8));
		  }
 }

// Multiplication of (6 x K) block of A and (K x 16) block of B (B - reordered) and storeing it to(6 x 16) block in C
 void kernel(int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc)
 {
	 __m256 a0, a1, b0, b1;
	
	 __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
	 __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
	 __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
	 __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
	 __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
	 __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();
	
	 const int offset0 = lda * 0; const int offset3 = lda * 3;
	 const int offset1 = lda * 1; const int offset4 = lda * 4;
	 const int offset2 = lda * 2; const int offset5 = lda * 5;
	
		 for (int k = 0; k < K; k++)
		 {
		 b0 = _mm256_loadu_ps(B + 0);
		 b1 = _mm256_loadu_ps(B + 8);
		 a0 = _mm256_broadcast_ss(A + offset0); a1 = _mm256_broadcast_ss(A + offset1);
		 c00 = _mm256_fmadd_ps(a0, b0, c00); c10 = _mm256_fmadd_ps(a1, b0, c10);
		 c01 = _mm256_fmadd_ps(a0, b1, c01); c11 = _mm256_fmadd_ps(a1, b1, c11);
		 a0 = _mm256_broadcast_ss(A + offset2); a1 = _mm256_broadcast_ss(A + offset3);
		 c20 = _mm256_fmadd_ps(a0, b0, c20); c30 = _mm256_fmadd_ps(a1, b0, c30);
		 c21 = _mm256_fmadd_ps(a0, b1, c21); c31 = _mm256_fmadd_ps(a1, b1, c31);
		 a0 = _mm256_broadcast_ss(A + offset4); a1 = _mm256_broadcast_ss(A + offset5);
		 c40 = _mm256_fmadd_ps(a0, b0, c40); c50 = _mm256_fmadd_ps(a1, b0, c50);
		 c41 = _mm256_fmadd_ps(a0, b1, c41); c51 = _mm256_fmadd_ps(a1, b1, c51);
		  B += ldb; A++;
		 }
	
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
	 C += ldc;
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
	 C += ldc;
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
	 C += ldc;
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
	 C += ldc;
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
	 C += ldc;
	 _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
	 _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

// Product of matrices A (M x K) and B (K x N)
void mult(int Z, int K, int N, const float* A, const float* B, float* C, int lda, int ldb, int ldc)
{
	float* B_tmp = new float[K * 16];
	 
	for (int j = 0; j < N; j += 16)
	{
		 reorder(K, B + j, ldb, B_tmp);
		 for (int i = 0; i < Z; i += 6)
			  {
			  init_c(6, 16, C + i * ldc + j, ldc);
			  kernel(K, A + i * lda, B_tmp, C + i * ldc + j, lda, 16, ldc);
			  }
	}
	 
		  delete[] B_tmp;
}

// Multithreaded product of matrices A (M x K) and B (K x N)
 float* mult_thread(int Z, const float* A, const float* B, int lda, int ldb, int ldc) {
	 int n = 8; // number of threads
	 int m = ldb / n; //int m = M / n;
	 thread t[8];
	 float* C = new float[Z * ldc];
	
	 for (int i = 0; i < n; i++) {
		 t[i] = thread([&, i]() { mult(Z, lda, m, A, B + i * m, C + i * m, lda, ldb, ldc); });
	 }
		

	 for (int i = 0; i < n; i++) {
		 t[i].join();
     }
	 
    return C;
}




