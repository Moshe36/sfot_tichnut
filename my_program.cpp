#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <string>
#include "matrix_double.h"

//typedef std::vector<std::vector<double>> Matrix;

const double EPSILON = std::numeric_limits<double>::epsilon();
const int MAX_FACT = 170;
const int N = 10;
const double ERROR_MARGIN = 1e-5;

//Create identity matrix
Matrix createIdentityMatrix(int size) {
    Matrix identity(size, size, 0.0);

    for (int i = 0; i < size; i++) {
        identity.p[i * size + i] = 1.0;
    }

    return identity;
}
double calculateMatrixSum(const Matrix& matrix) {
    double sum = 0.0;
    for (int i = 0; i < matrix.rows * matrix.cols; i++) {
        sum += matrix.p[i];
    }
    return sum;
}

void resultToFile(const Matrix& resultMatrix, const std::string& filename) {
    std::ofstream file(filename);
    if (file) {
        for (int i = 0; i < resultMatrix.rows; i++) {
            for (int j = 0; j < resultMatrix.cols - 1; j++) {
                file << resultMatrix.p[i * resultMatrix.cols + j] << ",";
            }
            file << resultMatrix.p[i * resultMatrix.cols + resultMatrix.cols - 1] << "\n";
        }
    }
}

double lagrange(const std::vector<double>& x, const std::vector<double>& y, double xi)
{
    size_t n = x.size();
    double yi = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double p = 1.0;
        for (size_t j = 0; j < n; j++)
        {
            if (i != j)
            {
                p *= (xi - x[j]) / (x[i] - x[j]);
            }
        }
        yi += y[i] * p;
    }
    return yi;
}

Matrix calculateExponential(Matrix matrix)
{
    size_t iterNum = 0;
    Matrix result = createIdentityMatrix(matrix.cols);
    std::vector<double> x, y;

    Matrix matrixPower = createIdentityMatrix(matrix.cols);
    double fact = 1.0;

    while (true)
    {
        if (iterNum > 0)
        {
            matrixPower = matrixPower * matrix;
        }

        Matrix matrixFactor = matrixPower * (1.0 /  fact);
        result = result + matrixFactor;

        if (x.size() == N)
        {
            x.erase(x.begin());
            y.erase(y.begin());
        }
        x.push_back(iterNum);
        double sum = calculateMatrixSum(matrixFactor);

        y.push_back(sum);

        if (x.size() == N)
        {
            double estimate = lagrange(x, y, iterNum + 1);
            if (std::abs(estimate - y.back()) < ERROR_MARGIN)
            {
                break;
            }
        }

        iterNum++;
        fact *= iterNum;

        if (iterNum > MAX_FACT)
        {
            fact = 1e100;
        }
    }

    return result;
}


int main()
{
    std::cout << "Matrix Exponential Calculation Program\n------------------------------------" << std::endl;
    Matrix inputMatrix("inv_matrix(1000x1000).txt");

    if (inputMatrix.cols != inputMatrix.rows)
    {
        std::cout << "Invalid input. The input matrix must be square." << std::endl;
        return 0;
    }

    std::clock_t startClock = Get_Time();
    Matrix resultMatrix = calculateExponential(inputMatrix);
    std::clock_t endClock = Get_Time();
    double elapsedTime = endClock - startClock;
    std::cout << "\nCustom implementation took: " << elapsedTime << " seconds." << std::endl;
    
    std::cout << "Top-left and bottom-right elements of the resultant matrix are: ["
        << resultMatrix.p[0] << ", " << resultMatrix.p[1] << ", ..., "
        << resultMatrix.p[resultMatrix.cols  * resultMatrix.rows - 2] << ", "
        << resultMatrix.p[resultMatrix.cols * resultMatrix.rows - 1] << "]" << std::endl;

    resultToFile(resultMatrix, "result.txt");

    std::cout << "\n------------------------------------" << std::endl;
    std::cout << "Execution completed successfully. Check the 'result.txt' file for the output matrix from the custom implementation." << std::endl;

    return 0;
}
