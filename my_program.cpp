#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <string>

typedef std::vector<std::vector<double>> Matrix;

const double EPSILON = std::numeric_limits<double>::epsilon();
const int MAX_FACT = 170;
const int N = 10;
const double ERROR_MARGIN = 1e-5;

Matrix matrixFromFile(const std::string& filename)
{
    Matrix matrix;
    std::ifstream file(filename);
    if (file)
    {
        std::string line;
        while (std::getline(file, line, '\n'))
        {
            std::vector<double> row;
            size_t start = 0;
            size_t end = line.find(',', start);
            while (end != std::string::npos)
            {
                std::string numStr = line.substr(start, end - start);
                double num = std::strtod(numStr.c_str(), nullptr);
                row.push_back(num);
                start = end + 1;
                end = line.find(',', start);
            }
            std::string lastNumStr = line.substr(start);
            double lastNum = std::strtod(lastNumStr.c_str(), nullptr);
            row.push_back(lastNum);
            matrix.push_back(row);
        }
    }
    return matrix;
}

bool validateSquare(const Matrix& matrix)
{
    size_t numRows = matrix.size();
    size_t numCols = matrix[0].size();
    return (numRows == numCols);
}

void resultToFile(const Matrix& resultMatrix, const std::string& filename)
{
    std::ofstream file(filename);
    if (file)
    {
        for (const auto& row : resultMatrix)
        {
            for (size_t i = 0; i < row.size() - 1; i++)
            {
                file << row[i] << ",";
            }
            file << row.back() << "\n";
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

Matrix calculateExponential(const Matrix& matrix)
{
    size_t iterNum = 0;
    Matrix result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    std::vector<double> x, y;

    Matrix matrixPower(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); i++)
    {
        matrixPower[i][i] = 1.0;  // Initialize matrixPower with the identity matrix
    }
    double fact = 1.0;

    while (true)
    {
        if (iterNum > 0)
        {
            Matrix tempPower(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
            for (size_t i = 0; i < matrix.size(); i++)
            {
                for (size_t j = 0; j < matrix[i].size(); j++)
                {
                    double sum = 0.0;
                    for (size_t k = 0; k < matrix.size(); k++)
                    {
                        sum += matrixPower[i][k] * matrix[k][j];
                    }
                    tempPower[i][j] = sum;
                }
            }
            matrixPower = tempPower;
        }

        Matrix matrixFactor(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
        for (size_t i = 0; i < matrix.size(); i++)
        {
            for (size_t j = 0; j < matrix[i].size(); j++)
            {
                matrixFactor[i][j] = matrixPower[i][j] / fact;
            }
        }

        for (size_t i = 0; i < matrix.size(); i++)
        {
            for (size_t j = 0; j < matrix[i].size(); j++)
            {
                result[i][j] += matrixFactor[i][j];
            }
        }

        if (x.size() == N)
        {
            x.erase(x.begin());
            y.erase(y.begin());
        }
        x.push_back(iterNum);
        double sum = 0.0;
        for (size_t i = 0; i < matrix.size(); i++)
        {
            for (size_t j = 0; j < matrix[i].size(); j++)
            {
                sum += matrixFactor[i][j];
            }
        }
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
    Matrix inputMatrix = matrixFromFile("inv_matrix(1000x1000).txt");
    if (!validateSquare(inputMatrix))
    {
        std::cout << "Invalid input. The input matrix must be square." << std::endl;
        return 0;
    }

    std::clock_t startClock = std::clock();
    Matrix resultMatrix = calculateExponential(inputMatrix);
    std::clock_t endClock = std::clock();
    double elapsedTime = static_cast<double>(endClock - startClock) / CLOCKS_PER_SEC;
    std::cout << "\nCustom implementation took: " << elapsedTime << " seconds." << std::endl;
    std::cout << "Top-left and bottom-right elements of the resultant matrix are: ["
        << resultMatrix[0][0] << ", " << resultMatrix[0][1] << ", ..., "
        << resultMatrix.back()[resultMatrix.back().size() - 2] << ", "
        << resultMatrix.back().back() << "]" << std::endl;

    resultToFile(resultMatrix, "result.txt");

    std::cout << "\n------------------------------------" << std::endl;
    std::cout << "Execution completed successfully. Check the 'result.txt' file for the output matrix from the custom implementation." << std::endl;

    return 0;
}
