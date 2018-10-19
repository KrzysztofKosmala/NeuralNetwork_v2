//
// Created by Lenovo on 2018-03-21.
//

#ifndef ZADANIE2_MATRIX_H
#define ZADANIE2_MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include "Neuron.h"
class Matrix
{
private:

    int numRows;
    int numCols;

public:
    Matrix(int numRows, int numCols, bool isRandom);
    static std::shared_ptr<Matrix> multiply (std::shared_ptr<Matrix> n, std::shared_ptr<Matrix> p);
    void multiplyElementWise ( std::shared_ptr<Matrix> p);
    static std::shared_ptr<Matrix> substract (std::shared_ptr<Matrix> n, std::shared_ptr<Matrix> p);
    static std::shared_ptr<Matrix> substractElement (std::shared_ptr<Matrix> n, int p);
    static std::shared_ptr<Matrix> transpose(std::shared_ptr<Matrix> m);
    double getValue(int r, int c);
    double generateRandomNumber(double fMin, double fMax);
    double generateRandomNumber();
    int getNumRows();
    int getNumCols();
    void multiply (double n);
    void sigmoid();
    void disigmoid();
    void printToConsole();
    void printToConsole2();
    void add (double n);
    void setValue(int r, int c, double v);
    void add (std::shared_ptr<Matrix> p);
    void refactor (std::shared_ptr<Matrix> p);
    void save();
    std::vector< std::vector<Neuron> > values;
};




#endif //ZADANIE2_MATRIX_H
