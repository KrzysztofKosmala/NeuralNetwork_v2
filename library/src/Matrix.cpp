
#include "Matrix.h"
#include <random>

Matrix::Matrix(int numRows, int numCols, bool isRandom): numRows(numRows),numCols(numCols)
{
    for (int i=0; i<numRows; i++)
    {
        std::vector<Neuron> colValues;
        for (int j=0; j<numCols; j++)
        {
            Neuron r = 0.00;
            if (isRandom)
            {
                r.setVal(this->generateRandomNumber(0,1));

            }
            colValues.push_back(r);
        }
        this->values.push_back(colValues);
    }

}

double Matrix::getValue(int r, int c) {
    return this->values.at(r).at(c).getVal();
}

std::shared_ptr<Matrix> Matrix::transpose(std::shared_ptr<Matrix> m)
{
    auto result = std::make_shared<Matrix>(m->getNumCols(),m->getNumRows(),false);

    for (int i=0; i<m->getNumRows(); i++)
    {
        for (int j=0; j<m->getNumCols(); j++)
        {
            result->setValue(j, i, m->getValue(i,j));
        }
    }
    return result;
}

void Matrix::setValue(int r, int c, double v)
{
    this->values.at(r).at(c).setVal(v);
}

int Matrix::getNumRows(){    return numRows;    }

int Matrix::getNumCols() {    return numCols;   }

void Matrix::printToConsole()
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            std::cout<<this->values.at(i).at(j).getVal()<<"\t\t";
        }
        std::cout<<std::endl;
    }
}

void Matrix::printToConsole2()
{
//    for (int i=0; i<numRows; i++)
//    {
//        for (int j=0; j<numCols; j++)
//        {
//            std::cout<<this->values.at(i).at(j).getVal()<<"\t\t";
//        }
//        std::cout<<std::endl;
//    }
    std::cout<<"-----"<<"\t\t";
    std::cout<<"Setosa "<<"\t\t";
    std::cout<<"Versicolour "<<"\t\t";
    std::cout<<"Virginica "<<"\t\t"<<std::endl;
    std::cout<<"Setosa "<<"\t\t\t";
    std::cout<<this->values.at(0).at(0).getVal()<<"\t\t";
    std::cout<<this->values.at(0).at(1).getVal()<<"\t\t";
    std::cout<<this->values.at(0).at(2).getVal()<<"\t\t"<<std::endl;
    std::cout<<"Versicolour "<<"\t\t";
    std::cout<<this->values.at(1).at(0).getVal()<<"\t\t";
    std::cout<<this->values.at(1).at(1).getVal()<<"\t\t";
    std::cout<<this->values.at(1).at(2).getVal()<<"\t\t"<<std::endl;
    std::cout<<"Virginica "<<"\t\t";
    std::cout<<this->values.at(2).at(0).getVal()<<"\t\t";
    std::cout<<this->values.at(2).at(1).getVal()<<"\t\t";
    std::cout<<this->values.at(2).at(2).getVal()<<"\t\t"<<std::endl;
}

double Matrix::generateRandomNumber(double fMin, double fMax)
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double Matrix::generateRandomNumber()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<> dis(0,1);
    return dis(generator);
}

void Matrix::add (double n)
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            this->values[i][j].setVal(this->values[i][j].getVal()+ n);
        }
    }
}

void Matrix::add (std::shared_ptr<Matrix> p)
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            this->values[i][j].setVal(this->values[i][j].getVal()+ p->getValue(i,j));
        }
    }
}

std::shared_ptr<Matrix> Matrix::multiply (std::shared_ptr<Matrix> n, std::shared_ptr<Matrix> p)
{
    if (n->getNumCols()!= p->getNumRows())
    {
        std::cerr<<"columns od n must match rows of columns p";
        return 0;
    }

    auto result = std::make_shared<Matrix>(n->getNumRows(),p->getNumCols(),false);

    for (int i=0; i < n->getNumRows(); i++)
    {
        for (int j=0; j < result->getNumCols(); j++)
        {
            double sum = 0;
            for(int k =0; k < n->getNumCols(); k++)
            {
                sum += n->getValue(i,k) * p->getValue(k,j);
            }
            result->setValue(i,j,sum);
        }

    }
    return result;
}

void Matrix::multiply (double n)
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            this->values[i][j].setVal(this->values[i][j].getVal()* n);
        }
    }
}

void Matrix::refactor (std::shared_ptr<Matrix> p)
{
    for (int i = 0; i<p->getNumRows(); i++)
    {
        for (int j=0; j<p->getNumCols(); j++)
        {
            this->values[i][j].setVal(p->getValue(i,j));
        }

    }
}

void Matrix::sigmoid ()
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            this->values[i][j].setVal(1/(1+exp(-(this->values[i][j].getVal()))));
        }
    }

}

std::shared_ptr<Matrix> Matrix::substract (std::shared_ptr<Matrix> n, std::shared_ptr<Matrix> p)
{


    auto result = std::make_shared<Matrix>(n->getNumRows(),n->getNumCols(),false);

    for (int i=0; i < result->getNumRows(); i++)
    {
        for (int j=0; j < result->getNumCols(); j++)
        {

            result->setValue(i,j,n->getValue(i,j)-p->getValue(i,j));

        }

    }
    return result;

}

void Matrix::disigmoid ()
{
    for (int i=0; i<numRows; i++)
    {
        for (int j=0; j<numCols; j++)
        {
            this->values[i][j].setVal(this->values[i][j].getVal()*(1-(this->values[i][j].getVal())));
        }
    }
}

void Matrix::multiplyElementWise (std::shared_ptr<Matrix> p)
{

    for (int i=0; i < this->getNumRows(); i++)
    {double sum = 0;
        for (int j=0; j < this->getNumCols(); j++)
        {


            sum += this->getValue(i,j) * p->getValue(i,j);

            this->setValue(i,j,sum);
        }

    }

}

std::shared_ptr<Matrix> Matrix::substractElement (std::shared_ptr<Matrix> n, int p)
{
    auto result = std::make_shared<Matrix>(n->getNumRows(),n->getNumCols(),false);

    for (int i=0; i < result->getNumRows(); i++)
    {
        for (int j=0; j < result->getNumCols(); j++)
        {

            result->setValue(i,j,p-n->getValue(i,j));

        }

    }
    return result;
}



