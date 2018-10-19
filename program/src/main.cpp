#include <iostream>
#include "Neuron.h"
#include "Matrix.h"
#include <random>
#include <ctime>

#include "NeuralNetwork.h"


using namespace std;


int main()
{
    srand( time( NULL ) );
   
    NeuralNetwork n1("..//..//input2b.txt","..//..//targets2b.txt");

    n1.backpropagation();
//
    n1.run();
    n1.printToConsole();
//    n1.train();
//    n1.printToConsole();
//    n1.run();
//    n1.printToConsole();
    n1.check();
//    n1.groupIris();

    return 0;
}