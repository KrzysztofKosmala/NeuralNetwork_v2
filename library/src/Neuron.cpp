//
// Created by Lenovo on 2018-03-21.
//

#include "../include/Neuron.h"



double Neuron::getVal()  {
    return val;
}

Neuron::Neuron(double val) : val(val)
{
}

void Neuron::setVal(double val)
{
    this->val=val;

}


