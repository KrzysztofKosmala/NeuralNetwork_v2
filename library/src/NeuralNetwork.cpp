#include "NeuralNetwork.h"
#include <fstream>
#include <math.h>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <DataContainer.h>

NeuralNetwork::NeuralNetwork (std::string i_p, std::string t_p) : targets_path(t_p),input_path(i_p)
{
    setup();
    setTopology();
    setBias();

    setTargets();

}


void NeuralNetwork::saveWeights()
{
    this->historyWeights.push_back(*(new DataContainer(this->weights_ih, this->weights_ho)));
}

void NeuralNetwork::setInputs ()
{
    std::ifstream file(input_path, std::ios::in);
    std::string   line;
    while(file.good()) {
        while (std::getline(file, line, '\n')) {
            std::string data;
            std::vector<std::string> inputData;
            std::vector<double> inputDoubleData;


            boost::split(inputData, line, boost::is_any_of(" "));

            for (unsigned int i = 0; i < inputData.size(); i++) {
                auto inputValue = stod(inputData[i]);

                inputDoubleData.push_back(inputValue);
            }

            auto newMatrix = std::make_shared<Matrix>(input_nodes, 1, false);

            for (unsigned int j = 0; j < input_nodes; j++) {
                newMatrix->setValue(j, 0, inputDoubleData[j]);
            }

            this->inputs.push_back(newMatrix);
        }
    }

}

void NeuralNetwork::printToConsole()
{
    for (unsigned int l = 0; l < inputs.size(); l++) {
        input=inputs[l];
        target=targets[l];
        output=outputs[l];

        saveWeights();

        std::cout << "-------------------------WARSTWY-------------------------" << std::endl;
        std::cout << "Ilosc epok: " << epok<<  std::endl;
        std::cout << "--------" << std::endl;
        std::cout << "inputs: " << std::endl;
        input->printToConsole();
        std::cout << "--------" << std::endl;
        std::cout << "target: " << std::endl;
        target->printToConsole();
        std::cout << "--------" << std::endl;
        std::cout << "outputs: " << std::endl;
        output->printToConsole();

    }

}

void NeuralNetwork::feedForward()
{

    hidden[0]->refactor(Matrix::multiply(weights_ih,input));
    hidden[0]->add(bias_h[0]);
    hidden[0]->sigmoid();
    if(hidden.size()>1)
    {
        for (int i = 1; i < hidden.size(); i++)
        {
            hidden[i]->refactor(Matrix::multiply(weights_hh[i-1], hidden[i-1]));
            hidden[i]->add(bias_h[i]);
            hidden[i]->sigmoid();
        }
    }
    output->refactor(Matrix::multiply(weights_ho,hidden[hidden.size()-1]));
    output->add(bias_o);
    output->sigmoid();


}

void NeuralNetwork::setHidden ()
{
    if (hidden_nodes.size() != hidden_layers)
    {std::cerr<<"nie podano wystarczających informacji na temat warstw ukrytych";}
    else
    for (int i =0; i<hidden_nodes.size(); i++)
    {
        hidden.push_back(std::make_shared<Matrix>(hidden_nodes[i],1,false));
    }
}

void NeuralNetwork::setHidden_W ()
{
    if (hidden_nodes.size()<=1)
    {
    }
    else
    {
        for(int i =0; i<hidden_layers-1; i++)
        {
                weights_hh.push_back(std::make_shared<Matrix>(hidden_nodes[i + 1], hidden_nodes[i], true));
                old_weights_hh.push_back(std::make_shared<Matrix>(hidden_nodes[i + 1], hidden_nodes[i], true));
        }
    }
}

void NeuralNetwork::setTopology ()
{
    setHidden();

    setInputs();

    for (unsigned int i =0; i<inputs.size(); i++) {

        output = std::make_shared<Matrix>(output_nodes, 1, false);
        this->outputs.push_back(output);
    }
    weights_ih = std::make_shared<Matrix>(hidden_nodes[0], input_nodes, true);

    weights_ho = std::make_shared<Matrix>(output_nodes, hidden_nodes.at(hidden_nodes.size()-1), true);
    old_weights_ih = std::make_shared<Matrix>(hidden_nodes[0], input_nodes, true);
    old_weights_ho = std::make_shared<Matrix>(output_nodes, hidden_nodes.at(hidden_nodes.size()-1), true);

    setHidden_W();
}

void NeuralNetwork::setBias ()
{
    bias_o = std::make_shared<Matrix>(output_nodes,1,false);
    for (int i =0; i<hidden.size(); i++)
    {
        bias_h.push_back(std::make_shared<Matrix>(hidden_nodes[i],1,false));
    }
    if(bias)
    {
        for (int i=0; i<output_nodes; i++)
        {
            bias_o->setValue(i,0,1);
        }
        for (int h = 0; h<hidden.size(); h++)
        {
            for(int j =0; j<bias_h[h]->getNumRows(); j++)
                bias_h[h]->setValue(j,0,1);
        }
    }
}


void NeuralNetwork::train ()
{


    gradients_o= std::make_shared<Matrix>(output->getNumRows(),output->getNumCols(),false);
    gradients_o->refactor(output);


    help_o = Matrix::substractElement(output,1);
    help_o->multiplyElementWise(output);
    gradients_o->refactor(help_o);


    // gradients_o->disigmoid();

    gradients_o->multiplyElementWise(error_o);
    gradients_o->multiply(learning_rate);
    h_t = Matrix::transpose(hidden[hidden.size()-1]);
    //  old_weights_ho->refactor(weights_ho);
    weight_ho_deltas = Matrix::multiply(gradients_o,h_t);

    if (momentum)
    {
        std::shared_ptr<Matrix> help1 = Matrix::substract(weights_ho,old_weights_ho);
        help1->multiply(momentum_rate);
        weight_ho_deltas->add(help1);
        old_weights_ho->refactor(weights_ho);
    }
    weights_ho->add(weight_ho_deltas);

    if (bias)
    {
        bias_o->add(gradients_o);
    }
    if (hidden_layers>1)
    {
        for(int i =0; i<hidden.size()-1; i++)
        {
            hh_gradients.push_back(std::make_shared<Matrix>(hidden[i+1]->getNumRows(),hidden[i+1]->getNumCols(),false));
            old_hh_gradients.push_back(std::make_shared<Matrix>(hidden[i+1]->getNumRows(),hidden[i+1]->getNumCols(),false));
            hh_gradients[i]->refactor(hidden[i+1]);
            help_hh[i] = Matrix::substractElement(hidden[i+1],1);
            help_hh[i]->multiplyElementWise(hidden[i+1]);
            hh_gradients[i]->refactor(help_hh[i]);
            hh_gradients[i]->multiplyElementWise(help_error);
            hh_gradients[i]->multiply(learning_rate);
            hh_t = Matrix::transpose(hidden[i]);

            if (momentum)
            {

                std::shared_ptr<Matrix> help2 = Matrix::substract(weights_hh[i+1],old_weights_hh[i+1]);
                help2->multiply(momentum_rate);
                weight_hh_deltas->add(help2);
                old_weights_hh[i+1]->refactor(weights_hh[i+1]);
            }
            weights_hh[i]->add(weight_hh_deltas);

            if (bias)
            {
                bias_h[i+1]->add(hh_gradients[i]);
            }

            saveToFileErrorO();

        }
    }
    {
        hidden_gradients.push_back(std::make_shared<Matrix>(hidden[0]->getNumRows(),hidden[0]->getNumCols(),false));
        hidden_gradients[0]->refactor(hidden[0]);

        help_h = Matrix::substractElement(hidden[0],1);
        help_h->multiplyElementWise(hidden[0]);
        hidden_gradients[0]->refactor(help_h);

        hidden_gradients[0]->multiplyElementWise(help_error);
        hidden_gradients[0]->multiply(learning_rate);
        input_t = Matrix::transpose(input);
//        old_weight_ih_deltas=std::make_shared<Matrix>(weight_ih_deltas->getNumRows(),weight_ih_deltas->getNumCols(),false);
//        old_weights_ih->refactor(weights_ih);
        weight_ih_deltas = Matrix::multiply(hidden_gradients[0],input_t);

        if (momentum)
        {
            std::shared_ptr<Matrix> help3 = Matrix::substract(weights_ih,old_weights_ih);
            help3->multiply(momentum_rate);
            weight_ih_deltas->add(help3);
            old_weights_ih->refactor(weights_ih);
        }
        weights_ih->add(weight_ih_deltas);
        if (bias)
        {
            bias_h[0]->add(hidden_gradients[0]);
        }

    }



}
void NeuralNetwork::setTargets ()
{
    std::ifstream file(targets_path, std::ios::in);
    std::string   line;

    while(std::getline(file, line, '\n'))
    {
        std::string                 data;
        std::vector<std::string>    targetData;
        std::vector<double>         targetDoubleData;

        boost::split(targetData, line, boost::is_any_of(" "));

        for(unsigned int i = 0; i < targetData.size(); i++) {
            auto targetValue = stod(targetData[i]);

            targetDoubleData.push_back(targetValue);
        }

        auto newMatrix = std::make_shared<Matrix>(output_nodes, 1, false);

        for(unsigned int j = 0; j < output_nodes; j++) {
            newMatrix->setValue(j, 0, targetDoubleData[j]);
        }

        this->targets.push_back(newMatrix);
    }
}

void NeuralNetwork::setErrors ()
{

    error_o = Matrix::substract(target,output);
    who_t = Matrix::transpose(weights_ho);
    help_error = Matrix::multiply(who_t,error_o);
    error_h.push_back(Matrix::multiply(who_t,error_o));
    if (hidden.size()>1)
    {

        for (int i = 0; i < weights_hh.size(); i++)
        {
            whh_t.push_back(Matrix::transpose(weights_hh[i]));
        }

        for (int i = 0; i < whh_t.size(); i++)
        {
            error_h.push_back(Matrix::multiply(whh_t[i], error_h[i]));
        }
    }

}

void NeuralNetwork::backpropagation()
{

    std::ofstream errorFile;

    // otworzenie i wyczyszczenie pliku z poprzednich wartosci
    errorFile.open("..//..//error_result_file.csv", std::ofstream::out | std::ofstream::trunc);

    double blad=1.0;
    int i=0;
    epok=0;
    double erravg=1.0;

    while (err<erravg && i<loop)
    {
        double b = 0;

        errorFile << std::endl;


        randomMatrix();
//        std::cout<<"    --------  " << randomInputs.size() << std::endl;
        for (unsigned int j = 0; j < randomInputs.size(); j++) {



            input = randomInputs[j];
            target = randomTargets[j];
            output = randomOutputs[j];

            feedForward();
            setErrors();
            train();
            setErrors();
            feedForward();

            double error_sum = 0;
            for (int k = 0; k < error_o->getNumRows(); k++) {
                error_sum = error_sum + (pow(error_o->getValue(k, 0), 2.00) / 2);
            }
            blad = error_sum / (error_o->getNumRows());

            b=b+blad;

        }

        i++;
        epok++;

        b = b / inputs.size();
        erravg=b;

//         printf(" %f \n", b);         // print do konsoli
        std::cout<<epok<<std::endl;
        errorFile << erravg << (",");          // print do pliku

    }

    errorFile.close();

}



void NeuralNetwork::saveToFileErrorO()
{
    double error_sum=0;
    for(int i =0; i< error_o->getNumRows();i++)
    {
        error_sum=error_sum+(pow(error_o->getValue(i,0), 2.00)/2);

    }

    if (errortxt.is_open())
    {
        errortxt << error_sum/error_o->getNumRows();
       errortxt << "\n";

    }


}


void NeuralNetwork::run() {
    for (unsigned int j = 0; j < inputs.size(); j++) {
        input = inputs[j];
        target = targets[j];
        output = outputs[j];
        feedForward();
        setErrors();
        train();

    }check();
}


void NeuralNetwork::nameIris(){
    for (unsigned int i = 0; i < targets.size(); i++) {
//        auto newMatrix = std::make_shared<Matrix>(targets[i].get()., 1, false);
//        newMatrix->setValue(0,0,1);
//        newMatrix->setValue(1,0,0);
//        newMatrix->setValue(2,0,0);
//        if(targets[i].get()->values==newMatrix->values){
//
//        }
        if (targets[i]->getValue(0,0)==1)
        {
            std::cout<<"Kwiat setosa "<<i+1<<std::endl;
        }
        else if (targets[i]->getValue(1,0)==1)
        {
            std::cout<<"Kwiat Versicolour "<<i+1<<std::endl;
        }
        else if (targets[i]->getValue(2,0)==1)
        {
            std::cout<<"Kwiat Virginica "<<i+1<<std::endl;
        }
    }
}

void NeuralNetwork::groupIris(){
    int s=1,ve=1,vi=1;
    for (unsigned int i = 0; i < targets.size(); i++) {

        if (outputs[i]->getValue(0,0)>0.7)
        {
            std::cout<<"Kwiat setosa "<<s++<<std::endl;
        }
        else if (outputs[i]->getValue(1,0)>0.7)
        {
            std::cout<<"Kwiat Versicolour "<<ve++<<std::endl;
        }
        else if (outputs[i]->getValue(2,0)>0.7)
        {
            std::cout<<"Kwiat Virginica "<<vi++<<std::endl;
        }
//        else
//        {
//            std::cout<<"Nie rozpoznano kwiatu  "<<std::endl;
//        }
    }
}

void NeuralNetwork::check()
{
    learning=std::make_shared<Matrix>(3, 3, false);

    for (unsigned int i = 0; i < targets.size(); i++)
    {

        for ( int j = 0; j < 3; j++)
        {
            if (targets[i]->getValue(j,0)==1)
            {
                for ( int k = 0; k < 3; k++)
                {
                    if (outputs[i]->getValue(k,0)>0.6)
                    {
                        learning->setValue(j,k,learning->getValue(j,k)+1);
                    }
                }
            }
        }
    }
    learning->printToConsole2();
}

void NeuralNetwork::randomMatrix()
{
    randomInputs.clear();
    randomOutputs.clear();
    randomTargets.clear();
    randomNumbers.clear();

        for(int i = 0; i < inputs.size(); i++) {
            int randomNumber = getRandomNumber(inputs.size());

//            std::cout << i+1 << "  " <<randomInputs.size()<< std::endl;

            randomInputs.push_back(inputs[randomNumber]);
            randomTargets.push_back(targets[randomNumber]);
            randomOutputs.push_back(outputs[randomNumber]);

        }

}

bool NeuralNetwork::isInRandomNumbersList(int randomNumberFromPetla) {
        std::vector<int>::iterator it;

        it = std::find(randomNumbers.begin(), randomNumbers.end(), randomNumberFromPetla);
        if (it != randomNumbers.end()) {
            return true;
        }
        return false;
    }

int NeuralNetwork::getRandomNumber(int inputsSize) {

    int randomNumber =( std::rand() % inputsSize ) + 0;

        if(isInRandomNumbersList(randomNumber)) {
            return getRandomNumber(inputsSize);
        }
        randomNumbers.push_back(randomNumber);
        return randomNumber;
    }


