#ifndef ZADANIE2_NEURALNETWORK_H
#define ZADANIE2_NEURALNETWORK_H

#include <vector>
#include <fstream>
#include "Matrix.h"
#include <DataContainer.h>
#include <random>
#include <algorithm>
#include <string>
#include <iostream>


class NeuralNetwork
{
    private:
         std::ofstream errortxt ;
        int input_nodes, hidden_layers,  output_nodes, loop, epok;
        bool bias, momentum;
        double learning_rate, momentum_rate, err, learnig_sensivity, sensivity;
        std::string targets_path, input_path, path;


        std::vector<DataContainer> historyWeights;
        std::vector<int> hidden_nodes;
        std::vector<std::shared_ptr<Matrix>> hidden;
        std::vector<std::shared_ptr<Matrix>> error_h;
        std::vector<std::shared_ptr<Matrix>> weights_hh;
        std::vector<std::shared_ptr<Matrix>> old_weights_hh;
        std::vector<std::shared_ptr<Matrix>> whh_t;
        std::vector<std::shared_ptr<Matrix>> bias_h;
        std::vector<std::shared_ptr<Matrix>> bias_hh;
        std::vector<std::shared_ptr<Matrix>> hidden_gradients;
        std::vector<std::shared_ptr<Matrix>> hh_gradients;
        std::vector<std::shared_ptr<Matrix>> old_hh_gradients;
        std::vector<std::shared_ptr<Matrix>> history;
        std::vector<std::shared_ptr<Matrix>> targets;
        std::vector<std::shared_ptr<Matrix>> outputs;
        std::vector<std::shared_ptr<Matrix>> help_hh;
        std::vector<std::shared_ptr<Matrix>> errors_o;

        std::vector<std::shared_ptr<Matrix>> randomInputs;
        std::vector<std::shared_ptr<Matrix>> randomTargets;
        std::vector<std::shared_ptr<Matrix>> randomOutputs;




    std::shared_ptr<Matrix> input;
        std::shared_ptr<Matrix> weights_ih;

        std::shared_ptr<Matrix> output;
        std::shared_ptr<Matrix> learning;
        std::shared_ptr<Matrix> weights_ho;
        std::shared_ptr<Matrix> bias_o;
        std::shared_ptr<Matrix> error_o;
        std::shared_ptr<Matrix> error_o_delta;
        std::shared_ptr<Matrix> error_o_old;

        std::shared_ptr<Matrix> gradients_o;
        std::shared_ptr<Matrix> h_t;
        std::shared_ptr<Matrix> hh_t;

        std::shared_ptr<Matrix> weight_ho_deltas;
        std::shared_ptr<Matrix> weight_hh_deltas;
        std::shared_ptr<Matrix> old_weight_ho_deltas;
        std::shared_ptr<Matrix> old_weights_ho;
        std::shared_ptr<Matrix> old_weight_ih_deltas;
        std::shared_ptr<Matrix> old_weights_ih;


        std::shared_ptr<Matrix> weight_ih_deltas;
        std::shared_ptr<Matrix> target;
        std::shared_ptr<Matrix> who_t;
        std::shared_ptr<Matrix> input_t;



        std::shared_ptr<Matrix> help_h;
        std::shared_ptr<Matrix> help_o;
        std::shared_ptr<Matrix> help_error;

        std::vector<int> randomNumbers;


        void setInputs ();
        void setTopology();
        void setHidden ();
        void setHidden_W();
        void setBias();

        void setTargets();

        //metoda w ktorej ustawiana jest topologia
        void setup ()
        {

            //ilosc neuronow w warstwie wejscowej
            input_nodes=4;
            //ilosc warstw ukrytych
            hidden_layers=1;
            //dodawanie informacji na temat warsw ukrytych (jeden push = jedna informacja na temat ilosci neuronow w warswie, powinno byc tyle ile hidden_layers)
            hidden_nodes.push_back(25);
            //ilosc neuronow wyjsciowych
            output_nodes=3;
            //obliczenia z dodaniem biasu - true / obliczenia bez biasu - false
            bias=true;
            //
            learning_rate=0.8;
            loop=500;
            momentum= true;
            momentum_rate=0.5;
            err=0.001;
        }

    public:
        std::vector<std::shared_ptr<Matrix>> inputs;
        NeuralNetwork (std::string t_p, std::string i_p);
        void backpropagation();
        void printToConsole();
        void feedForward();
        void train();
        void setErrors();
        void saveToFileErrorO();
        void saveWeights();
        void run();
        void nameIris();
        void groupIris();
        void check();
        void randomMatrix();
        bool isInRandomNumbersList(int randomNumberFromPetla);
        int getRandomNumber(int inputsSize);


    };


#endif //ZADANIE2_NEURALNETWORK_H
