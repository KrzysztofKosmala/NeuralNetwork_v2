//
// Created by Krzysztof Kosmala on 15.06.2018.
//

#include <Matrix.h>
#include <DataContainer.h>

DataContainer::DataContainer(std::shared_ptr<Matrix> hiddenInput, std::shared_ptr<Matrix> hiddenOutput) {
    this->hiddenInputMatrix=hiddenInput;
    this->hiddenOutputMatrix=hiddenOutput;
}

void DataContainer::setHiddenInputMatrix(std::shared_ptr<Matrix> matrix) {
    this->hiddenInputMatrix->refactor(matrix);
}

std::shared_ptr<Matrix> DataContainer::getHiddenInputMatrix() {
    return this->hiddenInputMatrix;
}

void DataContainer::setHiddenOutputMatrix(std::shared_ptr<Matrix> matrix) {
    this->hiddenOutputMatrix->refactor(matrix);
}

std::shared_ptr<Matrix> DataContainer::getHiddenOutputMatrix() {
    return this->hiddenOutputMatrix;
};