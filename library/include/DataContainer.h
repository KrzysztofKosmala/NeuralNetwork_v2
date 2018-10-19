#ifndef ZADANIE2_DATACONTAINER_H
#define ZADANIE2_DATACONTAINER_H

#include <Matrix.h>

class DataContainer {

public:
    DataContainer() {}
    DataContainer(std::shared_ptr<Matrix> hiddenInput, std::shared_ptr<Matrix> hiddenOutput);
    void setHiddenInputMatrix(std::shared_ptr<Matrix> matrix);
    std::shared_ptr<Matrix> getHiddenInputMatrix();

    void setHiddenOutputMatrix(std::shared_ptr<Matrix> matrix);
    std::shared_ptr<Matrix> getHiddenOutputMatrix();

private:
    std::shared_ptr<Matrix> hiddenInputMatrix;
    std::shared_ptr<Matrix> hiddenOutputMatrix;
};

#endif //ZADANIE2_DATACONTAINER_H
