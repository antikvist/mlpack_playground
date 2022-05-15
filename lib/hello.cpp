#include "hello.h"

#include <iostream>

#include <mlpack/methods/linear_regression/linear_regression.hpp>

void Hello::greet() const
{
    arma::mat regressors({1.0, 2.0, 3.0});
    arma::rowvec responses({1.0, 4.0, 9.0});
    mlpack::regression::LinearRegression lr (regressors, responses);
    // lr.Train(regressors, responses);
    arma::mat testX({2.0});
    arma::rowvec testY;
    lr.Predict(testX, testY);
    std::cout << testY << std::endl;

    std::cout << "greetings!" << "\n";
}