#ifndef _GAUSSIAN_PROCESS
#define _GAUSSIAN_PROCESS

#include <functional>
#include <vector>

class GP
{
private:
        // mean and covariance functions
        std::function<double(double)> m;
        std::function<double(double, double)> k;

        // Assumed noise hyperparameter
        double sigma;

        // number of inputs
        int n;

        //training set
        std::vector<double> xt, yt;

        //inverse gram matrix
        std::vector<std::vector<double>> L;

public:
        // Create a new GP with given mean and covariance function, and noise parameter
        GP(std::function<double(double)> m, std::function<double(double, double)> k, double sigma);

        // Condition the GP on a new example (x, y)
        void push(double x, double y);

        // Get the mean values at given points
        std::vector<double> get_means(std::vector<double> xs);

        // Get the covariances at all pairs of given points
        std::vector<std::vector<double>> get_covar(std::vector<double> xs);
};

#endif
