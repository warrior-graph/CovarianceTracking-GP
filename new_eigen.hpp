#ifndef NEW_EIGEN_HPP
#define NEW_EIGEN_HPP
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using Eigen::MatrixXd;

namespace new_eigen
{
    template <typename T>
    vector<double> get_eigenvals(const T &t);
    double diss(const Mat &c1, const Mat &c2);
}


#endif // NEW_EIGEN_HPP
