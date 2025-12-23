#include <iostream>
#include <math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/LU>

namespace py = pybind11;
using std::cout;
using std::endl;
using lui = long unsigned int;
template <lui... Args> using Matrix = Eigen::Matrix<double, Args...>;
template <typename T, size_t... Args> using Dense = Eigen::Matrix<T,Args...>;  
template <typename T, size_t... Args> using RowMaj = Eigen::Matrix<T,Args...,Eigen::RowMajor>;  

template <typename T> using Sparse = Eigen::SparseMatrix<T>;
template <typename T, size_t... Args> using Array = Eigen::Array<T,Args...>;
template <typename T, size_t rows> using Dynamic = Eigen::Matrix<T,Eigen::Dynamic,rows>;
template <typename T, size_t cols> using Cols = Eigen::Matrix<T,cols,Eigen::Dynamic>;

using Eigen::Map;
using Eigen::all;
template <typename T, size_t cols> using CDynamic = Eigen::Matrix<T,cols,Eigen::Dynamic>;
template <typename T, size_t cols> using DRowMaj = Eigen::Matrix<T,cols,Eigen::Dynamic,Eigen::RowMajor>;  
template <typename T, size_t rows> using RDynamic = Eigen::Matrix<T,Eigen::Dynamic,rows>;
template <typename T>              using Numpy = py::array_t<T>;


struct Indices {
  public: 
  Numpy<size_t> rows, cols;
  size_t *r, *c;

  Indices (size_t len) {
    rows = Numpy<size_t> (len);
    cols = Numpy<size_t> (len);
    c    = (size_t *) cols.request().ptr;
    r    = (size_t *) rows.request().ptr;
  }
};


template <typename T>
struct VCR {
  Numpy<T> values;
  Numpy<size_t> rows, cols;
  T *v; size_t *r, *c;

  VCR (size_t len) {
    values = Numpy<T>(len);
    rows   = Numpy<size_t> (len);
    cols   = Numpy<size_t> (len);
    v = (T *)     values.request().ptr;
    c = (size_t *)cols.request().ptr;
    r = (size_t *)rows.request().ptr;
  }
};


template <typename T>
Dense<T,192,1> gauss_alt () {
  const double s3 = 1 / sqrt(3);
  const double ps3 = 1 + s3;
  const double ns3 = 1 - s3;
  const double p2 = ps3 * ps3 / 8;
  const double pn = ns3 * ps3 / 8;
  const double n2 = ns3 * ns3 / 8;
  Dense<T,192,1> G;
  G << -p2 , -p2 , -p2 , p2 , -pn , -pn , pn ,  pn , -n2 ,-pn ,  p2 , -pn ,
    -pn , -pn ,  p2 , pn , -n2 ,  pn , n2 ,  n2 ,  n2 ,-n2 ,  pn ,  pn ,
    -p2 , -pn , -pn , p2 , -p2 , -p2 , pn ,  p2 , -pn ,-pn ,  pn , -n2 ,
    -pn , -n2 ,  pn , pn , -pn ,  p2 , n2 ,  pn ,  pn ,-n2 ,  n2 ,  n2 ,
    -pn , -pn , -n2 , pn , -p2 , -pn , p2 ,  p2 , -p2 ,-p2 ,  pn , -pn ,
    -n2 , -n2 ,  n2 , n2 , -pn ,  pn , pn ,  pn ,  p2 ,-pn ,  n2 ,  pn ,
    -pn , -p2 , -pn , pn , -pn , -n2 , p2 ,  pn , -pn ,-p2 ,  p2 , -p2 ,
    -n2 , -pn ,  pn , n2 , -n2 ,  n2 , pn ,  n2 ,  pn ,-pn ,  pn ,  p2 ,
    -pn , -pn , -p2 , pn , -n2 , -pn , n2 ,  n2 , -n2 ,-n2 ,  pn , -pn ,
    -p2 , -p2 ,  p2 , p2 , -pn ,  pn , pn ,  pn ,  n2 ,-pn ,  p2 ,  pn ,
    -pn , -n2 , -pn , pn , -pn , -p2 , n2 ,  pn , -pn ,-n2 ,  n2 , -n2 ,
    -p2 , -pn ,  pn , p2 , -p2 ,  p2 , pn ,  p2 ,  pn ,-pn ,  pn ,  n2 ,
    -n2 , -n2 , -n2 , n2 , -pn , -pn , pn ,  pn , -p2 ,-pn ,  n2 , -pn ,
    -pn , -pn ,  n2 , pn , -p2 ,  pn , p2 ,  p2 ,  p2 ,-p2 ,  pn ,  pn ,
    -n2 , -pn , -pn , n2 , -n2 , -n2 , pn ,  n2 , -pn ,-pn ,  pn , -p2 ,
    -pn , -p2 ,  pn , pn , -pn ,  n2 , p2 ,  pn ,  pn ,-p2 ,  p2 ,  p2 ;
  return G;
}


template <typename T>
Dense<T,64,3> gauss () {
  const double s3 = 1 / sqrt(3);
  const double ps3 = 1 + s3;
  const double ns3 = 1 - s3;
  const double p2 = ps3 * ps3 / 8;
  const double pn = ns3 * ps3 / 8;
  const double n2 = ns3 * ns3 / 8;
  Dense<T,64,3> G;
  G << -p2 , -p2 , -p2 , p2 , -pn , -pn , pn ,  pn , -n2 ,-pn ,  p2 , -pn ,
    -pn , -pn ,  p2 , pn , -n2 ,  pn , n2 ,  n2 ,  n2 ,-n2 ,  pn ,  pn ,
    -p2 , -pn , -pn , p2 , -p2 , -p2 , pn ,  p2 , -pn ,-pn ,  pn , -n2 ,
    -pn , -n2 ,  pn , pn , -pn ,  p2 , n2 ,  pn ,  pn ,-n2 ,  n2 ,  n2 ,
    -pn , -pn , -n2 , pn , -p2 , -pn , p2 ,  p2 , -p2 ,-p2 ,  pn , -pn ,
    -n2 , -n2 ,  n2 , n2 , -pn ,  pn , pn ,  pn ,  p2 ,-pn ,  n2 ,  pn ,
    -pn , -p2 , -pn , pn , -pn , -n2 , p2 ,  pn , -pn ,-p2 ,  p2 , -p2 ,
    -n2 , -pn ,  pn , n2 , -n2 ,  n2 , pn ,  n2 ,  pn ,-pn ,  pn ,  p2 ,
    -pn , -pn , -p2 , pn , -n2 , -pn , n2 ,  n2 , -n2 ,-n2 ,  pn , -pn ,
    -p2 , -p2 ,  p2 , p2 , -pn ,  pn , pn ,  pn ,  n2 ,-pn ,  p2 ,  pn ,
    -pn , -n2 , -pn , pn , -pn , -p2 , n2 ,  pn , -pn ,-n2 ,  n2 , -n2 ,
    -p2 , -pn ,  pn , p2 , -p2 ,  p2 , pn ,  p2 ,  pn ,-pn ,  pn ,  n2 ,
    -n2 , -n2 , -n2 , n2 , -pn , -pn , pn ,  pn , -p2 ,-pn ,  n2 , -pn ,
    -pn , -pn ,  n2 , pn , -p2 ,  pn , p2 ,  p2 ,  p2 ,-p2 ,  pn ,  pn ,
    -n2 , -pn , -pn , n2 , -n2 , -n2 , pn ,  n2 , -pn ,-pn ,  pn , -p2 ,
    -pn , -p2 ,  pn , pn , -pn ,  n2 , p2 ,  pn ,  pn ,-p2 ,  p2 ,  p2 ;
  return G;
}


template <typename T> 
Dense<T,24,24> K_point ( const Dense<T,8,3>& X ,
                          const Dense<T,8,3>& u ,
                          const Dense<T,8,3>& g , T c, T d ) {
  T dJ,dF,ldF;
  Dense<T,3,3> J = g.transpose() * X;
  Dense<T,3,3> Ji = J.inverse();
  Dense<T,8,3> Xi = g * Ji.transpose();
  Dense<T,3,3> F = Dense<T,3,3>::Identity() + u.transpose() * Xi;
  Dense<T,3,3> Fi = F.inverse();
  dJ = J.determinant(); dF = F.determinant(); ldF = log(dF);

  T PdF1111=2*c + 2*pow(Fi(0,0),2)*d + pow(Fi(0,0),2)*(2*c - 2*d*ldF);
  T PdF1112=2*Fi(0,0)*Fi(1,0)*(c + d - d*ldF);
  T PdF1113=2*Fi(0,0)*Fi(2,0)*(c + d - d*ldF);
  T PdF1121=2*Fi(0,0)*Fi(0,1)*(c + d - d*ldF);
  T PdF1122=2*Fi(0,0)*Fi(1,1)*d + Fi(0,1)*Fi(1,0)*(2*c - 2*d*ldF);
  T PdF1123=2*Fi(0,0)*Fi(2,1)*d + Fi(0,1)*Fi(2,0)*(2*c - 2*d*ldF);
  T PdF1131=2*Fi(0,0)*Fi(0,2)*(c + d - d*ldF);
  T PdF1132=2*Fi(0,0)*Fi(1,2)*d + Fi(0,2)*Fi(1,0)*(2*c - 2*d*ldF);
  T PdF1133=2*Fi(0,0)*Fi(2,2)*d + Fi(0,2)*Fi(2,0)*(2*c - 2*d*ldF);
  T PdF1212=2*c + 2*pow(Fi(1,0),2)*d + pow(Fi(1,0),2)*(2*c - 2*d*ldF);
  T PdF1213=2*Fi(1,0)*Fi(2,0)*(c + d - d*ldF);
  T PdF1221=2*Fi(0,1)*Fi(1,0)*d + Fi(0,0)*Fi(1,1)*(2*c - 2*d*ldF);
  T PdF1222=2*Fi(1,0)*Fi(1,1)*(c + d - d*ldF);
  T PdF1223=2*Fi(1,0)*Fi(2,1)*d + Fi(1,1)*Fi(2,0)*(2*c - 2*d*ldF);
  T PdF1231=2*Fi(0,2)*Fi(1,0)*d + Fi(0,0)*Fi(1,2)*(2*c - 2*d*ldF);
  T PdF1232=2*Fi(1,0)*Fi(1,2)*(c + d - d*ldF);
  T PdF1233=2*Fi(1,0)*Fi(2,2)*d + Fi(1,2)*Fi(2,0)*(2*c - 2*d*ldF);
  T PdF1313=2*c + 2*pow(Fi(2,0),2)*d + pow(Fi(2,0),2)*(2*c - 2*d*ldF);
  T PdF1321=2*Fi(0,1)*Fi(2,0)*d + Fi(0,0)*Fi(2,1)*(2*c - 2*d*ldF);
  T PdF1322=2*Fi(1,1)*Fi(2,0)*d + Fi(1,0)*Fi(2,1)*(2*c - 2*d*ldF);
  T PdF1323=2*Fi(2,0)*Fi(2,1)*(c + d - d*ldF);
  T PdF1331=2*Fi(0,2)*Fi(2,0)*d + Fi(0,0)*Fi(2,2)*(2*c - 2*d*ldF);
  T PdF1332=2*Fi(1,2)*Fi(2,0)*d + Fi(1,0)*Fi(2,2)*(2*c - 2*d*ldF);
  T PdF1333=2*Fi(2,0)*Fi(2,2)*(c + d - d*ldF);
  T PdF2121=2*c + 2*pow(Fi(0,1),2)*d + pow(Fi(0,1),2)*(2*c - 2*d*ldF);
  T PdF2122=2*Fi(0,1)*Fi(1,1)*(c + d - d*ldF);
  T PdF2123=2*Fi(0,1)*Fi(2,1)*(c + d - d*ldF);
  T PdF2131=2*Fi(0,1)*Fi(0,2)*(c + d - d*ldF);
  T PdF2132=2*Fi(0,1)*Fi(1,2)*d + Fi(0,2)*Fi(1,1)*(2*c - 2*d*ldF);
  T PdF2133=2*Fi(0,1)*Fi(2,2)*d + Fi(0,2)*Fi(2,1)*(2*c - 2*d*ldF);
  T PdF2222=2*c + 2*pow(Fi(1,1),2)*d + pow(Fi(1,1),2)*(2*c - 2*d*ldF);
  T PdF2223=2*Fi(1,1)*Fi(2,1)*(c + d - d*ldF);
  T PdF2231=2*Fi(0,2)*Fi(1,1)*d + Fi(0,1)*Fi(1,2)*(2*c - 2*d*ldF);
  T PdF2232=2*Fi(1,1)*Fi(1,2)*(c + d - d*ldF);
  T PdF2233=2*Fi(1,1)*Fi(2,2)*d + Fi(1,2)*Fi(2,1)*(2*c - 2*d*ldF);
  T PdF2323=2*c + 2*pow(Fi(2,1),2)*d + pow(Fi(2,1),2)*(2*c - 2*d*ldF);
  T PdF2331=2*Fi(0,2)*Fi(2,1)*d + Fi(0,1)*Fi(2,2)*(2*c - 2*d*ldF);
  T PdF2332=2*Fi(1,2)*Fi(2,1)*d + Fi(1,1)*Fi(2,2)*(2*c - 2*d*ldF);
  T PdF2333=2*Fi(2,1)*Fi(2,2)*(c + d - d*ldF);
  T PdF3131=2*c + 2*pow(Fi(0,2),2)*d + pow(Fi(0,2),2)*(2*c - 2*d*ldF);
  T PdF3132=2*Fi(0,2)*Fi(1,2)*(c + d - d*ldF);
  T PdF3133=2*Fi(0,2)*Fi(2,2)*(c + d - d*ldF);
  T PdF3232=2*c + 2*pow(Fi(1,2),2)*d + pow(Fi(1,2),2)*(2*c - 2*d*ldF);
  T PdF3233=2*Fi(1,2)*Fi(2,2)*(c + d - d*ldF);
  T PdF3333=2*c + 2*pow(Fi(2,2),2)*d + pow(Fi(2,2),2)*(2*c - 2*d*ldF);

  Dense<T,24,24> K = Dense<T,24,24>::Zero();
  K(0,0)=PdF1111*pow(Xi(0,0),2) + 2*PdF1112*Xi(0,0)*Xi(0,1) + 2*PdF1113*Xi(0,0)*Xi(0,2) + PdF1212*pow(Xi(0,1),2) + 2*PdF1213*Xi(0,1)*Xi(0,2) + PdF1313*pow(Xi(0,2),2);
  K(0,1)=Xi(0,0)*(PdF1121*Xi(0,0) + PdF1122*Xi(0,1) + PdF1123*Xi(0,2)) + PdF1222*pow(Xi(0,1),2) + PdF1323*pow(Xi(0,2),2) + PdF1221*Xi(0,0)*Xi(0,1) + PdF1223*Xi(0,1)*Xi(0,2) + PdF1321*Xi(0,0)*Xi(0,2) + PdF1322*Xi(0,1)*Xi(0,2);
  K(0,2)=Xi(0,0)*(PdF1131*Xi(0,0) + PdF1132*Xi(0,1) + PdF1133*Xi(0,2)) + PdF1232*pow(Xi(0,1),2) + PdF1333*pow(Xi(0,2),2) + PdF1231*Xi(0,0)*Xi(0,1) + PdF1233*Xi(0,1)*Xi(0,2) + PdF1331*Xi(0,0)*Xi(0,2) + PdF1332*Xi(0,1)*Xi(0,2);
  K(0,3)=Xi(0,0)*(PdF1111*Xi(1,0) + PdF1112*Xi(1,1)) + PdF1112*Xi(0,1)*Xi(1,0) + PdF1113*Xi(0,0)*Xi(1,2) + PdF1113*Xi(1,0)*Xi(0,2) + PdF1212*Xi(0,1)*Xi(1,1) + PdF1213*Xi(0,1)*Xi(1,2) + PdF1213*Xi(0,2)*Xi(1,1) + PdF1313*Xi(0,2)*Xi(1,2);
  K(0,4)=Xi(0,0)*(PdF1121*Xi(1,0) + PdF1122*Xi(1,1)) + PdF1123*Xi(0,0)*Xi(1,2) + PdF1221*Xi(0,1)*Xi(1,0) + PdF1222*Xi(0,1)*Xi(1,1) + PdF1223*Xi(0,1)*Xi(1,2) + PdF1321*Xi(1,0)*Xi(0,2) + PdF1322*Xi(0,2)*Xi(1,1) + PdF1323*Xi(0,2)*Xi(1,2);
  K(0,5)=Xi(0,0)*(PdF1131*Xi(1,0) + PdF1132*Xi(1,1)) + PdF1133*Xi(0,0)*Xi(1,2) + PdF1231*Xi(0,1)*Xi(1,0) + PdF1232*Xi(0,1)*Xi(1,1) + PdF1233*Xi(0,1)*Xi(1,2) + PdF1331*Xi(1,0)*Xi(0,2) + PdF1332*Xi(0,2)*Xi(1,1) + PdF1333*Xi(0,2)*Xi(1,2);
  K(0,6)=Xi(0,0)*(PdF1111*Xi(2,0) + PdF1112*Xi(2,1)) + PdF1112*Xi(0,1)*Xi(2,0) + PdF1113*Xi(0,0)*Xi(2,2) + PdF1113*Xi(0,2)*Xi(2,0) + PdF1212*Xi(0,1)*Xi(2,1) + PdF1213*Xi(0,1)*Xi(2,2) + PdF1213*Xi(0,2)*Xi(2,1) + PdF1313*Xi(0,2)*Xi(2,2);
  K(0,7)=Xi(0,0)*(PdF1121*Xi(2,0) + PdF1122*Xi(2,1)) + PdF1123*Xi(0,0)*Xi(2,2) + PdF1221*Xi(0,1)*Xi(2,0) + PdF1222*Xi(0,1)*Xi(2,1) + PdF1223*Xi(0,1)*Xi(2,2) + PdF1321*Xi(0,2)*Xi(2,0) + PdF1322*Xi(0,2)*Xi(2,1) + PdF1323*Xi(0,2)*Xi(2,2);
  K(0,8)=Xi(0,0)*(PdF1131*Xi(2,0) + PdF1132*Xi(2,1)) + PdF1133*Xi(0,0)*Xi(2,2) + PdF1231*Xi(0,1)*Xi(2,0) + PdF1232*Xi(0,1)*Xi(2,1) + PdF1233*Xi(0,1)*Xi(2,2) + PdF1331*Xi(0,2)*Xi(2,0) + PdF1332*Xi(0,2)*Xi(2,1) + PdF1333*Xi(0,2)*Xi(2,2);
  K(0,9)=Xi(0,0)*(PdF1111*Xi(3,0) + PdF1112*Xi(3,1)) + PdF1112*Xi(0,1)*Xi(3,0) + PdF1113*Xi(0,0)*Xi(3,2) + PdF1113*Xi(0,2)*Xi(3,0) + PdF1212*Xi(0,1)*Xi(3,1) + PdF1213*Xi(0,1)*Xi(3,2) + PdF1213*Xi(0,2)*Xi(3,1) + PdF1313*Xi(0,2)*Xi(3,2);
  K(0,10)=Xi(0,0)*(PdF1121*Xi(3,0) + PdF1122*Xi(3,1)) + PdF1123*Xi(0,0)*Xi(3,2) + PdF1221*Xi(0,1)*Xi(3,0) + PdF1222*Xi(0,1)*Xi(3,1) + PdF1223*Xi(0,1)*Xi(3,2) + PdF1321*Xi(0,2)*Xi(3,0) + PdF1322*Xi(0,2)*Xi(3,1) + PdF1323*Xi(0,2)*Xi(3,2);
  K(0,11)=Xi(0,0)*(PdF1131*Xi(3,0) + PdF1132*Xi(3,1)) + PdF1133*Xi(0,0)*Xi(3,2) + PdF1231*Xi(0,1)*Xi(3,0) + PdF1232*Xi(0,1)*Xi(3,1) + PdF1233*Xi(0,1)*Xi(3,2) + PdF1331*Xi(0,2)*Xi(3,0) + PdF1332*Xi(0,2)*Xi(3,1) + PdF1333*Xi(0,2)*Xi(3,2);
  K(0,12)=Xi(0,0)*(PdF1111*Xi(4,0) + PdF1112*Xi(4,1)) + PdF1112*Xi(0,1)*Xi(4,0) + PdF1113*Xi(0,0)*Xi(4,2) + PdF1113*Xi(0,2)*Xi(4,0) + PdF1212*Xi(0,1)*Xi(4,1) + PdF1213*Xi(0,1)*Xi(4,2) + PdF1213*Xi(0,2)*Xi(4,1) + PdF1313*Xi(0,2)*Xi(4,2);
  K(0,13)=Xi(0,0)*(PdF1121*Xi(4,0) + PdF1122*Xi(4,1)) + PdF1123*Xi(0,0)*Xi(4,2) + PdF1221*Xi(0,1)*Xi(4,0) + PdF1222*Xi(0,1)*Xi(4,1) + PdF1223*Xi(0,1)*Xi(4,2) + PdF1321*Xi(0,2)*Xi(4,0) + PdF1322*Xi(0,2)*Xi(4,1) + PdF1323*Xi(0,2)*Xi(4,2);
  K(0,14)=Xi(0,0)*(PdF1131*Xi(4,0) + PdF1132*Xi(4,1)) + PdF1133*Xi(0,0)*Xi(4,2) + PdF1231*Xi(0,1)*Xi(4,0) + PdF1232*Xi(0,1)*Xi(4,1) + PdF1233*Xi(0,1)*Xi(4,2) + PdF1331*Xi(0,2)*Xi(4,0) + PdF1332*Xi(0,2)*Xi(4,1) + PdF1333*Xi(0,2)*Xi(4,2);
  K(0,15)=Xi(0,0)*(PdF1111*Xi(5,0) + PdF1112*Xi(5,1)) + PdF1112*Xi(0,1)*Xi(5,0) + PdF1113*Xi(0,0)*Xi(5,2) + PdF1113*Xi(0,2)*Xi(5,0) + PdF1212*Xi(0,1)*Xi(5,1) + PdF1213*Xi(0,1)*Xi(5,2) + PdF1213*Xi(0,2)*Xi(5,1) + PdF1313*Xi(0,2)*Xi(5,2);
  K(0,16)=Xi(0,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1)) + PdF1123*Xi(0,0)*Xi(5,2) + PdF1221*Xi(0,1)*Xi(5,0) + PdF1222*Xi(0,1)*Xi(5,1) + PdF1223*Xi(0,1)*Xi(5,2) + PdF1321*Xi(0,2)*Xi(5,0) + PdF1322*Xi(0,2)*Xi(5,1) + PdF1323*Xi(0,2)*Xi(5,2);
  K(0,17)=Xi(0,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1)) + PdF1133*Xi(0,0)*Xi(5,2) + PdF1231*Xi(0,1)*Xi(5,0) + PdF1232*Xi(0,1)*Xi(5,1) + PdF1233*Xi(0,1)*Xi(5,2) + PdF1331*Xi(0,2)*Xi(5,0) + PdF1332*Xi(0,2)*Xi(5,1) + PdF1333*Xi(0,2)*Xi(5,2);
  K(0,18)=Xi(0,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(0,1)*Xi(6,0) + PdF1113*Xi(0,0)*Xi(6,2) + PdF1113*Xi(0,2)*Xi(6,0) + PdF1212*Xi(0,1)*Xi(6,1) + PdF1213*Xi(0,1)*Xi(6,2) + PdF1213*Xi(0,2)*Xi(6,1) + PdF1313*Xi(0,2)*Xi(6,2);
  K(0,19)=Xi(0,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(0,0)*Xi(6,2) + PdF1221*Xi(0,1)*Xi(6,0) + PdF1222*Xi(0,1)*Xi(6,1) + PdF1223*Xi(0,1)*Xi(6,2) + PdF1321*Xi(0,2)*Xi(6,0) + PdF1322*Xi(0,2)*Xi(6,1) + PdF1323*Xi(0,2)*Xi(6,2);
  K(0,20)=Xi(0,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(0,0)*Xi(6,2) + PdF1231*Xi(0,1)*Xi(6,0) + PdF1232*Xi(0,1)*Xi(6,1) + PdF1233*Xi(0,1)*Xi(6,2) + PdF1331*Xi(0,2)*Xi(6,0) + PdF1332*Xi(0,2)*Xi(6,1) + PdF1333*Xi(0,2)*Xi(6,2);
  K(0,21)=Xi(0,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(0,1)*Xi(7,0) + PdF1113*Xi(0,0)*Xi(7,2) + PdF1113*Xi(0,2)*Xi(7,0) + PdF1212*Xi(0,1)*Xi(7,1) + PdF1213*Xi(0,1)*Xi(7,2) + PdF1213*Xi(0,2)*Xi(7,1) + PdF1313*Xi(0,2)*Xi(7,2);
  K(0,22)=Xi(0,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(0,0)*Xi(7,2) + PdF1221*Xi(0,1)*Xi(7,0) + PdF1222*Xi(0,1)*Xi(7,1) + PdF1223*Xi(0,1)*Xi(7,2) + PdF1321*Xi(0,2)*Xi(7,0) + PdF1322*Xi(0,2)*Xi(7,1) + PdF1323*Xi(0,2)*Xi(7,2);
  K(0,23)=Xi(0,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(0,0)*Xi(7,2) + PdF1231*Xi(0,1)*Xi(7,0) + PdF1232*Xi(0,1)*Xi(7,1) + PdF1233*Xi(0,1)*Xi(7,2) + PdF1331*Xi(0,2)*Xi(7,0) + PdF1332*Xi(0,2)*Xi(7,1) + PdF1333*Xi(0,2)*Xi(7,2);
  K(1,1)=PdF2121*pow(Xi(0,0),2) + 2*PdF2122*Xi(0,0)*Xi(0,1) + 2*PdF2123*Xi(0,0)*Xi(0,2) + PdF2222*pow(Xi(0,1),2) + 2*PdF2223*Xi(0,1)*Xi(0,2) + PdF2323*pow(Xi(0,2),2);
  K(1,2)=Xi(0,0)*(PdF2131*Xi(0,0) + PdF2132*Xi(0,1) + PdF2133*Xi(0,2)) + PdF2232*pow(Xi(0,1),2) + PdF2333*pow(Xi(0,2),2) + PdF2231*Xi(0,0)*Xi(0,1) + PdF2233*Xi(0,1)*Xi(0,2) + PdF2331*Xi(0,0)*Xi(0,2) + PdF2332*Xi(0,1)*Xi(0,2);
  K(1,3)=Xi(0,0)*(PdF1121*Xi(1,0) + PdF1221*Xi(1,1)) + PdF1122*Xi(0,1)*Xi(1,0) + PdF1123*Xi(1,0)*Xi(0,2) + PdF1222*Xi(0,1)*Xi(1,1) + PdF1223*Xi(0,2)*Xi(1,1) + PdF1321*Xi(0,0)*Xi(1,2) + PdF1322*Xi(0,1)*Xi(1,2) + PdF1323*Xi(0,2)*Xi(1,2);
  K(1,4)=Xi(0,0)*(PdF2121*Xi(1,0) + PdF2122*Xi(1,1)) + PdF2122*Xi(0,1)*Xi(1,0) + PdF2123*Xi(0,0)*Xi(1,2) + PdF2123*Xi(1,0)*Xi(0,2) + PdF2222*Xi(0,1)*Xi(1,1) + PdF2223*Xi(0,1)*Xi(1,2) + PdF2223*Xi(0,2)*Xi(1,1) + PdF2323*Xi(0,2)*Xi(1,2);
  K(1,5)=Xi(0,0)*(PdF2131*Xi(1,0) + PdF2132*Xi(1,1)) + PdF2133*Xi(0,0)*Xi(1,2) + PdF2231*Xi(0,1)*Xi(1,0) + PdF2232*Xi(0,1)*Xi(1,1) + PdF2233*Xi(0,1)*Xi(1,2) + PdF2331*Xi(1,0)*Xi(0,2) + PdF2332*Xi(0,2)*Xi(1,1) + PdF2333*Xi(0,2)*Xi(1,2);
  K(1,6)=Xi(0,0)*(PdF1121*Xi(2,0) + PdF1221*Xi(2,1)) + PdF1122*Xi(0,1)*Xi(2,0) + PdF1123*Xi(0,2)*Xi(2,0) + PdF1222*Xi(0,1)*Xi(2,1) + PdF1223*Xi(0,2)*Xi(2,1) + PdF1321*Xi(0,0)*Xi(2,2) + PdF1322*Xi(0,1)*Xi(2,2) + PdF1323*Xi(0,2)*Xi(2,2);
  K(1,7)=Xi(0,0)*(PdF2121*Xi(2,0) + PdF2122*Xi(2,1)) + PdF2122*Xi(0,1)*Xi(2,0) + PdF2123*Xi(0,0)*Xi(2,2) + PdF2123*Xi(0,2)*Xi(2,0) + PdF2222*Xi(0,1)*Xi(2,1) + PdF2223*Xi(0,1)*Xi(2,2) + PdF2223*Xi(0,2)*Xi(2,1) + PdF2323*Xi(0,2)*Xi(2,2);
  K(1,8)=Xi(0,0)*(PdF2131*Xi(2,0) + PdF2132*Xi(2,1)) + PdF2133*Xi(0,0)*Xi(2,2) + PdF2231*Xi(0,1)*Xi(2,0) + PdF2232*Xi(0,1)*Xi(2,1) + PdF2233*Xi(0,1)*Xi(2,2) + PdF2331*Xi(0,2)*Xi(2,0) + PdF2332*Xi(0,2)*Xi(2,1) + PdF2333*Xi(0,2)*Xi(2,2);
  K(1,9)=Xi(0,0)*(PdF1121*Xi(3,0) + PdF1221*Xi(3,1)) + PdF1122*Xi(0,1)*Xi(3,0) + PdF1123*Xi(0,2)*Xi(3,0) + PdF1222*Xi(0,1)*Xi(3,1) + PdF1223*Xi(0,2)*Xi(3,1) + PdF1321*Xi(0,0)*Xi(3,2) + PdF1322*Xi(0,1)*Xi(3,2) + PdF1323*Xi(0,2)*Xi(3,2);
  K(1,10)=Xi(0,0)*(PdF2121*Xi(3,0) + PdF2122*Xi(3,1)) + PdF2122*Xi(0,1)*Xi(3,0) + PdF2123*Xi(0,0)*Xi(3,2) + PdF2123*Xi(0,2)*Xi(3,0) + PdF2222*Xi(0,1)*Xi(3,1) + PdF2223*Xi(0,1)*Xi(3,2) + PdF2223*Xi(0,2)*Xi(3,1) + PdF2323*Xi(0,2)*Xi(3,2);
  K(1,11)=Xi(0,0)*(PdF2131*Xi(3,0) + PdF2132*Xi(3,1)) + PdF2133*Xi(0,0)*Xi(3,2) + PdF2231*Xi(0,1)*Xi(3,0) + PdF2232*Xi(0,1)*Xi(3,1) + PdF2233*Xi(0,1)*Xi(3,2) + PdF2331*Xi(0,2)*Xi(3,0) + PdF2332*Xi(0,2)*Xi(3,1) + PdF2333*Xi(0,2)*Xi(3,2);
  K(1,12)=Xi(0,0)*(PdF1121*Xi(4,0) + PdF1221*Xi(4,1)) + PdF1122*Xi(0,1)*Xi(4,0) + PdF1123*Xi(0,2)*Xi(4,0) + PdF1222*Xi(0,1)*Xi(4,1) + PdF1223*Xi(0,2)*Xi(4,1) + PdF1321*Xi(0,0)*Xi(4,2) + PdF1322*Xi(0,1)*Xi(4,2) + PdF1323*Xi(0,2)*Xi(4,2);
  K(1,13)=Xi(0,0)*(PdF2121*Xi(4,0) + PdF2122*Xi(4,1)) + PdF2122*Xi(0,1)*Xi(4,0) + PdF2123*Xi(0,0)*Xi(4,2) + PdF2123*Xi(0,2)*Xi(4,0) + PdF2222*Xi(0,1)*Xi(4,1) + PdF2223*Xi(0,1)*Xi(4,2) + PdF2223*Xi(0,2)*Xi(4,1) + PdF2323*Xi(0,2)*Xi(4,2);
  K(1,14)=Xi(0,0)*(PdF2131*Xi(4,0) + PdF2132*Xi(4,1)) + PdF2133*Xi(0,0)*Xi(4,2) + PdF2231*Xi(0,1)*Xi(4,0) + PdF2232*Xi(0,1)*Xi(4,1) + PdF2233*Xi(0,1)*Xi(4,2) + PdF2331*Xi(0,2)*Xi(4,0) + PdF2332*Xi(0,2)*Xi(4,1) + PdF2333*Xi(0,2)*Xi(4,2);
  K(1,15)=Xi(0,0)*(PdF1121*Xi(5,0) + PdF1221*Xi(5,1)) + PdF1122*Xi(0,1)*Xi(5,0) + PdF1123*Xi(0,2)*Xi(5,0) + PdF1222*Xi(0,1)*Xi(5,1) + PdF1223*Xi(0,2)*Xi(5,1) + PdF1321*Xi(0,0)*Xi(5,2) + PdF1322*Xi(0,1)*Xi(5,2) + PdF1323*Xi(0,2)*Xi(5,2);
  K(1,16)=Xi(0,0)*(PdF2121*Xi(5,0) + PdF2122*Xi(5,1)) + PdF2122*Xi(0,1)*Xi(5,0) + PdF2123*Xi(0,0)*Xi(5,2) + PdF2123*Xi(0,2)*Xi(5,0) + PdF2222*Xi(0,1)*Xi(5,1) + PdF2223*Xi(0,1)*Xi(5,2) + PdF2223*Xi(0,2)*Xi(5,1) + PdF2323*Xi(0,2)*Xi(5,2);
  K(1,17)=Xi(0,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1)) + PdF2133*Xi(0,0)*Xi(5,2) + PdF2231*Xi(0,1)*Xi(5,0) + PdF2232*Xi(0,1)*Xi(5,1) + PdF2233*Xi(0,1)*Xi(5,2) + PdF2331*Xi(0,2)*Xi(5,0) + PdF2332*Xi(0,2)*Xi(5,1) + PdF2333*Xi(0,2)*Xi(5,2);
  K(1,18)=Xi(0,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(0,1)*Xi(6,0) + PdF1123*Xi(0,2)*Xi(6,0) + PdF1222*Xi(0,1)*Xi(6,1) + PdF1223*Xi(0,2)*Xi(6,1) + PdF1321*Xi(0,0)*Xi(6,2) + PdF1322*Xi(0,1)*Xi(6,2) + PdF1323*Xi(0,2)*Xi(6,2);
  K(1,19)=Xi(0,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(0,1)*Xi(6,0) + PdF2123*Xi(0,0)*Xi(6,2) + PdF2123*Xi(0,2)*Xi(6,0) + PdF2222*Xi(0,1)*Xi(6,1) + PdF2223*Xi(0,1)*Xi(6,2) + PdF2223*Xi(0,2)*Xi(6,1) + PdF2323*Xi(0,2)*Xi(6,2);
  K(1,20)=Xi(0,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(0,0)*Xi(6,2) + PdF2231*Xi(0,1)*Xi(6,0) + PdF2232*Xi(0,1)*Xi(6,1) + PdF2233*Xi(0,1)*Xi(6,2) + PdF2331*Xi(0,2)*Xi(6,0) + PdF2332*Xi(0,2)*Xi(6,1) + PdF2333*Xi(0,2)*Xi(6,2);
  K(1,21)=Xi(0,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(0,1)*Xi(7,0) + PdF1123*Xi(0,2)*Xi(7,0) + PdF1222*Xi(0,1)*Xi(7,1) + PdF1223*Xi(0,2)*Xi(7,1) + PdF1321*Xi(0,0)*Xi(7,2) + PdF1322*Xi(0,1)*Xi(7,2) + PdF1323*Xi(0,2)*Xi(7,2);
  K(1,22)=Xi(0,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(0,1)*Xi(7,0) + PdF2123*Xi(0,0)*Xi(7,2) + PdF2123*Xi(0,2)*Xi(7,0) + PdF2222*Xi(0,1)*Xi(7,1) + PdF2223*Xi(0,1)*Xi(7,2) + PdF2223*Xi(0,2)*Xi(7,1) + PdF2323*Xi(0,2)*Xi(7,2);
  K(1,23)=Xi(0,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(0,0)*Xi(7,2) + PdF2231*Xi(0,1)*Xi(7,0) + PdF2232*Xi(0,1)*Xi(7,1) + PdF2233*Xi(0,1)*Xi(7,2) + PdF2331*Xi(0,2)*Xi(7,0) + PdF2332*Xi(0,2)*Xi(7,1) + PdF2333*Xi(0,2)*Xi(7,2);
  K(2,2)=PdF3131*pow(Xi(0,0),2) + 2*PdF3132*Xi(0,0)*Xi(0,1) + 2*PdF3133*Xi(0,0)*Xi(0,2) + PdF3232*pow(Xi(0,1),2) + 2*PdF3233*Xi(0,1)*Xi(0,2) + PdF3333*pow(Xi(0,2),2);
  K(2,3)=Xi(0,0)*(PdF1131*Xi(1,0) + PdF1231*Xi(1,1)) + PdF1132*Xi(0,1)*Xi(1,0) + PdF1133*Xi(1,0)*Xi(0,2) + PdF1232*Xi(0,1)*Xi(1,1) + PdF1233*Xi(0,2)*Xi(1,1) + PdF1331*Xi(0,0)*Xi(1,2) + PdF1332*Xi(0,1)*Xi(1,2) + PdF1333*Xi(0,2)*Xi(1,2);
  K(2,4)=Xi(0,0)*(PdF2131*Xi(1,0) + PdF2231*Xi(1,1)) + PdF2132*Xi(0,1)*Xi(1,0) + PdF2133*Xi(1,0)*Xi(0,2) + PdF2232*Xi(0,1)*Xi(1,1) + PdF2233*Xi(0,2)*Xi(1,1) + PdF2331*Xi(0,0)*Xi(1,2) + PdF2332*Xi(0,1)*Xi(1,2) + PdF2333*Xi(0,2)*Xi(1,2);
  K(2,5)=Xi(0,0)*(PdF3131*Xi(1,0) + PdF3132*Xi(1,1)) + PdF3132*Xi(0,1)*Xi(1,0) + PdF3133*Xi(0,0)*Xi(1,2) + PdF3133*Xi(1,0)*Xi(0,2) + PdF3232*Xi(0,1)*Xi(1,1) + PdF3233*Xi(0,1)*Xi(1,2) + PdF3233*Xi(0,2)*Xi(1,1) + PdF3333*Xi(0,2)*Xi(1,2);
  K(2,6)=Xi(0,0)*(PdF1131*Xi(2,0) + PdF1231*Xi(2,1)) + PdF1132*Xi(0,1)*Xi(2,0) + PdF1133*Xi(0,2)*Xi(2,0) + PdF1232*Xi(0,1)*Xi(2,1) + PdF1233*Xi(0,2)*Xi(2,1) + PdF1331*Xi(0,0)*Xi(2,2) + PdF1332*Xi(0,1)*Xi(2,2) + PdF1333*Xi(0,2)*Xi(2,2);
  K(2,7)=Xi(0,0)*(PdF2131*Xi(2,0) + PdF2231*Xi(2,1)) + PdF2132*Xi(0,1)*Xi(2,0) + PdF2133*Xi(0,2)*Xi(2,0) + PdF2232*Xi(0,1)*Xi(2,1) + PdF2233*Xi(0,2)*Xi(2,1) + PdF2331*Xi(0,0)*Xi(2,2) + PdF2332*Xi(0,1)*Xi(2,2) + PdF2333*Xi(0,2)*Xi(2,2);
  K(2,8)=Xi(0,0)*(PdF3131*Xi(2,0) + PdF3132*Xi(2,1)) + PdF3132*Xi(0,1)*Xi(2,0) + PdF3133*Xi(0,0)*Xi(2,2) + PdF3133*Xi(0,2)*Xi(2,0) + PdF3232*Xi(0,1)*Xi(2,1) + PdF3233*Xi(0,1)*Xi(2,2) + PdF3233*Xi(0,2)*Xi(2,1) + PdF3333*Xi(0,2)*Xi(2,2);
  K(2,9)=Xi(0,0)*(PdF1131*Xi(3,0) + PdF1231*Xi(3,1)) + PdF1132*Xi(0,1)*Xi(3,0) + PdF1133*Xi(0,2)*Xi(3,0) + PdF1232*Xi(0,1)*Xi(3,1) + PdF1233*Xi(0,2)*Xi(3,1) + PdF1331*Xi(0,0)*Xi(3,2) + PdF1332*Xi(0,1)*Xi(3,2) + PdF1333*Xi(0,2)*Xi(3,2);
  K(2,10)=Xi(0,0)*(PdF2131*Xi(3,0) + PdF2231*Xi(3,1)) + PdF2132*Xi(0,1)*Xi(3,0) + PdF2133*Xi(0,2)*Xi(3,0) + PdF2232*Xi(0,1)*Xi(3,1) + PdF2233*Xi(0,2)*Xi(3,1) + PdF2331*Xi(0,0)*Xi(3,2) + PdF2332*Xi(0,1)*Xi(3,2) + PdF2333*Xi(0,2)*Xi(3,2);
  K(2,11)=Xi(0,0)*(PdF3131*Xi(3,0) + PdF3132*Xi(3,1)) + PdF3132*Xi(0,1)*Xi(3,0) + PdF3133*Xi(0,0)*Xi(3,2) + PdF3133*Xi(0,2)*Xi(3,0) + PdF3232*Xi(0,1)*Xi(3,1) + PdF3233*Xi(0,1)*Xi(3,2) + PdF3233*Xi(0,2)*Xi(3,1) + PdF3333*Xi(0,2)*Xi(3,2);
  K(2,12)=Xi(0,0)*(PdF1131*Xi(4,0) + PdF1231*Xi(4,1)) + PdF1132*Xi(0,1)*Xi(4,0) + PdF1133*Xi(0,2)*Xi(4,0) + PdF1232*Xi(0,1)*Xi(4,1) + PdF1233*Xi(0,2)*Xi(4,1) + PdF1331*Xi(0,0)*Xi(4,2) + PdF1332*Xi(0,1)*Xi(4,2) + PdF1333*Xi(0,2)*Xi(4,2);
  K(2,13)=Xi(0,0)*(PdF2131*Xi(4,0) + PdF2231*Xi(4,1)) + PdF2132*Xi(0,1)*Xi(4,0) + PdF2133*Xi(0,2)*Xi(4,0) + PdF2232*Xi(0,1)*Xi(4,1) + PdF2233*Xi(0,2)*Xi(4,1) + PdF2331*Xi(0,0)*Xi(4,2) + PdF2332*Xi(0,1)*Xi(4,2) + PdF2333*Xi(0,2)*Xi(4,2);
  K(2,14)=Xi(0,0)*(PdF3131*Xi(4,0) + PdF3132*Xi(4,1)) + PdF3132*Xi(0,1)*Xi(4,0) + PdF3133*Xi(0,0)*Xi(4,2) + PdF3133*Xi(0,2)*Xi(4,0) + PdF3232*Xi(0,1)*Xi(4,1) + PdF3233*Xi(0,1)*Xi(4,2) + PdF3233*Xi(0,2)*Xi(4,1) + PdF3333*Xi(0,2)*Xi(4,2);
  K(2,15)=Xi(0,0)*(PdF1131*Xi(5,0) + PdF1231*Xi(5,1)) + PdF1132*Xi(0,1)*Xi(5,0) + PdF1133*Xi(0,2)*Xi(5,0) + PdF1232*Xi(0,1)*Xi(5,1) + PdF1233*Xi(0,2)*Xi(5,1) + PdF1331*Xi(0,0)*Xi(5,2) + PdF1332*Xi(0,1)*Xi(5,2) + PdF1333*Xi(0,2)*Xi(5,2);
  K(2,16)=Xi(0,0)*(PdF2131*Xi(5,0) + PdF2231*Xi(5,1)) + PdF2132*Xi(0,1)*Xi(5,0) + PdF2133*Xi(0,2)*Xi(5,0) + PdF2232*Xi(0,1)*Xi(5,1) + PdF2233*Xi(0,2)*Xi(5,1) + PdF2331*Xi(0,0)*Xi(5,2) + PdF2332*Xi(0,1)*Xi(5,2) + PdF2333*Xi(0,2)*Xi(5,2);
  K(2,17)=Xi(0,0)*(PdF3131*Xi(5,0) + PdF3132*Xi(5,1)) + PdF3132*Xi(0,1)*Xi(5,0) + PdF3133*Xi(0,0)*Xi(5,2) + PdF3133*Xi(0,2)*Xi(5,0) + PdF3232*Xi(0,1)*Xi(5,1) + PdF3233*Xi(0,1)*Xi(5,2) + PdF3233*Xi(0,2)*Xi(5,1) + PdF3333*Xi(0,2)*Xi(5,2);
  K(2,18)=Xi(0,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(0,1)*Xi(6,0) + PdF1133*Xi(0,2)*Xi(6,0) + PdF1232*Xi(0,1)*Xi(6,1) + PdF1233*Xi(0,2)*Xi(6,1) + PdF1331*Xi(0,0)*Xi(6,2) + PdF1332*Xi(0,1)*Xi(6,2) + PdF1333*Xi(0,2)*Xi(6,2);
  K(2,19)=Xi(0,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(0,1)*Xi(6,0) + PdF2133*Xi(0,2)*Xi(6,0) + PdF2232*Xi(0,1)*Xi(6,1) + PdF2233*Xi(0,2)*Xi(6,1) + PdF2331*Xi(0,0)*Xi(6,2) + PdF2332*Xi(0,1)*Xi(6,2) + PdF2333*Xi(0,2)*Xi(6,2);
  K(2,20)=Xi(0,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(0,1)*Xi(6,0) + PdF3133*Xi(0,0)*Xi(6,2) + PdF3133*Xi(0,2)*Xi(6,0) + PdF3232*Xi(0,1)*Xi(6,1) + PdF3233*Xi(0,1)*Xi(6,2) + PdF3233*Xi(0,2)*Xi(6,1) + PdF3333*Xi(0,2)*Xi(6,2);
  K(2,21)=Xi(0,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(0,1)*Xi(7,0) + PdF1133*Xi(0,2)*Xi(7,0) + PdF1232*Xi(0,1)*Xi(7,1) + PdF1233*Xi(0,2)*Xi(7,1) + PdF1331*Xi(0,0)*Xi(7,2) + PdF1332*Xi(0,1)*Xi(7,2) + PdF1333*Xi(0,2)*Xi(7,2);
  K(2,22)=Xi(0,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(0,1)*Xi(7,0) + PdF2133*Xi(0,2)*Xi(7,0) + PdF2232*Xi(0,1)*Xi(7,1) + PdF2233*Xi(0,2)*Xi(7,1) + PdF2331*Xi(0,0)*Xi(7,2) + PdF2332*Xi(0,1)*Xi(7,2) + PdF2333*Xi(0,2)*Xi(7,2);
  K(2,23)=Xi(0,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(0,1)*Xi(7,0) + PdF3133*Xi(0,0)*Xi(7,2) + PdF3133*Xi(0,2)*Xi(7,0) + PdF3232*Xi(0,1)*Xi(7,1) + PdF3233*Xi(0,1)*Xi(7,2) + PdF3233*Xi(0,2)*Xi(7,1) + PdF3333*Xi(0,2)*Xi(7,2);
  K(3,3)=PdF1111*pow(Xi(1,0),2) + 2*PdF1112*Xi(1,0)*Xi(1,1) + 2*PdF1113*Xi(1,0)*Xi(1,2) + PdF1212*pow(Xi(1,1),2) + 2*PdF1213*Xi(1,1)*Xi(1,2) + PdF1313*pow(Xi(1,2),2);
  K(3,4)=Xi(1,0)*(PdF1121*Xi(1,0) + PdF1122*Xi(1,1) + PdF1123*Xi(1,2)) + PdF1222*pow(Xi(1,1),2) + PdF1323*pow(Xi(1,2),2) + PdF1221*Xi(1,0)*Xi(1,1) + PdF1223*Xi(1,1)*Xi(1,2) + PdF1321*Xi(1,0)*Xi(1,2) + PdF1322*Xi(1,1)*Xi(1,2);
  K(3,5)=Xi(1,0)*(PdF1131*Xi(1,0) + PdF1132*Xi(1,1) + PdF1133*Xi(1,2)) + PdF1232*pow(Xi(1,1),2) + PdF1333*pow(Xi(1,2),2) + PdF1231*Xi(1,0)*Xi(1,1) + PdF1233*Xi(1,1)*Xi(1,2) + PdF1331*Xi(1,0)*Xi(1,2) + PdF1332*Xi(1,1)*Xi(1,2);
  K(3,6)=Xi(1,0)*(PdF1111*Xi(2,0) + PdF1112*Xi(2,1)) + PdF1112*Xi(1,1)*Xi(2,0) + PdF1113*Xi(1,0)*Xi(2,2) + PdF1113*Xi(2,0)*Xi(1,2) + PdF1212*Xi(1,1)*Xi(2,1) + PdF1213*Xi(1,1)*Xi(2,2) + PdF1213*Xi(1,2)*Xi(2,1) + PdF1313*Xi(1,2)*Xi(2,2);
  K(3,7)=Xi(1,0)*(PdF1121*Xi(2,0) + PdF1122*Xi(2,1)) + PdF1123*Xi(1,0)*Xi(2,2) + PdF1221*Xi(1,1)*Xi(2,0) + PdF1222*Xi(1,1)*Xi(2,1) + PdF1223*Xi(1,1)*Xi(2,2) + PdF1321*Xi(2,0)*Xi(1,2) + PdF1322*Xi(1,2)*Xi(2,1) + PdF1323*Xi(1,2)*Xi(2,2);
  K(3,8)=Xi(1,0)*(PdF1131*Xi(2,0) + PdF1132*Xi(2,1)) + PdF1133*Xi(1,0)*Xi(2,2) + PdF1231*Xi(1,1)*Xi(2,0) + PdF1232*Xi(1,1)*Xi(2,1) + PdF1233*Xi(1,1)*Xi(2,2) + PdF1331*Xi(2,0)*Xi(1,2) + PdF1332*Xi(1,2)*Xi(2,1) + PdF1333*Xi(1,2)*Xi(2,2);
  K(3,9)=Xi(1,0)*(PdF1111*Xi(3,0) + PdF1112*Xi(3,1)) + PdF1112*Xi(1,1)*Xi(3,0) + PdF1113*Xi(1,0)*Xi(3,2) + PdF1113*Xi(1,2)*Xi(3,0) + PdF1212*Xi(1,1)*Xi(3,1) + PdF1213*Xi(1,1)*Xi(3,2) + PdF1213*Xi(1,2)*Xi(3,1) + PdF1313*Xi(1,2)*Xi(3,2);
  K(3,10)=Xi(1,0)*(PdF1121*Xi(3,0) + PdF1122*Xi(3,1)) + PdF1123*Xi(1,0)*Xi(3,2) + PdF1221*Xi(1,1)*Xi(3,0) + PdF1222*Xi(1,1)*Xi(3,1) + PdF1223*Xi(1,1)*Xi(3,2) + PdF1321*Xi(1,2)*Xi(3,0) + PdF1322*Xi(1,2)*Xi(3,1) + PdF1323*Xi(1,2)*Xi(3,2);
  K(3,11)=Xi(1,0)*(PdF1131*Xi(3,0) + PdF1132*Xi(3,1)) + PdF1133*Xi(1,0)*Xi(3,2) + PdF1231*Xi(1,1)*Xi(3,0) + PdF1232*Xi(1,1)*Xi(3,1) + PdF1233*Xi(1,1)*Xi(3,2) + PdF1331*Xi(1,2)*Xi(3,0) + PdF1332*Xi(1,2)*Xi(3,1) + PdF1333*Xi(1,2)*Xi(3,2);
  K(3,12)=Xi(1,0)*(PdF1111*Xi(4,0) + PdF1112*Xi(4,1)) + PdF1112*Xi(1,1)*Xi(4,0) + PdF1113*Xi(1,0)*Xi(4,2) + PdF1113*Xi(1,2)*Xi(4,0) + PdF1212*Xi(1,1)*Xi(4,1) + PdF1213*Xi(1,1)*Xi(4,2) + PdF1213*Xi(1,2)*Xi(4,1) + PdF1313*Xi(1,2)*Xi(4,2);
  K(3,13)=Xi(1,0)*(PdF1121*Xi(4,0) + PdF1122*Xi(4,1)) + PdF1123*Xi(1,0)*Xi(4,2) + PdF1221*Xi(1,1)*Xi(4,0) + PdF1222*Xi(1,1)*Xi(4,1) + PdF1223*Xi(1,1)*Xi(4,2) + PdF1321*Xi(1,2)*Xi(4,0) + PdF1322*Xi(1,2)*Xi(4,1) + PdF1323*Xi(1,2)*Xi(4,2);
  K(3,14)=Xi(1,0)*(PdF1131*Xi(4,0) + PdF1132*Xi(4,1)) + PdF1133*Xi(1,0)*Xi(4,2) + PdF1231*Xi(1,1)*Xi(4,0) + PdF1232*Xi(1,1)*Xi(4,1) + PdF1233*Xi(1,1)*Xi(4,2) + PdF1331*Xi(1,2)*Xi(4,0) + PdF1332*Xi(1,2)*Xi(4,1) + PdF1333*Xi(1,2)*Xi(4,2);
  K(3,15)=Xi(1,0)*(PdF1111*Xi(5,0) + PdF1112*Xi(5,1)) + PdF1112*Xi(1,1)*Xi(5,0) + PdF1113*Xi(1,0)*Xi(5,2) + PdF1113*Xi(1,2)*Xi(5,0) + PdF1212*Xi(1,1)*Xi(5,1) + PdF1213*Xi(1,1)*Xi(5,2) + PdF1213*Xi(1,2)*Xi(5,1) + PdF1313*Xi(1,2)*Xi(5,2);
  K(3,16)=Xi(1,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1)) + PdF1123*Xi(1,0)*Xi(5,2) + PdF1221*Xi(1,1)*Xi(5,0) + PdF1222*Xi(1,1)*Xi(5,1) + PdF1223*Xi(1,1)*Xi(5,2) + PdF1321*Xi(1,2)*Xi(5,0) + PdF1322*Xi(1,2)*Xi(5,1) + PdF1323*Xi(1,2)*Xi(5,2);
  K(3,17)=Xi(1,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1)) + PdF1133*Xi(1,0)*Xi(5,2) + PdF1231*Xi(1,1)*Xi(5,0) + PdF1232*Xi(1,1)*Xi(5,1) + PdF1233*Xi(1,1)*Xi(5,2) + PdF1331*Xi(1,2)*Xi(5,0) + PdF1332*Xi(1,2)*Xi(5,1) + PdF1333*Xi(1,2)*Xi(5,2);
  K(3,18)=Xi(1,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(1,1)*Xi(6,0) + PdF1113*Xi(1,0)*Xi(6,2) + PdF1113*Xi(1,2)*Xi(6,0) + PdF1212*Xi(1,1)*Xi(6,1) + PdF1213*Xi(1,1)*Xi(6,2) + PdF1213*Xi(1,2)*Xi(6,1) + PdF1313*Xi(1,2)*Xi(6,2);
  K(3,19)=Xi(1,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(1,0)*Xi(6,2) + PdF1221*Xi(1,1)*Xi(6,0) + PdF1222*Xi(1,1)*Xi(6,1) + PdF1223*Xi(1,1)*Xi(6,2) + PdF1321*Xi(1,2)*Xi(6,0) + PdF1322*Xi(1,2)*Xi(6,1) + PdF1323*Xi(1,2)*Xi(6,2);
  K(3,20)=Xi(1,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(1,0)*Xi(6,2) + PdF1231*Xi(1,1)*Xi(6,0) + PdF1232*Xi(1,1)*Xi(6,1) + PdF1233*Xi(1,1)*Xi(6,2) + PdF1331*Xi(1,2)*Xi(6,0) + PdF1332*Xi(1,2)*Xi(6,1) + PdF1333*Xi(1,2)*Xi(6,2);
  K(3,21)=Xi(1,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(1,1)*Xi(7,0) + PdF1113*Xi(1,0)*Xi(7,2) + PdF1113*Xi(1,2)*Xi(7,0) + PdF1212*Xi(1,1)*Xi(7,1) + PdF1213*Xi(1,1)*Xi(7,2) + PdF1213*Xi(1,2)*Xi(7,1) + PdF1313*Xi(1,2)*Xi(7,2);
  K(3,22)=Xi(1,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(1,0)*Xi(7,2) + PdF1221*Xi(1,1)*Xi(7,0) + PdF1222*Xi(1,1)*Xi(7,1) + PdF1223*Xi(1,1)*Xi(7,2) + PdF1321*Xi(1,2)*Xi(7,0) + PdF1322*Xi(1,2)*Xi(7,1) + PdF1323*Xi(1,2)*Xi(7,2);
  K(3,23)=Xi(1,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(1,0)*Xi(7,2) + PdF1231*Xi(1,1)*Xi(7,0) + PdF1232*Xi(1,1)*Xi(7,1) + PdF1233*Xi(1,1)*Xi(7,2) + PdF1331*Xi(1,2)*Xi(7,0) + PdF1332*Xi(1,2)*Xi(7,1) + PdF1333*Xi(1,2)*Xi(7,2);
  K(4,4)=PdF2121*pow(Xi(1,0),2) + 2*PdF2122*Xi(1,0)*Xi(1,1) + 2*PdF2123*Xi(1,0)*Xi(1,2) + PdF2222*pow(Xi(1,1),2) + 2*PdF2223*Xi(1,1)*Xi(1,2) + PdF2323*pow(Xi(1,2),2);
  K(4,5)=Xi(1,0)*(PdF2131*Xi(1,0) + PdF2132*Xi(1,1) + PdF2133*Xi(1,2)) + PdF2232*pow(Xi(1,1),2) + PdF2333*pow(Xi(1,2),2) + PdF2231*Xi(1,0)*Xi(1,1) + PdF2233*Xi(1,1)*Xi(1,2) + PdF2331*Xi(1,0)*Xi(1,2) + PdF2332*Xi(1,1)*Xi(1,2);
  K(4,6)=Xi(1,0)*(PdF1121*Xi(2,0) + PdF1221*Xi(2,1)) + PdF1122*Xi(1,1)*Xi(2,0) + PdF1123*Xi(2,0)*Xi(1,2) + PdF1222*Xi(1,1)*Xi(2,1) + PdF1223*Xi(1,2)*Xi(2,1) + PdF1321*Xi(1,0)*Xi(2,2) + PdF1322*Xi(1,1)*Xi(2,2) + PdF1323*Xi(1,2)*Xi(2,2);
  K(4,7)=Xi(1,0)*(PdF2121*Xi(2,0) + PdF2122*Xi(2,1)) + PdF2122*Xi(1,1)*Xi(2,0) + PdF2123*Xi(1,0)*Xi(2,2) + PdF2123*Xi(2,0)*Xi(1,2) + PdF2222*Xi(1,1)*Xi(2,1) + PdF2223*Xi(1,1)*Xi(2,2) + PdF2223*Xi(1,2)*Xi(2,1) + PdF2323*Xi(1,2)*Xi(2,2);
  K(4,8)=Xi(1,0)*(PdF2131*Xi(2,0) + PdF2132*Xi(2,1)) + PdF2133*Xi(1,0)*Xi(2,2) + PdF2231*Xi(1,1)*Xi(2,0) + PdF2232*Xi(1,1)*Xi(2,1) + PdF2233*Xi(1,1)*Xi(2,2) + PdF2331*Xi(2,0)*Xi(1,2) + PdF2332*Xi(1,2)*Xi(2,1) + PdF2333*Xi(1,2)*Xi(2,2);
  K(4,9)=Xi(1,0)*(PdF1121*Xi(3,0) + PdF1221*Xi(3,1)) + PdF1122*Xi(1,1)*Xi(3,0) + PdF1123*Xi(1,2)*Xi(3,0) + PdF1222*Xi(1,1)*Xi(3,1) + PdF1223*Xi(1,2)*Xi(3,1) + PdF1321*Xi(1,0)*Xi(3,2) + PdF1322*Xi(1,1)*Xi(3,2) + PdF1323*Xi(1,2)*Xi(3,2);
  K(4,10)=Xi(1,0)*(PdF2121*Xi(3,0) + PdF2122*Xi(3,1)) + PdF2122*Xi(1,1)*Xi(3,0) + PdF2123*Xi(1,0)*Xi(3,2) + PdF2123*Xi(1,2)*Xi(3,0) + PdF2222*Xi(1,1)*Xi(3,1) + PdF2223*Xi(1,1)*Xi(3,2) + PdF2223*Xi(1,2)*Xi(3,1) + PdF2323*Xi(1,2)*Xi(3,2);
  K(4,11)=Xi(1,0)*(PdF2131*Xi(3,0) + PdF2132*Xi(3,1)) + PdF2133*Xi(1,0)*Xi(3,2) + PdF2231*Xi(1,1)*Xi(3,0) + PdF2232*Xi(1,1)*Xi(3,1) + PdF2233*Xi(1,1)*Xi(3,2) + PdF2331*Xi(1,2)*Xi(3,0) + PdF2332*Xi(1,2)*Xi(3,1) + PdF2333*Xi(1,2)*Xi(3,2);
  K(4,12)=Xi(1,0)*(PdF1121*Xi(4,0) + PdF1221*Xi(4,1)) + PdF1122*Xi(1,1)*Xi(4,0) + PdF1123*Xi(1,2)*Xi(4,0) + PdF1222*Xi(1,1)*Xi(4,1) + PdF1223*Xi(1,2)*Xi(4,1) + PdF1321*Xi(1,0)*Xi(4,2) + PdF1322*Xi(1,1)*Xi(4,2) + PdF1323*Xi(1,2)*Xi(4,2);
  K(4,13)=Xi(1,0)*(PdF2121*Xi(4,0) + PdF2122*Xi(4,1)) + PdF2122*Xi(1,1)*Xi(4,0) + PdF2123*Xi(1,0)*Xi(4,2) + PdF2123*Xi(1,2)*Xi(4,0) + PdF2222*Xi(1,1)*Xi(4,1) + PdF2223*Xi(1,1)*Xi(4,2) + PdF2223*Xi(1,2)*Xi(4,1) + PdF2323*Xi(1,2)*Xi(4,2);
  K(4,14)=Xi(1,0)*(PdF2131*Xi(4,0) + PdF2132*Xi(4,1)) + PdF2133*Xi(1,0)*Xi(4,2) + PdF2231*Xi(1,1)*Xi(4,0) + PdF2232*Xi(1,1)*Xi(4,1) + PdF2233*Xi(1,1)*Xi(4,2) + PdF2331*Xi(1,2)*Xi(4,0) + PdF2332*Xi(1,2)*Xi(4,1) + PdF2333*Xi(1,2)*Xi(4,2);
  K(4,15)=Xi(1,0)*(PdF1121*Xi(5,0) + PdF1221*Xi(5,1)) + PdF1122*Xi(1,1)*Xi(5,0) + PdF1123*Xi(1,2)*Xi(5,0) + PdF1222*Xi(1,1)*Xi(5,1) + PdF1223*Xi(1,2)*Xi(5,1) + PdF1321*Xi(1,0)*Xi(5,2) + PdF1322*Xi(1,1)*Xi(5,2) + PdF1323*Xi(1,2)*Xi(5,2);
  K(4,16)=Xi(1,0)*(PdF2121*Xi(5,0) + PdF2122*Xi(5,1)) + PdF2122*Xi(1,1)*Xi(5,0) + PdF2123*Xi(1,0)*Xi(5,2) + PdF2123*Xi(1,2)*Xi(5,0) + PdF2222*Xi(1,1)*Xi(5,1) + PdF2223*Xi(1,1)*Xi(5,2) + PdF2223*Xi(1,2)*Xi(5,1) + PdF2323*Xi(1,2)*Xi(5,2);
  K(4,17)=Xi(1,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1)) + PdF2133*Xi(1,0)*Xi(5,2) + PdF2231*Xi(1,1)*Xi(5,0) + PdF2232*Xi(1,1)*Xi(5,1) + PdF2233*Xi(1,1)*Xi(5,2) + PdF2331*Xi(1,2)*Xi(5,0) + PdF2332*Xi(1,2)*Xi(5,1) + PdF2333*Xi(1,2)*Xi(5,2);
  K(4,18)=Xi(1,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(1,1)*Xi(6,0) + PdF1123*Xi(1,2)*Xi(6,0) + PdF1222*Xi(1,1)*Xi(6,1) + PdF1223*Xi(1,2)*Xi(6,1) + PdF1321*Xi(1,0)*Xi(6,2) + PdF1322*Xi(1,1)*Xi(6,2) + PdF1323*Xi(1,2)*Xi(6,2);
  K(4,19)=Xi(1,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(1,1)*Xi(6,0) + PdF2123*Xi(1,0)*Xi(6,2) + PdF2123*Xi(1,2)*Xi(6,0) + PdF2222*Xi(1,1)*Xi(6,1) + PdF2223*Xi(1,1)*Xi(6,2) + PdF2223*Xi(1,2)*Xi(6,1) + PdF2323*Xi(1,2)*Xi(6,2);
  K(4,20)=Xi(1,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(1,0)*Xi(6,2) + PdF2231*Xi(1,1)*Xi(6,0) + PdF2232*Xi(1,1)*Xi(6,1) + PdF2233*Xi(1,1)*Xi(6,2) + PdF2331*Xi(1,2)*Xi(6,0) + PdF2332*Xi(1,2)*Xi(6,1) + PdF2333*Xi(1,2)*Xi(6,2);
  K(4,21)=Xi(1,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(1,1)*Xi(7,0) + PdF1123*Xi(1,2)*Xi(7,0) + PdF1222*Xi(1,1)*Xi(7,1) + PdF1223*Xi(1,2)*Xi(7,1) + PdF1321*Xi(1,0)*Xi(7,2) + PdF1322*Xi(1,1)*Xi(7,2) + PdF1323*Xi(1,2)*Xi(7,2);
  K(4,22)=Xi(1,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(1,1)*Xi(7,0) + PdF2123*Xi(1,0)*Xi(7,2) + PdF2123*Xi(1,2)*Xi(7,0) + PdF2222*Xi(1,1)*Xi(7,1) + PdF2223*Xi(1,1)*Xi(7,2) + PdF2223*Xi(1,2)*Xi(7,1) + PdF2323*Xi(1,2)*Xi(7,2);
  K(4,23)=Xi(1,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(1,0)*Xi(7,2) + PdF2231*Xi(1,1)*Xi(7,0) + PdF2232*Xi(1,1)*Xi(7,1) + PdF2233*Xi(1,1)*Xi(7,2) + PdF2331*Xi(1,2)*Xi(7,0) + PdF2332*Xi(1,2)*Xi(7,1) + PdF2333*Xi(1,2)*Xi(7,2);
  K(5,5)=PdF3131*pow(Xi(1,0),2) + 2*PdF3132*Xi(1,0)*Xi(1,1) + 2*PdF3133*Xi(1,0)*Xi(1,2) + PdF3232*pow(Xi(1,1),2) + 2*PdF3233*Xi(1,1)*Xi(1,2) + PdF3333*pow(Xi(1,2),2);
  K(5,6)=Xi(1,0)*(PdF1131*Xi(2,0) + PdF1231*Xi(2,1)) + PdF1132*Xi(1,1)*Xi(2,0) + PdF1133*Xi(2,0)*Xi(1,2) + PdF1232*Xi(1,1)*Xi(2,1) + PdF1233*Xi(1,2)*Xi(2,1) + PdF1331*Xi(1,0)*Xi(2,2) + PdF1332*Xi(1,1)*Xi(2,2) + PdF1333*Xi(1,2)*Xi(2,2);
  K(5,7)=Xi(1,0)*(PdF2131*Xi(2,0) + PdF2231*Xi(2,1)) + PdF2132*Xi(1,1)*Xi(2,0) + PdF2133*Xi(2,0)*Xi(1,2) + PdF2232*Xi(1,1)*Xi(2,1) + PdF2233*Xi(1,2)*Xi(2,1) + PdF2331*Xi(1,0)*Xi(2,2) + PdF2332*Xi(1,1)*Xi(2,2) + PdF2333*Xi(1,2)*Xi(2,2);
  K(5,8)=Xi(1,0)*(PdF3131*Xi(2,0) + PdF3132*Xi(2,1)) + PdF3132*Xi(1,1)*Xi(2,0) + PdF3133*Xi(1,0)*Xi(2,2) + PdF3133*Xi(2,0)*Xi(1,2) + PdF3232*Xi(1,1)*Xi(2,1) + PdF3233*Xi(1,1)*Xi(2,2) + PdF3233*Xi(1,2)*Xi(2,1) + PdF3333*Xi(1,2)*Xi(2,2);
  K(5,9)=Xi(1,0)*(PdF1131*Xi(3,0) + PdF1231*Xi(3,1)) + PdF1132*Xi(1,1)*Xi(3,0) + PdF1133*Xi(1,2)*Xi(3,0) + PdF1232*Xi(1,1)*Xi(3,1) + PdF1233*Xi(1,2)*Xi(3,1) + PdF1331*Xi(1,0)*Xi(3,2) + PdF1332*Xi(1,1)*Xi(3,2) + PdF1333*Xi(1,2)*Xi(3,2);
  K(5,10)=Xi(1,0)*(PdF2131*Xi(3,0) + PdF2231*Xi(3,1)) + PdF2132*Xi(1,1)*Xi(3,0) + PdF2133*Xi(1,2)*Xi(3,0) + PdF2232*Xi(1,1)*Xi(3,1) + PdF2233*Xi(1,2)*Xi(3,1) + PdF2331*Xi(1,0)*Xi(3,2) + PdF2332*Xi(1,1)*Xi(3,2) + PdF2333*Xi(1,2)*Xi(3,2);
  K(5,11)=Xi(1,0)*(PdF3131*Xi(3,0) + PdF3132*Xi(3,1)) + PdF3132*Xi(1,1)*Xi(3,0) + PdF3133*Xi(1,0)*Xi(3,2) + PdF3133*Xi(1,2)*Xi(3,0) + PdF3232*Xi(1,1)*Xi(3,1) + PdF3233*Xi(1,1)*Xi(3,2) + PdF3233*Xi(1,2)*Xi(3,1) + PdF3333*Xi(1,2)*Xi(3,2);
  K(5,12)=Xi(1,0)*(PdF1131*Xi(4,0) + PdF1231*Xi(4,1)) + PdF1132*Xi(1,1)*Xi(4,0) + PdF1133*Xi(1,2)*Xi(4,0) + PdF1232*Xi(1,1)*Xi(4,1) + PdF1233*Xi(1,2)*Xi(4,1) + PdF1331*Xi(1,0)*Xi(4,2) + PdF1332*Xi(1,1)*Xi(4,2) + PdF1333*Xi(1,2)*Xi(4,2);
  K(5,13)=Xi(1,0)*(PdF2131*Xi(4,0) + PdF2231*Xi(4,1)) + PdF2132*Xi(1,1)*Xi(4,0) + PdF2133*Xi(1,2)*Xi(4,0) + PdF2232*Xi(1,1)*Xi(4,1) + PdF2233*Xi(1,2)*Xi(4,1) + PdF2331*Xi(1,0)*Xi(4,2) + PdF2332*Xi(1,1)*Xi(4,2) + PdF2333*Xi(1,2)*Xi(4,2);
  K(5,14)=Xi(1,0)*(PdF3131*Xi(4,0) + PdF3132*Xi(4,1)) + PdF3132*Xi(1,1)*Xi(4,0) + PdF3133*Xi(1,0)*Xi(4,2) + PdF3133*Xi(1,2)*Xi(4,0) + PdF3232*Xi(1,1)*Xi(4,1) + PdF3233*Xi(1,1)*Xi(4,2) + PdF3233*Xi(1,2)*Xi(4,1) + PdF3333*Xi(1,2)*Xi(4,2);
  K(5,15)=Xi(1,0)*(PdF1131*Xi(5,0) + PdF1231*Xi(5,1)) + PdF1132*Xi(1,1)*Xi(5,0) + PdF1133*Xi(1,2)*Xi(5,0) + PdF1232*Xi(1,1)*Xi(5,1) + PdF1233*Xi(1,2)*Xi(5,1) + PdF1331*Xi(1,0)*Xi(5,2) + PdF1332*Xi(1,1)*Xi(5,2) + PdF1333*Xi(1,2)*Xi(5,2);
  K(5,16)=Xi(1,0)*(PdF2131*Xi(5,0) + PdF2231*Xi(5,1)) + PdF2132*Xi(1,1)*Xi(5,0) + PdF2133*Xi(1,2)*Xi(5,0) + PdF2232*Xi(1,1)*Xi(5,1) + PdF2233*Xi(1,2)*Xi(5,1) + PdF2331*Xi(1,0)*Xi(5,2) + PdF2332*Xi(1,1)*Xi(5,2) + PdF2333*Xi(1,2)*Xi(5,2);
  K(5,17)=Xi(1,0)*(PdF3131*Xi(5,0) + PdF3132*Xi(5,1)) + PdF3132*Xi(1,1)*Xi(5,0) + PdF3133*Xi(1,0)*Xi(5,2) + PdF3133*Xi(1,2)*Xi(5,0) + PdF3232*Xi(1,1)*Xi(5,1) + PdF3233*Xi(1,1)*Xi(5,2) + PdF3233*Xi(1,2)*Xi(5,1) + PdF3333*Xi(1,2)*Xi(5,2);
  K(5,18)=Xi(1,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(1,1)*Xi(6,0) + PdF1133*Xi(1,2)*Xi(6,0) + PdF1232*Xi(1,1)*Xi(6,1) + PdF1233*Xi(1,2)*Xi(6,1) + PdF1331*Xi(1,0)*Xi(6,2) + PdF1332*Xi(1,1)*Xi(6,2) + PdF1333*Xi(1,2)*Xi(6,2);
  K(5,19)=Xi(1,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(1,1)*Xi(6,0) + PdF2133*Xi(1,2)*Xi(6,0) + PdF2232*Xi(1,1)*Xi(6,1) + PdF2233*Xi(1,2)*Xi(6,1) + PdF2331*Xi(1,0)*Xi(6,2) + PdF2332*Xi(1,1)*Xi(6,2) + PdF2333*Xi(1,2)*Xi(6,2);
  K(5,20)=Xi(1,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(1,1)*Xi(6,0) + PdF3133*Xi(1,0)*Xi(6,2) + PdF3133*Xi(1,2)*Xi(6,0) + PdF3232*Xi(1,1)*Xi(6,1) + PdF3233*Xi(1,1)*Xi(6,2) + PdF3233*Xi(1,2)*Xi(6,1) + PdF3333*Xi(1,2)*Xi(6,2);
  K(5,21)=Xi(1,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(1,1)*Xi(7,0) + PdF1133*Xi(1,2)*Xi(7,0) + PdF1232*Xi(1,1)*Xi(7,1) + PdF1233*Xi(1,2)*Xi(7,1) + PdF1331*Xi(1,0)*Xi(7,2) + PdF1332*Xi(1,1)*Xi(7,2) + PdF1333*Xi(1,2)*Xi(7,2);
  K(5,22)=Xi(1,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(1,1)*Xi(7,0) + PdF2133*Xi(1,2)*Xi(7,0) + PdF2232*Xi(1,1)*Xi(7,1) + PdF2233*Xi(1,2)*Xi(7,1) + PdF2331*Xi(1,0)*Xi(7,2) + PdF2332*Xi(1,1)*Xi(7,2) + PdF2333*Xi(1,2)*Xi(7,2);
  K(5,23)=Xi(1,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(1,1)*Xi(7,0) + PdF3133*Xi(1,0)*Xi(7,2) + PdF3133*Xi(1,2)*Xi(7,0) + PdF3232*Xi(1,1)*Xi(7,1) + PdF3233*Xi(1,1)*Xi(7,2) + PdF3233*Xi(1,2)*Xi(7,1) + PdF3333*Xi(1,2)*Xi(7,2);
  K(6,6)=PdF1111*pow(Xi(2,0),2) + 2*PdF1112*Xi(2,0)*Xi(2,1) + 2*PdF1113*Xi(2,0)*Xi(2,2) + PdF1212*pow(Xi(2,1),2) + 2*PdF1213*Xi(2,1)*Xi(2,2) + PdF1313*pow(Xi(2,2),2);
  K(6,7)=Xi(2,0)*(PdF1121*Xi(2,0) + PdF1122*Xi(2,1) + PdF1123*Xi(2,2)) + PdF1222*pow(Xi(2,1),2) + PdF1323*pow(Xi(2,2),2) + PdF1221*Xi(2,0)*Xi(2,1) + PdF1223*Xi(2,1)*Xi(2,2) + PdF1321*Xi(2,0)*Xi(2,2) + PdF1322*Xi(2,1)*Xi(2,2);
  K(6,8)=Xi(2,0)*(PdF1131*Xi(2,0) + PdF1132*Xi(2,1) + PdF1133*Xi(2,2)) + PdF1232*pow(Xi(2,1),2) + PdF1333*pow(Xi(2,2),2) + PdF1231*Xi(2,0)*Xi(2,1) + PdF1233*Xi(2,1)*Xi(2,2) + PdF1331*Xi(2,0)*Xi(2,2) + PdF1332*Xi(2,1)*Xi(2,2);
  K(6,9)=Xi(2,0)*(PdF1111*Xi(3,0) + PdF1112*Xi(3,1)) + PdF1112*Xi(2,1)*Xi(3,0) + PdF1113*Xi(2,0)*Xi(3,2) + PdF1113*Xi(3,0)*Xi(2,2) + PdF1212*Xi(2,1)*Xi(3,1) + PdF1213*Xi(2,1)*Xi(3,2) + PdF1213*Xi(2,2)*Xi(3,1) + PdF1313*Xi(2,2)*Xi(3,2);
  K(6,10)=Xi(2,0)*(PdF1121*Xi(3,0) + PdF1122*Xi(3,1)) + PdF1123*Xi(2,0)*Xi(3,2) + PdF1221*Xi(2,1)*Xi(3,0) + PdF1222*Xi(2,1)*Xi(3,1) + PdF1223*Xi(2,1)*Xi(3,2) + PdF1321*Xi(3,0)*Xi(2,2) + PdF1322*Xi(2,2)*Xi(3,1) + PdF1323*Xi(2,2)*Xi(3,2);
  K(6,11)=Xi(2,0)*(PdF1131*Xi(3,0) + PdF1132*Xi(3,1)) + PdF1133*Xi(2,0)*Xi(3,2) + PdF1231*Xi(2,1)*Xi(3,0) + PdF1232*Xi(2,1)*Xi(3,1) + PdF1233*Xi(2,1)*Xi(3,2) + PdF1331*Xi(3,0)*Xi(2,2) + PdF1332*Xi(2,2)*Xi(3,1) + PdF1333*Xi(2,2)*Xi(3,2);
  K(6,12)=Xi(2,0)*(PdF1111*Xi(4,0) + PdF1112*Xi(4,1)) + PdF1112*Xi(2,1)*Xi(4,0) + PdF1113*Xi(2,0)*Xi(4,2) + PdF1113*Xi(2,2)*Xi(4,0) + PdF1212*Xi(2,1)*Xi(4,1) + PdF1213*Xi(2,1)*Xi(4,2) + PdF1213*Xi(2,2)*Xi(4,1) + PdF1313*Xi(2,2)*Xi(4,2);
  K(6,13)=Xi(2,0)*(PdF1121*Xi(4,0) + PdF1122*Xi(4,1)) + PdF1123*Xi(2,0)*Xi(4,2) + PdF1221*Xi(2,1)*Xi(4,0) + PdF1222*Xi(2,1)*Xi(4,1) + PdF1223*Xi(2,1)*Xi(4,2) + PdF1321*Xi(2,2)*Xi(4,0) + PdF1322*Xi(2,2)*Xi(4,1) + PdF1323*Xi(2,2)*Xi(4,2);
  K(6,14)=Xi(2,0)*(PdF1131*Xi(4,0) + PdF1132*Xi(4,1)) + PdF1133*Xi(2,0)*Xi(4,2) + PdF1231*Xi(2,1)*Xi(4,0) + PdF1232*Xi(2,1)*Xi(4,1) + PdF1233*Xi(2,1)*Xi(4,2) + PdF1331*Xi(2,2)*Xi(4,0) + PdF1332*Xi(2,2)*Xi(4,1) + PdF1333*Xi(2,2)*Xi(4,2);
  K(6,15)=Xi(2,0)*(PdF1111*Xi(5,0) + PdF1112*Xi(5,1)) + PdF1112*Xi(2,1)*Xi(5,0) + PdF1113*Xi(2,0)*Xi(5,2) + PdF1113*Xi(2,2)*Xi(5,0) + PdF1212*Xi(2,1)*Xi(5,1) + PdF1213*Xi(2,1)*Xi(5,2) + PdF1213*Xi(2,2)*Xi(5,1) + PdF1313*Xi(2,2)*Xi(5,2);
  K(6,16)=Xi(2,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1)) + PdF1123*Xi(2,0)*Xi(5,2) + PdF1221*Xi(2,1)*Xi(5,0) + PdF1222*Xi(2,1)*Xi(5,1) + PdF1223*Xi(2,1)*Xi(5,2) + PdF1321*Xi(2,2)*Xi(5,0) + PdF1322*Xi(2,2)*Xi(5,1) + PdF1323*Xi(2,2)*Xi(5,2);
  K(6,17)=Xi(2,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1)) + PdF1133*Xi(2,0)*Xi(5,2) + PdF1231*Xi(2,1)*Xi(5,0) + PdF1232*Xi(2,1)*Xi(5,1) + PdF1233*Xi(2,1)*Xi(5,2) + PdF1331*Xi(2,2)*Xi(5,0) + PdF1332*Xi(2,2)*Xi(5,1) + PdF1333*Xi(2,2)*Xi(5,2);
  K(6,18)=Xi(2,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(2,1)*Xi(6,0) + PdF1113*Xi(2,0)*Xi(6,2) + PdF1113*Xi(2,2)*Xi(6,0) + PdF1212*Xi(2,1)*Xi(6,1) + PdF1213*Xi(2,1)*Xi(6,2) + PdF1213*Xi(2,2)*Xi(6,1) + PdF1313*Xi(2,2)*Xi(6,2);
  K(6,19)=Xi(2,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(2,0)*Xi(6,2) + PdF1221*Xi(2,1)*Xi(6,0) + PdF1222*Xi(2,1)*Xi(6,1) + PdF1223*Xi(2,1)*Xi(6,2) + PdF1321*Xi(2,2)*Xi(6,0) + PdF1322*Xi(2,2)*Xi(6,1) + PdF1323*Xi(2,2)*Xi(6,2);
  K(6,20)=Xi(2,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(2,0)*Xi(6,2) + PdF1231*Xi(2,1)*Xi(6,0) + PdF1232*Xi(2,1)*Xi(6,1) + PdF1233*Xi(2,1)*Xi(6,2) + PdF1331*Xi(2,2)*Xi(6,0) + PdF1332*Xi(2,2)*Xi(6,1) + PdF1333*Xi(2,2)*Xi(6,2);
  K(6,21)=Xi(2,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(2,1)*Xi(7,0) + PdF1113*Xi(2,0)*Xi(7,2) + PdF1113*Xi(2,2)*Xi(7,0) + PdF1212*Xi(2,1)*Xi(7,1) + PdF1213*Xi(2,1)*Xi(7,2) + PdF1213*Xi(2,2)*Xi(7,1) + PdF1313*Xi(2,2)*Xi(7,2);
  K(6,22)=Xi(2,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(2,0)*Xi(7,2) + PdF1221*Xi(2,1)*Xi(7,0) + PdF1222*Xi(2,1)*Xi(7,1) + PdF1223*Xi(2,1)*Xi(7,2) + PdF1321*Xi(2,2)*Xi(7,0) + PdF1322*Xi(2,2)*Xi(7,1) + PdF1323*Xi(2,2)*Xi(7,2);
  K(6,23)=Xi(2,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(2,0)*Xi(7,2) + PdF1231*Xi(2,1)*Xi(7,0) + PdF1232*Xi(2,1)*Xi(7,1) + PdF1233*Xi(2,1)*Xi(7,2) + PdF1331*Xi(2,2)*Xi(7,0) + PdF1332*Xi(2,2)*Xi(7,1) + PdF1333*Xi(2,2)*Xi(7,2);
  K(7,7)=PdF2121*pow(Xi(2,0),2) + 2*PdF2122*Xi(2,0)*Xi(2,1) + 2*PdF2123*Xi(2,0)*Xi(2,2) + PdF2222*pow(Xi(2,1),2) + 2*PdF2223*Xi(2,1)*Xi(2,2) + PdF2323*pow(Xi(2,2),2);
  K(7,8)=Xi(2,0)*(PdF2131*Xi(2,0) + PdF2132*Xi(2,1) + PdF2133*Xi(2,2)) + PdF2232*pow(Xi(2,1),2) + PdF2333*pow(Xi(2,2),2) + PdF2231*Xi(2,0)*Xi(2,1) + PdF2233*Xi(2,1)*Xi(2,2) + PdF2331*Xi(2,0)*Xi(2,2) + PdF2332*Xi(2,1)*Xi(2,2);
  K(7,9)=Xi(2,0)*(PdF1121*Xi(3,0) + PdF1221*Xi(3,1)) + PdF1122*Xi(2,1)*Xi(3,0) + PdF1123*Xi(3,0)*Xi(2,2) + PdF1222*Xi(2,1)*Xi(3,1) + PdF1223*Xi(2,2)*Xi(3,1) + PdF1321*Xi(2,0)*Xi(3,2) + PdF1322*Xi(2,1)*Xi(3,2) + PdF1323*Xi(2,2)*Xi(3,2);
  K(7,10)=Xi(2,0)*(PdF2121*Xi(3,0) + PdF2122*Xi(3,1)) + PdF2122*Xi(2,1)*Xi(3,0) + PdF2123*Xi(2,0)*Xi(3,2) + PdF2123*Xi(3,0)*Xi(2,2) + PdF2222*Xi(2,1)*Xi(3,1) + PdF2223*Xi(2,1)*Xi(3,2) + PdF2223*Xi(2,2)*Xi(3,1) + PdF2323*Xi(2,2)*Xi(3,2);
  K(7,11)=Xi(2,0)*(PdF2131*Xi(3,0) + PdF2132*Xi(3,1)) + PdF2133*Xi(2,0)*Xi(3,2) + PdF2231*Xi(2,1)*Xi(3,0) + PdF2232*Xi(2,1)*Xi(3,1) + PdF2233*Xi(2,1)*Xi(3,2) + PdF2331*Xi(3,0)*Xi(2,2) + PdF2332*Xi(2,2)*Xi(3,1) + PdF2333*Xi(2,2)*Xi(3,2);
  K(7,12)=Xi(2,0)*(PdF1121*Xi(4,0) + PdF1221*Xi(4,1)) + PdF1122*Xi(2,1)*Xi(4,0) + PdF1123*Xi(2,2)*Xi(4,0) + PdF1222*Xi(2,1)*Xi(4,1) + PdF1223*Xi(2,2)*Xi(4,1) + PdF1321*Xi(2,0)*Xi(4,2) + PdF1322*Xi(2,1)*Xi(4,2) + PdF1323*Xi(2,2)*Xi(4,2);
  K(7,13)=Xi(2,0)*(PdF2121*Xi(4,0) + PdF2122*Xi(4,1)) + PdF2122*Xi(2,1)*Xi(4,0) + PdF2123*Xi(2,0)*Xi(4,2) + PdF2123*Xi(2,2)*Xi(4,0) + PdF2222*Xi(2,1)*Xi(4,1) + PdF2223*Xi(2,1)*Xi(4,2) + PdF2223*Xi(2,2)*Xi(4,1) + PdF2323*Xi(2,2)*Xi(4,2);
  K(7,14)=Xi(2,0)*(PdF2131*Xi(4,0) + PdF2132*Xi(4,1)) + PdF2133*Xi(2,0)*Xi(4,2) + PdF2231*Xi(2,1)*Xi(4,0) + PdF2232*Xi(2,1)*Xi(4,1) + PdF2233*Xi(2,1)*Xi(4,2) + PdF2331*Xi(2,2)*Xi(4,0) + PdF2332*Xi(2,2)*Xi(4,1) + PdF2333*Xi(2,2)*Xi(4,2);
  K(7,15)=Xi(2,0)*(PdF1121*Xi(5,0) + PdF1221*Xi(5,1)) + PdF1122*Xi(2,1)*Xi(5,0) + PdF1123*Xi(2,2)*Xi(5,0) + PdF1222*Xi(2,1)*Xi(5,1) + PdF1223*Xi(2,2)*Xi(5,1) + PdF1321*Xi(2,0)*Xi(5,2) + PdF1322*Xi(2,1)*Xi(5,2) + PdF1323*Xi(2,2)*Xi(5,2);
  K(7,16)=Xi(2,0)*(PdF2121*Xi(5,0) + PdF2122*Xi(5,1)) + PdF2122*Xi(2,1)*Xi(5,0) + PdF2123*Xi(2,0)*Xi(5,2) + PdF2123*Xi(2,2)*Xi(5,0) + PdF2222*Xi(2,1)*Xi(5,1) + PdF2223*Xi(2,1)*Xi(5,2) + PdF2223*Xi(2,2)*Xi(5,1) + PdF2323*Xi(2,2)*Xi(5,2);
  K(7,17)=Xi(2,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1)) + PdF2133*Xi(2,0)*Xi(5,2) + PdF2231*Xi(2,1)*Xi(5,0) + PdF2232*Xi(2,1)*Xi(5,1) + PdF2233*Xi(2,1)*Xi(5,2) + PdF2331*Xi(2,2)*Xi(5,0) + PdF2332*Xi(2,2)*Xi(5,1) + PdF2333*Xi(2,2)*Xi(5,2);
  K(7,18)=Xi(2,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(2,1)*Xi(6,0) + PdF1123*Xi(2,2)*Xi(6,0) + PdF1222*Xi(2,1)*Xi(6,1) + PdF1223*Xi(2,2)*Xi(6,1) + PdF1321*Xi(2,0)*Xi(6,2) + PdF1322*Xi(2,1)*Xi(6,2) + PdF1323*Xi(2,2)*Xi(6,2);
  K(7,19)=Xi(2,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(2,1)*Xi(6,0) + PdF2123*Xi(2,0)*Xi(6,2) + PdF2123*Xi(2,2)*Xi(6,0) + PdF2222*Xi(2,1)*Xi(6,1) + PdF2223*Xi(2,1)*Xi(6,2) + PdF2223*Xi(2,2)*Xi(6,1) + PdF2323*Xi(2,2)*Xi(6,2);
  K(7,20)=Xi(2,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(2,0)*Xi(6,2) + PdF2231*Xi(2,1)*Xi(6,0) + PdF2232*Xi(2,1)*Xi(6,1) + PdF2233*Xi(2,1)*Xi(6,2) + PdF2331*Xi(2,2)*Xi(6,0) + PdF2332*Xi(2,2)*Xi(6,1) + PdF2333*Xi(2,2)*Xi(6,2);
  K(7,21)=Xi(2,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(2,1)*Xi(7,0) + PdF1123*Xi(2,2)*Xi(7,0) + PdF1222*Xi(2,1)*Xi(7,1) + PdF1223*Xi(2,2)*Xi(7,1) + PdF1321*Xi(2,0)*Xi(7,2) + PdF1322*Xi(2,1)*Xi(7,2) + PdF1323*Xi(2,2)*Xi(7,2);
  K(7,22)=Xi(2,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(2,1)*Xi(7,0) + PdF2123*Xi(2,0)*Xi(7,2) + PdF2123*Xi(2,2)*Xi(7,0) + PdF2222*Xi(2,1)*Xi(7,1) + PdF2223*Xi(2,1)*Xi(7,2) + PdF2223*Xi(2,2)*Xi(7,1) + PdF2323*Xi(2,2)*Xi(7,2);
  K(7,23)=Xi(2,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(2,0)*Xi(7,2) + PdF2231*Xi(2,1)*Xi(7,0) + PdF2232*Xi(2,1)*Xi(7,1) + PdF2233*Xi(2,1)*Xi(7,2) + PdF2331*Xi(2,2)*Xi(7,0) + PdF2332*Xi(2,2)*Xi(7,1) + PdF2333*Xi(2,2)*Xi(7,2);
  K(8,8)=PdF3131*pow(Xi(2,0),2) + 2*PdF3132*Xi(2,0)*Xi(2,1) + 2*PdF3133*Xi(2,0)*Xi(2,2) + PdF3232*pow(Xi(2,1),2) + 2*PdF3233*Xi(2,1)*Xi(2,2) + PdF3333*pow(Xi(2,2),2);
  K(8,9)=Xi(2,0)*(PdF1131*Xi(3,0) + PdF1231*Xi(3,1)) + PdF1132*Xi(2,1)*Xi(3,0) + PdF1133*Xi(3,0)*Xi(2,2) + PdF1232*Xi(2,1)*Xi(3,1) + PdF1233*Xi(2,2)*Xi(3,1) + PdF1331*Xi(2,0)*Xi(3,2) + PdF1332*Xi(2,1)*Xi(3,2) + PdF1333*Xi(2,2)*Xi(3,2);
  K(8,10)=Xi(2,0)*(PdF2131*Xi(3,0) + PdF2231*Xi(3,1)) + PdF2132*Xi(2,1)*Xi(3,0) + PdF2133*Xi(3,0)*Xi(2,2) + PdF2232*Xi(2,1)*Xi(3,1) + PdF2233*Xi(2,2)*Xi(3,1) + PdF2331*Xi(2,0)*Xi(3,2) + PdF2332*Xi(2,1)*Xi(3,2) + PdF2333*Xi(2,2)*Xi(3,2);
  K(8,11)=Xi(2,0)*(PdF3131*Xi(3,0) + PdF3132*Xi(3,1)) + PdF3132*Xi(2,1)*Xi(3,0) + PdF3133*Xi(2,0)*Xi(3,2) + PdF3133*Xi(3,0)*Xi(2,2) + PdF3232*Xi(2,1)*Xi(3,1) + PdF3233*Xi(2,1)*Xi(3,2) + PdF3233*Xi(2,2)*Xi(3,1) + PdF3333*Xi(2,2)*Xi(3,2);
  K(8,12)=Xi(2,0)*(PdF1131*Xi(4,0) + PdF1231*Xi(4,1)) + PdF1132*Xi(2,1)*Xi(4,0) + PdF1133*Xi(2,2)*Xi(4,0) + PdF1232*Xi(2,1)*Xi(4,1) + PdF1233*Xi(2,2)*Xi(4,1) + PdF1331*Xi(2,0)*Xi(4,2) + PdF1332*Xi(2,1)*Xi(4,2) + PdF1333*Xi(2,2)*Xi(4,2);
  K(8,13)=Xi(2,0)*(PdF2131*Xi(4,0) + PdF2231*Xi(4,1)) + PdF2132*Xi(2,1)*Xi(4,0) + PdF2133*Xi(2,2)*Xi(4,0) + PdF2232*Xi(2,1)*Xi(4,1) + PdF2233*Xi(2,2)*Xi(4,1) + PdF2331*Xi(2,0)*Xi(4,2) + PdF2332*Xi(2,1)*Xi(4,2) + PdF2333*Xi(2,2)*Xi(4,2);
  K(8,14)=Xi(2,0)*(PdF3131*Xi(4,0) + PdF3132*Xi(4,1)) + PdF3132*Xi(2,1)*Xi(4,0) + PdF3133*Xi(2,0)*Xi(4,2) + PdF3133*Xi(2,2)*Xi(4,0) + PdF3232*Xi(2,1)*Xi(4,1) + PdF3233*Xi(2,1)*Xi(4,2) + PdF3233*Xi(2,2)*Xi(4,1) + PdF3333*Xi(2,2)*Xi(4,2);
  K(8,15)=Xi(2,0)*(PdF1131*Xi(5,0) + PdF1231*Xi(5,1)) + PdF1132*Xi(2,1)*Xi(5,0) + PdF1133*Xi(2,2)*Xi(5,0) + PdF1232*Xi(2,1)*Xi(5,1) + PdF1233*Xi(2,2)*Xi(5,1) + PdF1331*Xi(2,0)*Xi(5,2) + PdF1332*Xi(2,1)*Xi(5,2) + PdF1333*Xi(2,2)*Xi(5,2);
  K(8,16)=Xi(2,0)*(PdF2131*Xi(5,0) + PdF2231*Xi(5,1)) + PdF2132*Xi(2,1)*Xi(5,0) + PdF2133*Xi(2,2)*Xi(5,0) + PdF2232*Xi(2,1)*Xi(5,1) + PdF2233*Xi(2,2)*Xi(5,1) + PdF2331*Xi(2,0)*Xi(5,2) + PdF2332*Xi(2,1)*Xi(5,2) + PdF2333*Xi(2,2)*Xi(5,2);
  K(8,17)=Xi(2,0)*(PdF3131*Xi(5,0) + PdF3132*Xi(5,1)) + PdF3132*Xi(2,1)*Xi(5,0) + PdF3133*Xi(2,0)*Xi(5,2) + PdF3133*Xi(2,2)*Xi(5,0) + PdF3232*Xi(2,1)*Xi(5,1) + PdF3233*Xi(2,1)*Xi(5,2) + PdF3233*Xi(2,2)*Xi(5,1) + PdF3333*Xi(2,2)*Xi(5,2);
  K(8,18)=Xi(2,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(2,1)*Xi(6,0) + PdF1133*Xi(2,2)*Xi(6,0) + PdF1232*Xi(2,1)*Xi(6,1) + PdF1233*Xi(2,2)*Xi(6,1) + PdF1331*Xi(2,0)*Xi(6,2) + PdF1332*Xi(2,1)*Xi(6,2) + PdF1333*Xi(2,2)*Xi(6,2);
  K(8,19)=Xi(2,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(2,1)*Xi(6,0) + PdF2133*Xi(2,2)*Xi(6,0) + PdF2232*Xi(2,1)*Xi(6,1) + PdF2233*Xi(2,2)*Xi(6,1) + PdF2331*Xi(2,0)*Xi(6,2) + PdF2332*Xi(2,1)*Xi(6,2) + PdF2333*Xi(2,2)*Xi(6,2);
  K(8,20)=Xi(2,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(2,1)*Xi(6,0) + PdF3133*Xi(2,0)*Xi(6,2) + PdF3133*Xi(2,2)*Xi(6,0) + PdF3232*Xi(2,1)*Xi(6,1) + PdF3233*Xi(2,1)*Xi(6,2) + PdF3233*Xi(2,2)*Xi(6,1) + PdF3333*Xi(2,2)*Xi(6,2);
  K(8,21)=Xi(2,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(2,1)*Xi(7,0) + PdF1133*Xi(2,2)*Xi(7,0) + PdF1232*Xi(2,1)*Xi(7,1) + PdF1233*Xi(2,2)*Xi(7,1) + PdF1331*Xi(2,0)*Xi(7,2) + PdF1332*Xi(2,1)*Xi(7,2) + PdF1333*Xi(2,2)*Xi(7,2);
  K(8,22)=Xi(2,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(2,1)*Xi(7,0) + PdF2133*Xi(2,2)*Xi(7,0) + PdF2232*Xi(2,1)*Xi(7,1) + PdF2233*Xi(2,2)*Xi(7,1) + PdF2331*Xi(2,0)*Xi(7,2) + PdF2332*Xi(2,1)*Xi(7,2) + PdF2333*Xi(2,2)*Xi(7,2);
  K(8,23)=Xi(2,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(2,1)*Xi(7,0) + PdF3133*Xi(2,0)*Xi(7,2) + PdF3133*Xi(2,2)*Xi(7,0) + PdF3232*Xi(2,1)*Xi(7,1) + PdF3233*Xi(2,1)*Xi(7,2) + PdF3233*Xi(2,2)*Xi(7,1) + PdF3333*Xi(2,2)*Xi(7,2);
  K(9,9)=PdF1111*pow(Xi(3,0),2) + 2*PdF1112*Xi(3,0)*Xi(3,1) + 2*PdF1113*Xi(3,0)*Xi(3,2) + PdF1212*pow(Xi(3,1),2) + 2*PdF1213*Xi(3,1)*Xi(3,2) + PdF1313*pow(Xi(3,2),2);
  K(9,10)=Xi(3,0)*(PdF1121*Xi(3,0) + PdF1122*Xi(3,1) + PdF1123*Xi(3,2)) + PdF1222*pow(Xi(3,1),2) + PdF1323*pow(Xi(3,2),2) + PdF1221*Xi(3,0)*Xi(3,1) + PdF1223*Xi(3,1)*Xi(3,2) + PdF1321*Xi(3,0)*Xi(3,2) + PdF1322*Xi(3,1)*Xi(3,2);
  K(9,11)=Xi(3,0)*(PdF1131*Xi(3,0) + PdF1132*Xi(3,1) + PdF1133*Xi(3,2)) + PdF1232*pow(Xi(3,1),2) + PdF1333*pow(Xi(3,2),2) + PdF1231*Xi(3,0)*Xi(3,1) + PdF1233*Xi(3,1)*Xi(3,2) + PdF1331*Xi(3,0)*Xi(3,2) + PdF1332*Xi(3,1)*Xi(3,2);
  K(9,12)=Xi(3,0)*(PdF1111*Xi(4,0) + PdF1112*Xi(4,1)) + PdF1112*Xi(3,1)*Xi(4,0) + PdF1113*Xi(3,0)*Xi(4,2) + PdF1113*Xi(4,0)*Xi(3,2) + PdF1212*Xi(3,1)*Xi(4,1) + PdF1213*Xi(3,1)*Xi(4,2) + PdF1213*Xi(3,2)*Xi(4,1) + PdF1313*Xi(3,2)*Xi(4,2);
  K(9,13)=Xi(3,0)*(PdF1121*Xi(4,0) + PdF1122*Xi(4,1)) + PdF1123*Xi(3,0)*Xi(4,2) + PdF1221*Xi(3,1)*Xi(4,0) + PdF1222*Xi(3,1)*Xi(4,1) + PdF1223*Xi(3,1)*Xi(4,2) + PdF1321*Xi(4,0)*Xi(3,2) + PdF1322*Xi(3,2)*Xi(4,1) + PdF1323*Xi(3,2)*Xi(4,2);
  K(9,14)=Xi(3,0)*(PdF1131*Xi(4,0) + PdF1132*Xi(4,1)) + PdF1133*Xi(3,0)*Xi(4,2) + PdF1231*Xi(3,1)*Xi(4,0) + PdF1232*Xi(3,1)*Xi(4,1) + PdF1233*Xi(3,1)*Xi(4,2) + PdF1331*Xi(4,0)*Xi(3,2) + PdF1332*Xi(3,2)*Xi(4,1) + PdF1333*Xi(3,2)*Xi(4,2);
  K(9,15)=Xi(3,0)*(PdF1111*Xi(5,0) + PdF1112*Xi(5,1)) + PdF1112*Xi(3,1)*Xi(5,0) + PdF1113*Xi(3,0)*Xi(5,2) + PdF1113*Xi(3,2)*Xi(5,0) + PdF1212*Xi(3,1)*Xi(5,1) + PdF1213*Xi(3,1)*Xi(5,2) + PdF1213*Xi(3,2)*Xi(5,1) + PdF1313*Xi(3,2)*Xi(5,2);
  K(9,16)=Xi(3,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1)) + PdF1123*Xi(3,0)*Xi(5,2) + PdF1221*Xi(3,1)*Xi(5,0) + PdF1222*Xi(3,1)*Xi(5,1) + PdF1223*Xi(3,1)*Xi(5,2) + PdF1321*Xi(3,2)*Xi(5,0) + PdF1322*Xi(3,2)*Xi(5,1) + PdF1323*Xi(3,2)*Xi(5,2);
  K(9,17)=Xi(3,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1)) + PdF1133*Xi(3,0)*Xi(5,2) + PdF1231*Xi(3,1)*Xi(5,0) + PdF1232*Xi(3,1)*Xi(5,1) + PdF1233*Xi(3,1)*Xi(5,2) + PdF1331*Xi(3,2)*Xi(5,0) + PdF1332*Xi(3,2)*Xi(5,1) + PdF1333*Xi(3,2)*Xi(5,2);
  K(9,18)=Xi(3,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(3,1)*Xi(6,0) + PdF1113*Xi(3,0)*Xi(6,2) + PdF1113*Xi(3,2)*Xi(6,0) + PdF1212*Xi(3,1)*Xi(6,1) + PdF1213*Xi(3,1)*Xi(6,2) + PdF1213*Xi(3,2)*Xi(6,1) + PdF1313*Xi(3,2)*Xi(6,2);
  K(9,19)=Xi(3,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(3,0)*Xi(6,2) + PdF1221*Xi(3,1)*Xi(6,0) + PdF1222*Xi(3,1)*Xi(6,1) + PdF1223*Xi(3,1)*Xi(6,2) + PdF1321*Xi(3,2)*Xi(6,0) + PdF1322*Xi(3,2)*Xi(6,1) + PdF1323*Xi(3,2)*Xi(6,2);
  K(9,20)=Xi(3,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(3,0)*Xi(6,2) + PdF1231*Xi(3,1)*Xi(6,0) + PdF1232*Xi(3,1)*Xi(6,1) + PdF1233*Xi(3,1)*Xi(6,2) + PdF1331*Xi(3,2)*Xi(6,0) + PdF1332*Xi(3,2)*Xi(6,1) + PdF1333*Xi(3,2)*Xi(6,2);
  K(9,21)=Xi(3,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(3,1)*Xi(7,0) + PdF1113*Xi(3,0)*Xi(7,2) + PdF1113*Xi(3,2)*Xi(7,0) + PdF1212*Xi(3,1)*Xi(7,1) + PdF1213*Xi(3,1)*Xi(7,2) + PdF1213*Xi(3,2)*Xi(7,1) + PdF1313*Xi(3,2)*Xi(7,2);
  K(9,22)=Xi(3,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(3,0)*Xi(7,2) + PdF1221*Xi(3,1)*Xi(7,0) + PdF1222*Xi(3,1)*Xi(7,1) + PdF1223*Xi(3,1)*Xi(7,2) + PdF1321*Xi(3,2)*Xi(7,0) + PdF1322*Xi(3,2)*Xi(7,1) + PdF1323*Xi(3,2)*Xi(7,2);
  K(9,23)=Xi(3,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(3,0)*Xi(7,2) + PdF1231*Xi(3,1)*Xi(7,0) + PdF1232*Xi(3,1)*Xi(7,1) + PdF1233*Xi(3,1)*Xi(7,2) + PdF1331*Xi(3,2)*Xi(7,0) + PdF1332*Xi(3,2)*Xi(7,1) + PdF1333*Xi(3,2)*Xi(7,2);
  K(10,10)=PdF2121*pow(Xi(3,0),2) + 2*PdF2122*Xi(3,0)*Xi(3,1) + 2*PdF2123*Xi(3,0)*Xi(3,2) + PdF2222*pow(Xi(3,1),2) + 2*PdF2223*Xi(3,1)*Xi(3,2) + PdF2323*pow(Xi(3,2),2);
  K(10,11)=Xi(3,0)*(PdF2131*Xi(3,0) + PdF2132*Xi(3,1) + PdF2133*Xi(3,2)) + PdF2232*pow(Xi(3,1),2) + PdF2333*pow(Xi(3,2),2) + PdF2231*Xi(3,0)*Xi(3,1) + PdF2233*Xi(3,1)*Xi(3,2) + PdF2331*Xi(3,0)*Xi(3,2) + PdF2332*Xi(3,1)*Xi(3,2);
  K(10,12)=Xi(3,0)*(PdF1121*Xi(4,0) + PdF1221*Xi(4,1)) + PdF1122*Xi(3,1)*Xi(4,0) + PdF1123*Xi(4,0)*Xi(3,2) + PdF1222*Xi(3,1)*Xi(4,1) + PdF1223*Xi(3,2)*Xi(4,1) + PdF1321*Xi(3,0)*Xi(4,2) + PdF1322*Xi(3,1)*Xi(4,2) + PdF1323*Xi(3,2)*Xi(4,2);
  K(10,13)=Xi(3,0)*(PdF2121*Xi(4,0) + PdF2122*Xi(4,1)) + PdF2122*Xi(3,1)*Xi(4,0) + PdF2123*Xi(3,0)*Xi(4,2) + PdF2123*Xi(4,0)*Xi(3,2) + PdF2222*Xi(3,1)*Xi(4,1) + PdF2223*Xi(3,1)*Xi(4,2) + PdF2223*Xi(3,2)*Xi(4,1) + PdF2323*Xi(3,2)*Xi(4,2);
  K(10,14)=Xi(3,0)*(PdF2131*Xi(4,0) + PdF2132*Xi(4,1)) + PdF2133*Xi(3,0)*Xi(4,2) + PdF2231*Xi(3,1)*Xi(4,0) + PdF2232*Xi(3,1)*Xi(4,1) + PdF2233*Xi(3,1)*Xi(4,2) + PdF2331*Xi(4,0)*Xi(3,2) + PdF2332*Xi(3,2)*Xi(4,1) + PdF2333*Xi(3,2)*Xi(4,2);
  K(10,15)=Xi(3,0)*(PdF1121*Xi(5,0) + PdF1221*Xi(5,1)) + PdF1122*Xi(3,1)*Xi(5,0) + PdF1123*Xi(3,2)*Xi(5,0) + PdF1222*Xi(3,1)*Xi(5,1) + PdF1223*Xi(3,2)*Xi(5,1) + PdF1321*Xi(3,0)*Xi(5,2) + PdF1322*Xi(3,1)*Xi(5,2) + PdF1323*Xi(3,2)*Xi(5,2);
  K(10,16)=Xi(3,0)*(PdF2121*Xi(5,0) + PdF2122*Xi(5,1)) + PdF2122*Xi(3,1)*Xi(5,0) + PdF2123*Xi(3,0)*Xi(5,2) + PdF2123*Xi(3,2)*Xi(5,0) + PdF2222*Xi(3,1)*Xi(5,1) + PdF2223*Xi(3,1)*Xi(5,2) + PdF2223*Xi(3,2)*Xi(5,1) + PdF2323*Xi(3,2)*Xi(5,2);
  K(10,17)=Xi(3,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1)) + PdF2133*Xi(3,0)*Xi(5,2) + PdF2231*Xi(3,1)*Xi(5,0) + PdF2232*Xi(3,1)*Xi(5,1) + PdF2233*Xi(3,1)*Xi(5,2) + PdF2331*Xi(3,2)*Xi(5,0) + PdF2332*Xi(3,2)*Xi(5,1) + PdF2333*Xi(3,2)*Xi(5,2);
  K(10,18)=Xi(3,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(3,1)*Xi(6,0) + PdF1123*Xi(3,2)*Xi(6,0) + PdF1222*Xi(3,1)*Xi(6,1) + PdF1223*Xi(3,2)*Xi(6,1) + PdF1321*Xi(3,0)*Xi(6,2) + PdF1322*Xi(3,1)*Xi(6,2) + PdF1323*Xi(3,2)*Xi(6,2);
  K(10,19)=Xi(3,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(3,1)*Xi(6,0) + PdF2123*Xi(3,0)*Xi(6,2) + PdF2123*Xi(3,2)*Xi(6,0) + PdF2222*Xi(3,1)*Xi(6,1) + PdF2223*Xi(3,1)*Xi(6,2) + PdF2223*Xi(3,2)*Xi(6,1) + PdF2323*Xi(3,2)*Xi(6,2);
  K(10,20)=Xi(3,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(3,0)*Xi(6,2) + PdF2231*Xi(3,1)*Xi(6,0) + PdF2232*Xi(3,1)*Xi(6,1) + PdF2233*Xi(3,1)*Xi(6,2) + PdF2331*Xi(3,2)*Xi(6,0) + PdF2332*Xi(3,2)*Xi(6,1) + PdF2333*Xi(3,2)*Xi(6,2);
  K(10,21)=Xi(3,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(3,1)*Xi(7,0) + PdF1123*Xi(3,2)*Xi(7,0) + PdF1222*Xi(3,1)*Xi(7,1) + PdF1223*Xi(3,2)*Xi(7,1) + PdF1321*Xi(3,0)*Xi(7,2) + PdF1322*Xi(3,1)*Xi(7,2) + PdF1323*Xi(3,2)*Xi(7,2);
  K(10,22)=Xi(3,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(3,1)*Xi(7,0) + PdF2123*Xi(3,0)*Xi(7,2) + PdF2123*Xi(3,2)*Xi(7,0) + PdF2222*Xi(3,1)*Xi(7,1) + PdF2223*Xi(3,1)*Xi(7,2) + PdF2223*Xi(3,2)*Xi(7,1) + PdF2323*Xi(3,2)*Xi(7,2);
  K(10,23)=Xi(3,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(3,0)*Xi(7,2) + PdF2231*Xi(3,1)*Xi(7,0) + PdF2232*Xi(3,1)*Xi(7,1) + PdF2233*Xi(3,1)*Xi(7,2) + PdF2331*Xi(3,2)*Xi(7,0) + PdF2332*Xi(3,2)*Xi(7,1) + PdF2333*Xi(3,2)*Xi(7,2);
  K(11,11)=PdF3131*pow(Xi(3,0),2) + 2*PdF3132*Xi(3,0)*Xi(3,1) + 2*PdF3133*Xi(3,0)*Xi(3,2) + PdF3232*pow(Xi(3,1),2) + 2*PdF3233*Xi(3,1)*Xi(3,2) + PdF3333*pow(Xi(3,2),2);
  K(11,12)=Xi(3,0)*(PdF1131*Xi(4,0) + PdF1231*Xi(4,1)) + PdF1132*Xi(3,1)*Xi(4,0) + PdF1133*Xi(4,0)*Xi(3,2) + PdF1232*Xi(3,1)*Xi(4,1) + PdF1233*Xi(3,2)*Xi(4,1) + PdF1331*Xi(3,0)*Xi(4,2) + PdF1332*Xi(3,1)*Xi(4,2) + PdF1333*Xi(3,2)*Xi(4,2);
  K(11,13)=Xi(3,0)*(PdF2131*Xi(4,0) + PdF2231*Xi(4,1)) + PdF2132*Xi(3,1)*Xi(4,0) + PdF2133*Xi(4,0)*Xi(3,2) + PdF2232*Xi(3,1)*Xi(4,1) + PdF2233*Xi(3,2)*Xi(4,1) + PdF2331*Xi(3,0)*Xi(4,2) + PdF2332*Xi(3,1)*Xi(4,2) + PdF2333*Xi(3,2)*Xi(4,2);
  K(11,14)=Xi(3,0)*(PdF3131*Xi(4,0) + PdF3132*Xi(4,1)) + PdF3132*Xi(3,1)*Xi(4,0) + PdF3133*Xi(3,0)*Xi(4,2) + PdF3133*Xi(4,0)*Xi(3,2) + PdF3232*Xi(3,1)*Xi(4,1) + PdF3233*Xi(3,1)*Xi(4,2) + PdF3233*Xi(3,2)*Xi(4,1) + PdF3333*Xi(3,2)*Xi(4,2);
  K(11,15)=Xi(3,0)*(PdF1131*Xi(5,0) + PdF1231*Xi(5,1)) + PdF1132*Xi(3,1)*Xi(5,0) + PdF1133*Xi(3,2)*Xi(5,0) + PdF1232*Xi(3,1)*Xi(5,1) + PdF1233*Xi(3,2)*Xi(5,1) + PdF1331*Xi(3,0)*Xi(5,2) + PdF1332*Xi(3,1)*Xi(5,2) + PdF1333*Xi(3,2)*Xi(5,2);
  K(11,16)=Xi(3,0)*(PdF2131*Xi(5,0) + PdF2231*Xi(5,1)) + PdF2132*Xi(3,1)*Xi(5,0) + PdF2133*Xi(3,2)*Xi(5,0) + PdF2232*Xi(3,1)*Xi(5,1) + PdF2233*Xi(3,2)*Xi(5,1) + PdF2331*Xi(3,0)*Xi(5,2) + PdF2332*Xi(3,1)*Xi(5,2) + PdF2333*Xi(3,2)*Xi(5,2);
  K(11,17)=Xi(3,0)*(PdF3131*Xi(5,0) + PdF3132*Xi(5,1)) + PdF3132*Xi(3,1)*Xi(5,0) + PdF3133*Xi(3,0)*Xi(5,2) + PdF3133*Xi(3,2)*Xi(5,0) + PdF3232*Xi(3,1)*Xi(5,1) + PdF3233*Xi(3,1)*Xi(5,2) + PdF3233*Xi(3,2)*Xi(5,1) + PdF3333*Xi(3,2)*Xi(5,2);
  K(11,18)=Xi(3,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(3,1)*Xi(6,0) + PdF1133*Xi(3,2)*Xi(6,0) + PdF1232*Xi(3,1)*Xi(6,1) + PdF1233*Xi(3,2)*Xi(6,1) + PdF1331*Xi(3,0)*Xi(6,2) + PdF1332*Xi(3,1)*Xi(6,2) + PdF1333*Xi(3,2)*Xi(6,2);
  K(11,19)=Xi(3,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(3,1)*Xi(6,0) + PdF2133*Xi(3,2)*Xi(6,0) + PdF2232*Xi(3,1)*Xi(6,1) + PdF2233*Xi(3,2)*Xi(6,1) + PdF2331*Xi(3,0)*Xi(6,2) + PdF2332*Xi(3,1)*Xi(6,2) + PdF2333*Xi(3,2)*Xi(6,2);
  K(11,20)=Xi(3,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(3,1)*Xi(6,0) + PdF3133*Xi(3,0)*Xi(6,2) + PdF3133*Xi(3,2)*Xi(6,0) + PdF3232*Xi(3,1)*Xi(6,1) + PdF3233*Xi(3,1)*Xi(6,2) + PdF3233*Xi(3,2)*Xi(6,1) + PdF3333*Xi(3,2)*Xi(6,2);
  K(11,21)=Xi(3,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(3,1)*Xi(7,0) + PdF1133*Xi(3,2)*Xi(7,0) + PdF1232*Xi(3,1)*Xi(7,1) + PdF1233*Xi(3,2)*Xi(7,1) + PdF1331*Xi(3,0)*Xi(7,2) + PdF1332*Xi(3,1)*Xi(7,2) + PdF1333*Xi(3,2)*Xi(7,2);
  K(11,22)=Xi(3,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(3,1)*Xi(7,0) + PdF2133*Xi(3,2)*Xi(7,0) + PdF2232*Xi(3,1)*Xi(7,1) + PdF2233*Xi(3,2)*Xi(7,1) + PdF2331*Xi(3,0)*Xi(7,2) + PdF2332*Xi(3,1)*Xi(7,2) + PdF2333*Xi(3,2)*Xi(7,2);
  K(11,23)=Xi(3,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(3,1)*Xi(7,0) + PdF3133*Xi(3,0)*Xi(7,2) + PdF3133*Xi(3,2)*Xi(7,0) + PdF3232*Xi(3,1)*Xi(7,1) + PdF3233*Xi(3,1)*Xi(7,2) + PdF3233*Xi(3,2)*Xi(7,1) + PdF3333*Xi(3,2)*Xi(7,2);
  K(12,12)=PdF1111*pow(Xi(4,0),2) + 2*PdF1112*Xi(4,0)*Xi(4,1) + 2*PdF1113*Xi(4,0)*Xi(4,2) + PdF1212*pow(Xi(4,1),2) + 2*PdF1213*Xi(4,1)*Xi(4,2) + PdF1313*pow(Xi(4,2),2);
  K(12,13)=Xi(4,0)*(PdF1121*Xi(4,0) + PdF1122*Xi(4,1) + PdF1123*Xi(4,2)) + PdF1222*pow(Xi(4,1),2) + PdF1323*pow(Xi(4,2),2) + PdF1221*Xi(4,0)*Xi(4,1) + PdF1223*Xi(4,1)*Xi(4,2) + PdF1321*Xi(4,0)*Xi(4,2) + PdF1322*Xi(4,1)*Xi(4,2);
  K(12,14)=Xi(4,0)*(PdF1131*Xi(4,0) + PdF1132*Xi(4,1) + PdF1133*Xi(4,2)) + PdF1232*pow(Xi(4,1),2) + PdF1333*pow(Xi(4,2),2) + PdF1231*Xi(4,0)*Xi(4,1) + PdF1233*Xi(4,1)*Xi(4,2) + PdF1331*Xi(4,0)*Xi(4,2) + PdF1332*Xi(4,1)*Xi(4,2);
  K(12,15)=Xi(4,0)*(PdF1111*Xi(5,0) + PdF1112*Xi(5,1)) + PdF1112*Xi(4,1)*Xi(5,0) + PdF1113*Xi(4,0)*Xi(5,2) + PdF1113*Xi(5,0)*Xi(4,2) + PdF1212*Xi(4,1)*Xi(5,1) + PdF1213*Xi(4,1)*Xi(5,2) + PdF1213*Xi(4,2)*Xi(5,1) + PdF1313*Xi(4,2)*Xi(5,2);
  K(12,16)=Xi(4,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1)) + PdF1123*Xi(4,0)*Xi(5,2) + PdF1221*Xi(4,1)*Xi(5,0) + PdF1222*Xi(4,1)*Xi(5,1) + PdF1223*Xi(4,1)*Xi(5,2) + PdF1321*Xi(5,0)*Xi(4,2) + PdF1322*Xi(4,2)*Xi(5,1) + PdF1323*Xi(4,2)*Xi(5,2);
  K(12,17)=Xi(4,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1)) + PdF1133*Xi(4,0)*Xi(5,2) + PdF1231*Xi(4,1)*Xi(5,0) + PdF1232*Xi(4,1)*Xi(5,1) + PdF1233*Xi(4,1)*Xi(5,2) + PdF1331*Xi(5,0)*Xi(4,2) + PdF1332*Xi(4,2)*Xi(5,1) + PdF1333*Xi(4,2)*Xi(5,2);
  K(12,18)=Xi(4,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(4,1)*Xi(6,0) + PdF1113*Xi(4,0)*Xi(6,2) + PdF1113*Xi(4,2)*Xi(6,0) + PdF1212*Xi(4,1)*Xi(6,1) + PdF1213*Xi(4,1)*Xi(6,2) + PdF1213*Xi(4,2)*Xi(6,1) + PdF1313*Xi(4,2)*Xi(6,2);
  K(12,19)=Xi(4,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(4,0)*Xi(6,2) + PdF1221*Xi(4,1)*Xi(6,0) + PdF1222*Xi(4,1)*Xi(6,1) + PdF1223*Xi(4,1)*Xi(6,2) + PdF1321*Xi(4,2)*Xi(6,0) + PdF1322*Xi(4,2)*Xi(6,1) + PdF1323*Xi(4,2)*Xi(6,2);
  K(12,20)=Xi(4,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(4,0)*Xi(6,2) + PdF1231*Xi(4,1)*Xi(6,0) + PdF1232*Xi(4,1)*Xi(6,1) + PdF1233*Xi(4,1)*Xi(6,2) + PdF1331*Xi(4,2)*Xi(6,0) + PdF1332*Xi(4,2)*Xi(6,1) + PdF1333*Xi(4,2)*Xi(6,2);
  K(12,21)=Xi(4,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(4,1)*Xi(7,0) + PdF1113*Xi(4,0)*Xi(7,2) + PdF1113*Xi(4,2)*Xi(7,0) + PdF1212*Xi(4,1)*Xi(7,1) + PdF1213*Xi(4,1)*Xi(7,2) + PdF1213*Xi(4,2)*Xi(7,1) + PdF1313*Xi(4,2)*Xi(7,2);
  K(12,22)=Xi(4,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(4,0)*Xi(7,2) + PdF1221*Xi(4,1)*Xi(7,0) + PdF1222*Xi(4,1)*Xi(7,1) + PdF1223*Xi(4,1)*Xi(7,2) + PdF1321*Xi(4,2)*Xi(7,0) + PdF1322*Xi(4,2)*Xi(7,1) + PdF1323*Xi(4,2)*Xi(7,2);
  K(12,23)=Xi(4,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(4,0)*Xi(7,2) + PdF1231*Xi(4,1)*Xi(7,0) + PdF1232*Xi(4,1)*Xi(7,1) + PdF1233*Xi(4,1)*Xi(7,2) + PdF1331*Xi(4,2)*Xi(7,0) + PdF1332*Xi(4,2)*Xi(7,1) + PdF1333*Xi(4,2)*Xi(7,2);
  K(13,13)=PdF2121*pow(Xi(4,0),2) + 2*PdF2122*Xi(4,0)*Xi(4,1) + 2*PdF2123*Xi(4,0)*Xi(4,2) + PdF2222*pow(Xi(4,1),2) + 2*PdF2223*Xi(4,1)*Xi(4,2) + PdF2323*pow(Xi(4,2),2);
  K(13,14)=Xi(4,0)*(PdF2131*Xi(4,0) + PdF2132*Xi(4,1) + PdF2133*Xi(4,2)) + PdF2232*pow(Xi(4,1),2) + PdF2333*pow(Xi(4,2),2) + PdF2231*Xi(4,0)*Xi(4,1) + PdF2233*Xi(4,1)*Xi(4,2) + PdF2331*Xi(4,0)*Xi(4,2) + PdF2332*Xi(4,1)*Xi(4,2);
  K(13,15)=Xi(4,0)*(PdF1121*Xi(5,0) + PdF1221*Xi(5,1)) + PdF1122*Xi(4,1)*Xi(5,0) + PdF1123*Xi(5,0)*Xi(4,2) + PdF1222*Xi(4,1)*Xi(5,1) + PdF1223*Xi(4,2)*Xi(5,1) + PdF1321*Xi(4,0)*Xi(5,2) + PdF1322*Xi(4,1)*Xi(5,2) + PdF1323*Xi(4,2)*Xi(5,2);
  K(13,16)=Xi(4,0)*(PdF2121*Xi(5,0) + PdF2122*Xi(5,1)) + PdF2122*Xi(4,1)*Xi(5,0) + PdF2123*Xi(4,0)*Xi(5,2) + PdF2123*Xi(5,0)*Xi(4,2) + PdF2222*Xi(4,1)*Xi(5,1) + PdF2223*Xi(4,1)*Xi(5,2) + PdF2223*Xi(4,2)*Xi(5,1) + PdF2323*Xi(4,2)*Xi(5,2);
  K(13,17)=Xi(4,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1)) + PdF2133*Xi(4,0)*Xi(5,2) + PdF2231*Xi(4,1)*Xi(5,0) + PdF2232*Xi(4,1)*Xi(5,1) + PdF2233*Xi(4,1)*Xi(5,2) + PdF2331*Xi(5,0)*Xi(4,2) + PdF2332*Xi(4,2)*Xi(5,1) + PdF2333*Xi(4,2)*Xi(5,2);
  K(13,18)=Xi(4,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(4,1)*Xi(6,0) + PdF1123*Xi(4,2)*Xi(6,0) + PdF1222*Xi(4,1)*Xi(6,1) + PdF1223*Xi(4,2)*Xi(6,1) + PdF1321*Xi(4,0)*Xi(6,2) + PdF1322*Xi(4,1)*Xi(6,2) + PdF1323*Xi(4,2)*Xi(6,2);
  K(13,19)=Xi(4,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(4,1)*Xi(6,0) + PdF2123*Xi(4,0)*Xi(6,2) + PdF2123*Xi(4,2)*Xi(6,0) + PdF2222*Xi(4,1)*Xi(6,1) + PdF2223*Xi(4,1)*Xi(6,2) + PdF2223*Xi(4,2)*Xi(6,1) + PdF2323*Xi(4,2)*Xi(6,2);
  K(13,20)=Xi(4,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(4,0)*Xi(6,2) + PdF2231*Xi(4,1)*Xi(6,0) + PdF2232*Xi(4,1)*Xi(6,1) + PdF2233*Xi(4,1)*Xi(6,2) + PdF2331*Xi(4,2)*Xi(6,0) + PdF2332*Xi(4,2)*Xi(6,1) + PdF2333*Xi(4,2)*Xi(6,2);
  K(13,21)=Xi(4,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(4,1)*Xi(7,0) + PdF1123*Xi(4,2)*Xi(7,0) + PdF1222*Xi(4,1)*Xi(7,1) + PdF1223*Xi(4,2)*Xi(7,1) + PdF1321*Xi(4,0)*Xi(7,2) + PdF1322*Xi(4,1)*Xi(7,2) + PdF1323*Xi(4,2)*Xi(7,2);
  K(13,22)=Xi(4,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(4,1)*Xi(7,0) + PdF2123*Xi(4,0)*Xi(7,2) + PdF2123*Xi(4,2)*Xi(7,0) + PdF2222*Xi(4,1)*Xi(7,1) + PdF2223*Xi(4,1)*Xi(7,2) + PdF2223*Xi(4,2)*Xi(7,1) + PdF2323*Xi(4,2)*Xi(7,2);
  K(13,23)=Xi(4,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(4,0)*Xi(7,2) + PdF2231*Xi(4,1)*Xi(7,0) + PdF2232*Xi(4,1)*Xi(7,1) + PdF2233*Xi(4,1)*Xi(7,2) + PdF2331*Xi(4,2)*Xi(7,0) + PdF2332*Xi(4,2)*Xi(7,1) + PdF2333*Xi(4,2)*Xi(7,2);
  K(14,14)=PdF3131*pow(Xi(4,0),2) + 2*PdF3132*Xi(4,0)*Xi(4,1) + 2*PdF3133*Xi(4,0)*Xi(4,2) + PdF3232*pow(Xi(4,1),2) + 2*PdF3233*Xi(4,1)*Xi(4,2) + PdF3333*pow(Xi(4,2),2);
  K(14,15)=Xi(4,0)*(PdF1131*Xi(5,0) + PdF1231*Xi(5,1)) + PdF1132*Xi(4,1)*Xi(5,0) + PdF1133*Xi(5,0)*Xi(4,2) + PdF1232*Xi(4,1)*Xi(5,1) + PdF1233*Xi(4,2)*Xi(5,1) + PdF1331*Xi(4,0)*Xi(5,2) + PdF1332*Xi(4,1)*Xi(5,2) + PdF1333*Xi(4,2)*Xi(5,2);
  K(14,16)=Xi(4,0)*(PdF2131*Xi(5,0) + PdF2231*Xi(5,1)) + PdF2132*Xi(4,1)*Xi(5,0) + PdF2133*Xi(5,0)*Xi(4,2) + PdF2232*Xi(4,1)*Xi(5,1) + PdF2233*Xi(4,2)*Xi(5,1) + PdF2331*Xi(4,0)*Xi(5,2) + PdF2332*Xi(4,1)*Xi(5,2) + PdF2333*Xi(4,2)*Xi(5,2);
  K(14,17)=Xi(4,0)*(PdF3131*Xi(5,0) + PdF3132*Xi(5,1)) + PdF3132*Xi(4,1)*Xi(5,0) + PdF3133*Xi(4,0)*Xi(5,2) + PdF3133*Xi(5,0)*Xi(4,2) + PdF3232*Xi(4,1)*Xi(5,1) + PdF3233*Xi(4,1)*Xi(5,2) + PdF3233*Xi(4,2)*Xi(5,1) + PdF3333*Xi(4,2)*Xi(5,2);
  K(14,18)=Xi(4,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(4,1)*Xi(6,0) + PdF1133*Xi(4,2)*Xi(6,0) + PdF1232*Xi(4,1)*Xi(6,1) + PdF1233*Xi(4,2)*Xi(6,1) + PdF1331*Xi(4,0)*Xi(6,2) + PdF1332*Xi(4,1)*Xi(6,2) + PdF1333*Xi(4,2)*Xi(6,2);
  K(14,19)=Xi(4,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(4,1)*Xi(6,0) + PdF2133*Xi(4,2)*Xi(6,0) + PdF2232*Xi(4,1)*Xi(6,1) + PdF2233*Xi(4,2)*Xi(6,1) + PdF2331*Xi(4,0)*Xi(6,2) + PdF2332*Xi(4,1)*Xi(6,2) + PdF2333*Xi(4,2)*Xi(6,2);
  K(14,20)=Xi(4,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(4,1)*Xi(6,0) + PdF3133*Xi(4,0)*Xi(6,2) + PdF3133*Xi(4,2)*Xi(6,0) + PdF3232*Xi(4,1)*Xi(6,1) + PdF3233*Xi(4,1)*Xi(6,2) + PdF3233*Xi(4,2)*Xi(6,1) + PdF3333*Xi(4,2)*Xi(6,2);
  K(14,21)=Xi(4,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(4,1)*Xi(7,0) + PdF1133*Xi(4,2)*Xi(7,0) + PdF1232*Xi(4,1)*Xi(7,1) + PdF1233*Xi(4,2)*Xi(7,1) + PdF1331*Xi(4,0)*Xi(7,2) + PdF1332*Xi(4,1)*Xi(7,2) + PdF1333*Xi(4,2)*Xi(7,2);
  K(14,22)=Xi(4,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(4,1)*Xi(7,0) + PdF2133*Xi(4,2)*Xi(7,0) + PdF2232*Xi(4,1)*Xi(7,1) + PdF2233*Xi(4,2)*Xi(7,1) + PdF2331*Xi(4,0)*Xi(7,2) + PdF2332*Xi(4,1)*Xi(7,2) + PdF2333*Xi(4,2)*Xi(7,2);
  K(14,23)=Xi(4,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(4,1)*Xi(7,0) + PdF3133*Xi(4,0)*Xi(7,2) + PdF3133*Xi(4,2)*Xi(7,0) + PdF3232*Xi(4,1)*Xi(7,1) + PdF3233*Xi(4,1)*Xi(7,2) + PdF3233*Xi(4,2)*Xi(7,1) + PdF3333*Xi(4,2)*Xi(7,2);
  K(15,15)=PdF1111*pow(Xi(5,0),2) + 2*PdF1112*Xi(5,0)*Xi(5,1) + 2*PdF1113*Xi(5,0)*Xi(5,2) + PdF1212*pow(Xi(5,1),2) + 2*PdF1213*Xi(5,1)*Xi(5,2) + PdF1313*pow(Xi(5,2),2);
  K(15,16)=Xi(5,0)*(PdF1121*Xi(5,0) + PdF1122*Xi(5,1) + PdF1123*Xi(5,2)) + PdF1222*pow(Xi(5,1),2) + PdF1323*pow(Xi(5,2),2) + PdF1221*Xi(5,0)*Xi(5,1) + PdF1223*Xi(5,1)*Xi(5,2) + PdF1321*Xi(5,0)*Xi(5,2) + PdF1322*Xi(5,1)*Xi(5,2);
  K(15,17)=Xi(5,0)*(PdF1131*Xi(5,0) + PdF1132*Xi(5,1) + PdF1133*Xi(5,2)) + PdF1232*pow(Xi(5,1),2) + PdF1333*pow(Xi(5,2),2) + PdF1231*Xi(5,0)*Xi(5,1) + PdF1233*Xi(5,1)*Xi(5,2) + PdF1331*Xi(5,0)*Xi(5,2) + PdF1332*Xi(5,1)*Xi(5,2);
  K(15,18)=Xi(5,0)*(PdF1111*Xi(6,0) + PdF1112*Xi(6,1)) + PdF1112*Xi(5,1)*Xi(6,0) + PdF1113*Xi(5,0)*Xi(6,2) + PdF1113*Xi(6,0)*Xi(5,2) + PdF1212*Xi(5,1)*Xi(6,1) + PdF1213*Xi(5,1)*Xi(6,2) + PdF1213*Xi(5,2)*Xi(6,1) + PdF1313*Xi(5,2)*Xi(6,2);
  K(15,19)=Xi(5,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1)) + PdF1123*Xi(5,0)*Xi(6,2) + PdF1221*Xi(5,1)*Xi(6,0) + PdF1222*Xi(5,1)*Xi(6,1) + PdF1223*Xi(5,1)*Xi(6,2) + PdF1321*Xi(6,0)*Xi(5,2) + PdF1322*Xi(5,2)*Xi(6,1) + PdF1323*Xi(5,2)*Xi(6,2);
  K(15,20)=Xi(5,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1)) + PdF1133*Xi(5,0)*Xi(6,2) + PdF1231*Xi(5,1)*Xi(6,0) + PdF1232*Xi(5,1)*Xi(6,1) + PdF1233*Xi(5,1)*Xi(6,2) + PdF1331*Xi(6,0)*Xi(5,2) + PdF1332*Xi(5,2)*Xi(6,1) + PdF1333*Xi(5,2)*Xi(6,2);
  K(15,21)=Xi(5,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(5,1)*Xi(7,0) + PdF1113*Xi(5,0)*Xi(7,2) + PdF1113*Xi(5,2)*Xi(7,0) + PdF1212*Xi(5,1)*Xi(7,1) + PdF1213*Xi(5,1)*Xi(7,2) + PdF1213*Xi(5,2)*Xi(7,1) + PdF1313*Xi(5,2)*Xi(7,2);
  K(15,22)=Xi(5,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(5,0)*Xi(7,2) + PdF1221*Xi(5,1)*Xi(7,0) + PdF1222*Xi(5,1)*Xi(7,1) + PdF1223*Xi(5,1)*Xi(7,2) + PdF1321*Xi(5,2)*Xi(7,0) + PdF1322*Xi(5,2)*Xi(7,1) + PdF1323*Xi(5,2)*Xi(7,2);
  K(15,23)=Xi(5,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(5,0)*Xi(7,2) + PdF1231*Xi(5,1)*Xi(7,0) + PdF1232*Xi(5,1)*Xi(7,1) + PdF1233*Xi(5,1)*Xi(7,2) + PdF1331*Xi(5,2)*Xi(7,0) + PdF1332*Xi(5,2)*Xi(7,1) + PdF1333*Xi(5,2)*Xi(7,2);
  K(16,16)=PdF2121*pow(Xi(5,0),2) + 2*PdF2122*Xi(5,0)*Xi(5,1) + 2*PdF2123*Xi(5,0)*Xi(5,2) + PdF2222*pow(Xi(5,1),2) + 2*PdF2223*Xi(5,1)*Xi(5,2) + PdF2323*pow(Xi(5,2),2);
  K(16,17)=Xi(5,0)*(PdF2131*Xi(5,0) + PdF2132*Xi(5,1) + PdF2133*Xi(5,2)) + PdF2232*pow(Xi(5,1),2) + PdF2333*pow(Xi(5,2),2) + PdF2231*Xi(5,0)*Xi(5,1) + PdF2233*Xi(5,1)*Xi(5,2) + PdF2331*Xi(5,0)*Xi(5,2) + PdF2332*Xi(5,1)*Xi(5,2);
  K(16,18)=Xi(5,0)*(PdF1121*Xi(6,0) + PdF1221*Xi(6,1)) + PdF1122*Xi(5,1)*Xi(6,0) + PdF1123*Xi(6,0)*Xi(5,2) + PdF1222*Xi(5,1)*Xi(6,1) + PdF1223*Xi(5,2)*Xi(6,1) + PdF1321*Xi(5,0)*Xi(6,2) + PdF1322*Xi(5,1)*Xi(6,2) + PdF1323*Xi(5,2)*Xi(6,2);
  K(16,19)=Xi(5,0)*(PdF2121*Xi(6,0) + PdF2122*Xi(6,1)) + PdF2122*Xi(5,1)*Xi(6,0) + PdF2123*Xi(5,0)*Xi(6,2) + PdF2123*Xi(6,0)*Xi(5,2) + PdF2222*Xi(5,1)*Xi(6,1) + PdF2223*Xi(5,1)*Xi(6,2) + PdF2223*Xi(5,2)*Xi(6,1) + PdF2323*Xi(5,2)*Xi(6,2);
  K(16,20)=Xi(5,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1)) + PdF2133*Xi(5,0)*Xi(6,2) + PdF2231*Xi(5,1)*Xi(6,0) + PdF2232*Xi(5,1)*Xi(6,1) + PdF2233*Xi(5,1)*Xi(6,2) + PdF2331*Xi(6,0)*Xi(5,2) + PdF2332*Xi(5,2)*Xi(6,1) + PdF2333*Xi(5,2)*Xi(6,2);
  K(16,21)=Xi(5,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(5,1)*Xi(7,0) + PdF1123*Xi(5,2)*Xi(7,0) + PdF1222*Xi(5,1)*Xi(7,1) + PdF1223*Xi(5,2)*Xi(7,1) + PdF1321*Xi(5,0)*Xi(7,2) + PdF1322*Xi(5,1)*Xi(7,2) + PdF1323*Xi(5,2)*Xi(7,2);
  K(16,22)=Xi(5,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(5,1)*Xi(7,0) + PdF2123*Xi(5,0)*Xi(7,2) + PdF2123*Xi(5,2)*Xi(7,0) + PdF2222*Xi(5,1)*Xi(7,1) + PdF2223*Xi(5,1)*Xi(7,2) + PdF2223*Xi(5,2)*Xi(7,1) + PdF2323*Xi(5,2)*Xi(7,2);
  K(16,23)=Xi(5,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(5,0)*Xi(7,2) + PdF2231*Xi(5,1)*Xi(7,0) + PdF2232*Xi(5,1)*Xi(7,1) + PdF2233*Xi(5,1)*Xi(7,2) + PdF2331*Xi(5,2)*Xi(7,0) + PdF2332*Xi(5,2)*Xi(7,1) + PdF2333*Xi(5,2)*Xi(7,2);
  K(17,17)=PdF3131*pow(Xi(5,0),2) + 2*PdF3132*Xi(5,0)*Xi(5,1) + 2*PdF3133*Xi(5,0)*Xi(5,2) + PdF3232*pow(Xi(5,1),2) + 2*PdF3233*Xi(5,1)*Xi(5,2) + PdF3333*pow(Xi(5,2),2);
  K(17,18)=Xi(5,0)*(PdF1131*Xi(6,0) + PdF1231*Xi(6,1)) + PdF1132*Xi(5,1)*Xi(6,0) + PdF1133*Xi(6,0)*Xi(5,2) + PdF1232*Xi(5,1)*Xi(6,1) + PdF1233*Xi(5,2)*Xi(6,1) + PdF1331*Xi(5,0)*Xi(6,2) + PdF1332*Xi(5,1)*Xi(6,2) + PdF1333*Xi(5,2)*Xi(6,2);
  K(17,19)=Xi(5,0)*(PdF2131*Xi(6,0) + PdF2231*Xi(6,1)) + PdF2132*Xi(5,1)*Xi(6,0) + PdF2133*Xi(6,0)*Xi(5,2) + PdF2232*Xi(5,1)*Xi(6,1) + PdF2233*Xi(5,2)*Xi(6,1) + PdF2331*Xi(5,0)*Xi(6,2) + PdF2332*Xi(5,1)*Xi(6,2) + PdF2333*Xi(5,2)*Xi(6,2);
  K(17,20)=Xi(5,0)*(PdF3131*Xi(6,0) + PdF3132*Xi(6,1)) + PdF3132*Xi(5,1)*Xi(6,0) + PdF3133*Xi(5,0)*Xi(6,2) + PdF3133*Xi(6,0)*Xi(5,2) + PdF3232*Xi(5,1)*Xi(6,1) + PdF3233*Xi(5,1)*Xi(6,2) + PdF3233*Xi(5,2)*Xi(6,1) + PdF3333*Xi(5,2)*Xi(6,2);
  K(17,21)=Xi(5,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(5,1)*Xi(7,0) + PdF1133*Xi(5,2)*Xi(7,0) + PdF1232*Xi(5,1)*Xi(7,1) + PdF1233*Xi(5,2)*Xi(7,1) + PdF1331*Xi(5,0)*Xi(7,2) + PdF1332*Xi(5,1)*Xi(7,2) + PdF1333*Xi(5,2)*Xi(7,2);
  K(17,22)=Xi(5,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(5,1)*Xi(7,0) + PdF2133*Xi(5,2)*Xi(7,0) + PdF2232*Xi(5,1)*Xi(7,1) + PdF2233*Xi(5,2)*Xi(7,1) + PdF2331*Xi(5,0)*Xi(7,2) + PdF2332*Xi(5,1)*Xi(7,2) + PdF2333*Xi(5,2)*Xi(7,2);
  K(17,23)=Xi(5,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(5,1)*Xi(7,0) + PdF3133*Xi(5,0)*Xi(7,2) + PdF3133*Xi(5,2)*Xi(7,0) + PdF3232*Xi(5,1)*Xi(7,1) + PdF3233*Xi(5,1)*Xi(7,2) + PdF3233*Xi(5,2)*Xi(7,1) + PdF3333*Xi(5,2)*Xi(7,2);
  K(18,18)=PdF1111*pow(Xi(6,0),2) + 2*PdF1112*Xi(6,0)*Xi(6,1) + 2*PdF1113*Xi(6,0)*Xi(6,2) + PdF1212*pow(Xi(6,1),2) + 2*PdF1213*Xi(6,1)*Xi(6,2) + PdF1313*pow(Xi(6,2),2);
  K(18,19)=Xi(6,0)*(PdF1121*Xi(6,0) + PdF1122*Xi(6,1) + PdF1123*Xi(6,2)) + PdF1222*pow(Xi(6,1),2) + PdF1323*pow(Xi(6,2),2) + PdF1221*Xi(6,0)*Xi(6,1) + PdF1223*Xi(6,1)*Xi(6,2) + PdF1321*Xi(6,0)*Xi(6,2) + PdF1322*Xi(6,1)*Xi(6,2);
  K(18,20)=Xi(6,0)*(PdF1131*Xi(6,0) + PdF1132*Xi(6,1) + PdF1133*Xi(6,2)) + PdF1232*pow(Xi(6,1),2) + PdF1333*pow(Xi(6,2),2) + PdF1231*Xi(6,0)*Xi(6,1) + PdF1233*Xi(6,1)*Xi(6,2) + PdF1331*Xi(6,0)*Xi(6,2) + PdF1332*Xi(6,1)*Xi(6,2);
  K(18,21)=Xi(6,0)*(PdF1111*Xi(7,0) + PdF1112*Xi(7,1)) + PdF1112*Xi(6,1)*Xi(7,0) + PdF1113*Xi(6,0)*Xi(7,2) + PdF1113*Xi(7,0)*Xi(6,2) + PdF1212*Xi(6,1)*Xi(7,1) + PdF1213*Xi(6,1)*Xi(7,2) + PdF1213*Xi(6,2)*Xi(7,1) + PdF1313*Xi(6,2)*Xi(7,2);
  K(18,22)=Xi(6,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1)) + PdF1123*Xi(6,0)*Xi(7,2) + PdF1221*Xi(6,1)*Xi(7,0) + PdF1222*Xi(6,1)*Xi(7,1) + PdF1223*Xi(6,1)*Xi(7,2) + PdF1321*Xi(7,0)*Xi(6,2) + PdF1322*Xi(6,2)*Xi(7,1) + PdF1323*Xi(6,2)*Xi(7,2);
  K(18,23)=Xi(6,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1)) + PdF1133*Xi(6,0)*Xi(7,2) + PdF1231*Xi(6,1)*Xi(7,0) + PdF1232*Xi(6,1)*Xi(7,1) + PdF1233*Xi(6,1)*Xi(7,2) + PdF1331*Xi(7,0)*Xi(6,2) + PdF1332*Xi(6,2)*Xi(7,1) + PdF1333*Xi(6,2)*Xi(7,2);
  K(19,19)=PdF2121*pow(Xi(6,0),2) + 2*PdF2122*Xi(6,0)*Xi(6,1) + 2*PdF2123*Xi(6,0)*Xi(6,2) + PdF2222*pow(Xi(6,1),2) + 2*PdF2223*Xi(6,1)*Xi(6,2) + PdF2323*pow(Xi(6,2),2);
  K(19,20)=Xi(6,0)*(PdF2131*Xi(6,0) + PdF2132*Xi(6,1) + PdF2133*Xi(6,2)) + PdF2232*pow(Xi(6,1),2) + PdF2333*pow(Xi(6,2),2) + PdF2231*Xi(6,0)*Xi(6,1) + PdF2233*Xi(6,1)*Xi(6,2) + PdF2331*Xi(6,0)*Xi(6,2) + PdF2332*Xi(6,1)*Xi(6,2);
  K(19,21)=Xi(6,0)*(PdF1121*Xi(7,0) + PdF1221*Xi(7,1)) + PdF1122*Xi(6,1)*Xi(7,0) + PdF1123*Xi(7,0)*Xi(6,2) + PdF1222*Xi(6,1)*Xi(7,1) + PdF1223*Xi(6,2)*Xi(7,1) + PdF1321*Xi(6,0)*Xi(7,2) + PdF1322*Xi(6,1)*Xi(7,2) + PdF1323*Xi(6,2)*Xi(7,2);
  K(19,22)=Xi(6,0)*(PdF2121*Xi(7,0) + PdF2122*Xi(7,1)) + PdF2122*Xi(6,1)*Xi(7,0) + PdF2123*Xi(6,0)*Xi(7,2) + PdF2123*Xi(7,0)*Xi(6,2) + PdF2222*Xi(6,1)*Xi(7,1) + PdF2223*Xi(6,1)*Xi(7,2) + PdF2223*Xi(6,2)*Xi(7,1) + PdF2323*Xi(6,2)*Xi(7,2);
  K(19,23)=Xi(6,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1)) + PdF2133*Xi(6,0)*Xi(7,2) + PdF2231*Xi(6,1)*Xi(7,0) + PdF2232*Xi(6,1)*Xi(7,1) + PdF2233*Xi(6,1)*Xi(7,2) + PdF2331*Xi(7,0)*Xi(6,2) + PdF2332*Xi(6,2)*Xi(7,1) + PdF2333*Xi(6,2)*Xi(7,2);
  K(20,20)=PdF3131*pow(Xi(6,0),2) + 2*PdF3132*Xi(6,0)*Xi(6,1) + 2*PdF3133*Xi(6,0)*Xi(6,2) + PdF3232*pow(Xi(6,1),2) + 2*PdF3233*Xi(6,1)*Xi(6,2) + PdF3333*pow(Xi(6,2),2);
  K(20,21)=Xi(6,0)*(PdF1131*Xi(7,0) + PdF1231*Xi(7,1)) + PdF1132*Xi(6,1)*Xi(7,0) + PdF1133*Xi(7,0)*Xi(6,2) + PdF1232*Xi(6,1)*Xi(7,1) + PdF1233*Xi(6,2)*Xi(7,1) + PdF1331*Xi(6,0)*Xi(7,2) + PdF1332*Xi(6,1)*Xi(7,2) + PdF1333*Xi(6,2)*Xi(7,2);
  K(20,22)=Xi(6,0)*(PdF2131*Xi(7,0) + PdF2231*Xi(7,1)) + PdF2132*Xi(6,1)*Xi(7,0) + PdF2133*Xi(7,0)*Xi(6,2) + PdF2232*Xi(6,1)*Xi(7,1) + PdF2233*Xi(6,2)*Xi(7,1) + PdF2331*Xi(6,0)*Xi(7,2) + PdF2332*Xi(6,1)*Xi(7,2) + PdF2333*Xi(6,2)*Xi(7,2);
  K(20,23)=Xi(6,0)*(PdF3131*Xi(7,0) + PdF3132*Xi(7,1)) + PdF3132*Xi(6,1)*Xi(7,0) + PdF3133*Xi(6,0)*Xi(7,2) + PdF3133*Xi(7,0)*Xi(6,2) + PdF3232*Xi(6,1)*Xi(7,1) + PdF3233*Xi(6,1)*Xi(7,2) + PdF3233*Xi(6,2)*Xi(7,1) + PdF3333*Xi(6,2)*Xi(7,2);
  K(21,21)=PdF1111*pow(Xi(7,0),2) + 2*PdF1112*Xi(7,0)*Xi(7,1) + 2*PdF1113*Xi(7,0)*Xi(7,2) + PdF1212*pow(Xi(7,1),2) + 2*PdF1213*Xi(7,1)*Xi(7,2) + PdF1313*pow(Xi(7,2),2);
  K(21,22)=Xi(7,0)*(PdF1121*Xi(7,0) + PdF1122*Xi(7,1) + PdF1123*Xi(7,2)) + PdF1222*pow(Xi(7,1),2) + PdF1323*pow(Xi(7,2),2) + PdF1221*Xi(7,0)*Xi(7,1) + PdF1223*Xi(7,1)*Xi(7,2) + PdF1321*Xi(7,0)*Xi(7,2) + PdF1322*Xi(7,1)*Xi(7,2);
  K(21,23)=Xi(7,0)*(PdF1131*Xi(7,0) + PdF1132*Xi(7,1) + PdF1133*Xi(7,2)) + PdF1232*pow(Xi(7,1),2) + PdF1333*pow(Xi(7,2),2) + PdF1231*Xi(7,0)*Xi(7,1) + PdF1233*Xi(7,1)*Xi(7,2) + PdF1331*Xi(7,0)*Xi(7,2) + PdF1332*Xi(7,1)*Xi(7,2);
  K(22,22)=PdF2121*pow(Xi(7,0),2) + 2*PdF2122*Xi(7,0)*Xi(7,1) + 2*PdF2123*Xi(7,0)*Xi(7,2) + PdF2222*pow(Xi(7,1),2) + 2*PdF2223*Xi(7,1)*Xi(7,2) + PdF2323*pow(Xi(7,2),2);
  K(22,23)=Xi(7,0)*(PdF2131*Xi(7,0) + PdF2132*Xi(7,1) + PdF2133*Xi(7,2)) + PdF2232*pow(Xi(7,1),2) + PdF2333*pow(Xi(7,2),2) + PdF2231*Xi(7,0)*Xi(7,1) + PdF2233*Xi(7,1)*Xi(7,2) + PdF2331*Xi(7,0)*Xi(7,2) + PdF2332*Xi(7,1)*Xi(7,2);
  K(23,23)=PdF3131*pow(Xi(7,0),2) + 2*PdF3132*Xi(7,0)*Xi(7,1) + 2*PdF3133*Xi(7,0)*Xi(7,2) + PdF3232*pow(Xi(7,1),2) + 2*PdF3233*Xi(7,1)*Xi(7,2) + PdF3333*pow(Xi(7,2),2);
  return dJ * K;
}


template <typename T> 
Dense<T,24,24> K_element_alt ( const Dense<T,8,3>& X ,
                           const Dense<T,8,3>& u ,
                           const Dense<T,192,1>& G , T c, T d ) {
  Dense<T,24,24> K = Dense<T,24,24>::Zero();
  for (size_t i=0; i < 8; ++i) {
    Map<const RowMaj<T,8,3>> g(G.data() + i * 24);
    K += K_point(X,u,(Dense<T,8,3>)g,c,d);
  }
  return K.template selfadjointView<Eigen::UpLoType::Upper>();
}


template <typename T> 
Dense<T,24,24> K_element ( const Dense<T,8,3>& X ,
                           const Dense<T,8,3>& u ,
                           const Dense<T,64,3>& G , T c, T d ) {
  Dense<T,24,24> K = Dense<T,24,24>::Zero();
  for (size_t i=0; i < 8; ++i) {
    Dense<T,8,3> g = G.template block<8,3>(i * 8,0);
    K += K_point(X,u,g,c,d);
  }
  return K.template selfadjointView<Eigen::UpLoType::Upper>();
}


template <typename T>
Dense<T,24,24> K_element_python ( const Dense<T,8,3>& X ,
                                  const Dense<T,8,3>& u ,
                                  T E, T nu ) {
  T d = E * nu / (2 * (1+nu) * (1 - 2 * nu));
  T c = E / (4 * (1 + nu));
  return K_element(X,u,gauss<T>(),c,d);
}


template <typename T>
Indices K_global_cr (const Dynamic<T,1>& XG,
                         const Dynamic<T,1>& uG,
                         const Numpy<size_t>& Hex,
                         const Numpy<size_t>& Dof,
                         T E, T nu ) {
  size_t limit = Hex.shape(0), i = 0;
  Map<CDynamic<size_t,8>> Hexm((size_t *) Hex.request().ptr, 8, limit);
  Map<CDynamic<size_t,3>> Dofm((size_t *) Dof.request().ptr, 3, Dof.shape(0));
  Indices indices(576 * limit);
  for (;i < limit; ++i) {
    Dense<size_t,3,8> index = Dofm(all, Hexm.col(i));
    size_t *ptr = index.data();
    for (size_t j=0; j<24;++j) {
      std::fill(indices.r + i * 576 + j * 24, indices.r + i * 576 + ((j+1)*24),index(j));
      std::copy(ptr,ptr+24,indices.c + i * 576 + (j*24));
    }
  }
  return indices;
}


template <typename T>
Numpy<T> K_global_val (const Numpy<T>& XG,
                       const Numpy<T>& uG,
                       const Numpy<size_t>& Hex,
                       const Numpy<size_t>& Dof,
                       T E, T nu ) {
  T c = E / (4 * (1 + nu));
  T d = E * nu / (2 * (1+nu) * (1 - 2 * nu));
  size_t limit = Hex.shape(0), i = 0;
  Map<CDynamic<T,1>> XGm((T *) XG.request().ptr, XG.size());
  Map<CDynamic<T,1>> uGm((T *) uG.request().ptr, uG.size());
  Map<CDynamic<size_t,8>> Hexm((size_t *) Hex.request().ptr, 8, limit);
  Map<CDynamic<size_t,3>> Dofm((size_t *) Dof.request().ptr, 3, Dof.shape(0));
  Numpy<T> values(576 * limit);
  T *val_ptr = (T *) values.request().ptr;
  Dense<T,192,1> g = gauss_alt<T>();
  for (;i<limit; ++i) {
    RowMaj<size_t,3,8> index = Dofm(all,Hexm.col(i));
    Map<Dense<size_t,1,24>> inm(index.data());
    Dense<T,24,24> K = K_element_alt((Dense<T,8,3>)XGm(inm).reshaped(8,3),(Dense<T,8,3>) uGm(inm).reshaped(8,3),g,c,d);
    std::copy(K.data(), K.data()+576, val_ptr + (i * 576));
  }
  return values;
}


template <typename T>
Dense<T,8,3> f_internal ( const Dense<T,8,3>& X ,
    const Dense<T,8,3>& u ,
    const Dense<T,8,3>& g ,T c, T d ) {
  Dense<T,3,3> J = g.transpose() * X;
  Dense<T,3,3> Ji = J.inverse();
  Dense<T,8,3> Xi = g * Ji.transpose();
  Dense<T,3,3> F = Dense<T,3,3>::Identity() + u.transpose * Xi;
  Dense<T,3,3> Fi = F.inverse().transpose();
  T dF, dJ;
  dF = F.determinant(); dJ = J.determinant();
  Dense<T,3,3> P = 2 * c * (F - Fi) + 2 * d * Fi * log(dF);
  return Xi * P.transpose() * dJ;
}

template <lui nc, lui nd>
Matrix<nc,nd> f_internal (const Matrix<nc,nd>& X,
    const Matrix<nc,nd>& u,
    const Matrix<nc,nd>& g,
    double c, double d) {
  const Matrix<nd,nd> J = g.transpose() * X;
  const double dJ = J.determinant();
  const Matrix<nd,nd> Ji = J.inverse();
  const Matrix<nc,nd> Xi = g * Ji.transpose();
  const Matrix<nd,nd> F = Matrix<nd,nd>::Identity() + u.transpose() * Xi;
  const Matrix<nd,nd> Fi = F.inverse().transpose();
  const double dF = F.determinant();
  const Matrix<nd,nd> P = 2 * c * (F - Fi) + 2 * d * Fi * log(dF);
  return Xi * P.transpose() * dJ;
}

template <size_t ng, size_t nc, size_t nd>
Matrix<nc, nd> f_element (const Matrix<nc,nd>& X,
    const Matrix<nc,nd>& u,
    const Matrix<nc * ng, nd>& G,
    double c, double d) {
  Matrix<nc,nd> result = Matrix<nc,nd>::Zero();
  for (size_t i=0; i<ng; ++i) {
    result += f_internal<nc,nd>(X,u,G.template block<nc,nd>(i*nc,0),c,d);
  }
  return result;
}

template <typename T>
Dense<T,8,3> f_element_python ( const Dense<T,8,3>& X ,
    const Dense<T,8,3>& u ,
    T E, T nu ) {
  T d = E * nu / (2 * (1+nu) * (1 - 2 * nu));
  T c = E / (4 * (1 + nu));
  return f_element<8,8,3>(X,u,gauss<T>(),c,d);
}


PYBIND11_MODULE(eigen_backend, m) {
  m.def("f_internal", &f_internal<8,3>, "blah blah");
  m.def("f_element", &f_element<8,8,3>, "blah blah");
  m.def("K_point", &K_point<double>, "blah blah");
  m.def("K_element", &K_element<double>, "blah blah");
  m.def("K_element_py", &K_element_python<double>, "blah blah");
  m.def("K_global_cr", &K_global_cr<double>); 
  m.def("K_global_val", &K_global_val<double>); 
  m.def("gauss_points", &gauss<double>);
  m.def("f_element_py", &f_element_python<double>);
  py::class_<Indices>(m,"Indices")
    .def(py::init<size_t &>())
    .def_readwrite("rows"  , &Indices::rows)
    .def_readwrite("cols"  , &Indices::cols);
}
// g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) eigen_backend.cpp -o eigen_backend$(python3-config --extension-suffix) -llapack -L ~/OpenBLAS/build/lib/ -l:libopenblas.a -DNGP=<insert number of global points> -DNGE=<insert number of global elements>
