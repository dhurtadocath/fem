#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <limits>

namespace py = pybind11;

// Bernstein basis functions (generic)
double Bernstein(int n, int k, double t) {
    if (k < 0 || k > n) {
        return 0; 
    }
    // Calculate binomial coefficient C(n, i)
    double binom = 1;
    for (int i = 1; i <= k; ++i) {
        binom = binom * (n - i + 1) / i; 
    }
    return binom * std::pow(t, k) * std::pow(1 - t, n - k);
}

// Factorial function
double factorial(int n) {
    if (n <= 1) return 1;
    double result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Binomial coefficient
double comb(int n, int k) {
    if (k > n || k < 0) return 0;
    if (k == 0 || k == n) return 1;
    
    double result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

// dnBernstein function
double dnBernstein(int n, int k, double x, int p) {
    double coef = factorial(n) / factorial(n - p);
    int desde = std::max(0, k + p - n);
    int hasta = std::min(k, p);
    
    double dnB = 0.0;
    for (int i = desde; i <= hasta; ++i) {
        double sign = (i + p) % 2 == 0 ? 1.0 : -1.0;
        dnB += sign * comb(p, i) * Bernstein(n - p, k - i, x);
    }
    return coef * dnB;
}

// Optimized Bernstein basis for cubic case n = 3
inline void cubic_bernstein(double t, double B[4]) {
    double omt = 1.0 - t;
    double omt2 = omt * omt;
    double omt3 = omt2 * omt;
    double t2 = t * t;
    double t3 = t2 * t;

    B[0] = omt3;              // (1 - t)^3
    B[1] = 3.0 * t * omt2;    // 3 t (1 - t)^2
    B[2] = 3.0 * t2 * omt;    // 3 t^2 (1 - t)
    B[3] = t3;                // t^3
}

inline void cubic_bernstein_deriv(double t, double dB[4]) {
    double omt = 1.0 - t;
    double omt2 = omt * omt;
    double t2 = t * t;

    dB[0] = -3.0 * omt2;                      // d/dt (1 - t)^3
    dB[1] = 3.0 * omt2 - 6.0 * t * omt;       // d/dt [3 t (1 - t)^2]
    dB[2] = 6.0 * t * omt - 3.0 * t2;         // d/dt [3 t^2 (1 - t)]
    dB[3] = 3.0 * t2;                         // d/dt t^3
}

inline void cubic_bernstein_second_deriv(double t, double ddB[4]) {
    double omt = 1.0 - t;

    ddB[0] = 6.0 * omt;               // d2/dt2 (1 - t)^3
    ddB[1] = -12.0 + 18.0 * t;        // d2/dt2 [3 t (1 - t)^2]
    ddB[2] = 6.0 - 18.0 * t;          // d2/dt2 [3 t^2 (1 - t)]
    ddB[3] = 6.0 * t;                 // d2/dt2 t^3
}


// Grg function: core implementation templated on the control-point storage
template <typename Derived>
Eigen::Vector3d Grg_impl(const Eigen::DenseBase<Derived> &CtrlPts_base, double u, double v, double eps) {
    const auto &CtrlPts = CtrlPts_base.derived();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    int n = 3; // Degree of Bernstein polynomial in u
    int m = 3; // Degree of Bernstein polynomial in v

    // Precompute cubic Bernstein basis values for u and v
    double Bu[4], Bv[4];
    cubic_bernstein(u, Bu);
    cubic_bernstein(v, Bv);

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            Eigen::Vector3d xij = Eigen::Vector3d::Zero(); // Initialize xij
            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                Eigen::Vector3d cp1;
                Eigen::Vector3d cp2;

                // These indices must match the flatCtrlPts order in Python
                if (i == 1 && j == 1) {
                    cp1 = CtrlPts.row(12);
                    cp2 = CtrlPts.row(13);
                } else if (i == 1 && j == 2) {
                    cp1 = CtrlPts.row(18);
                    cp2 = CtrlPts.row(19);
                } else if (i == 2 && j == 1) {
                    cp1 = CtrlPts.row(14);
                    cp2 = CtrlPts.row(15);
                } else { // i == 2 && j == 2
                    cp1 = CtrlPts.row(16);
                    cp2 = CtrlPts.row(17);
                }

                double den;
                if (i == 1 && j == 1) {
                    den = std::max(eps, u + v);
                    xij = (u * cp1 + v * cp2) / den;
                } else if (i == 1 && j == 2) {
                    den = std::max(eps, u + 1 - v);
                    xij = (u * cp1 + (1 - v) * cp2) / den;
                } else if (i == 2 && j == 1) {
                    den = std::max(eps, v + 1 - u);
                    xij = ((1 - u) * cp1 + v * cp2) / den;
                } else { // i == 2 && j == 2
                    den = std::max(eps, 2 - u - v);
                    xij = ((1 - u) * cp1 + (1 - v) * cp2) / den;
                }
            } else {
                // Direct access for boundary and corner nodes
                // These indices must match the flatCtrlPts order in Python
                if (i == 0 && j == 0) xij = CtrlPts.row(0);
                else if (i == 3 && j == 0) xij = CtrlPts.row(1);
                else if (i == 3 && j == 3) xij = CtrlPts.row(2);
                else if (i == 0 && j == 3) xij = CtrlPts.row(3);
                else if (i == 1 && j == 0) xij = CtrlPts.row(4);
                else if (i == 2 && j == 0) xij = CtrlPts.row(5);
                else if (i == 3 && j == 1) xij = CtrlPts.row(6);
                else if (i == 3 && j == 2) xij = CtrlPts.row(7);
                else if (i == 2 && j == 3) xij = CtrlPts.row(8);
                else if (i == 1 && j == 3) xij = CtrlPts.row(9);
                else if (i == 0 && j == 2) xij = CtrlPts.row(10);
                else if (i == 0 && j == 1) xij = CtrlPts.row(11);
            }
            
            double Bi = Bu[i];
            double Bj = Bv[j];
            p += Bi * Bj * xij;
        }
    }
    return p;
}

// Public Grg wrapper used by Python/pybind, keeps original signature
Eigen::Vector3d Grg(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    return Grg_impl(CtrlPts, u, v, eps);
}

// Structs for derivative results (pure C++)
struct GrgDerivsResult {
    Eigen::Vector3d p;
    Eigen::Vector3d D1p;
    Eigen::Vector3d D2p;
};

struct GrgDerivs2Result {
    Eigen::Vector3d p;
    Eigen::Vector3d D1p;
    Eigen::Vector3d D2p;
    Eigen::Vector3d D1D1p;
    Eigen::Vector3d D1D2p;
    Eigen::Vector3d D2D2p;
};

// Internal implementation of Grg_derivs (no pybind types)
GrgDerivsResult Grg_derivs_impl(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    GrgDerivsResult res;
    res.p.setZero();
    res.D1p.setZero();
    res.D2p.setZero();
    int n = 3;
    int m = 3;

    // Precompute cubic Bernstein basis and first derivatives for u and v
    double Bu[4], Bv[4];
    double dBu[4], dBv[4];
    cubic_bernstein(u, Bu);
    cubic_bernstein(v, Bv);
    cubic_bernstein_deriv(u, dBu);
    cubic_bernstein_deriv(v, dBv);

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            Eigen::Vector3d xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D1xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D2xij = Eigen::Vector3d::Zero();

            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                Eigen::Vector3d cp1, cp2;

                if (i == 1 && j == 1) {
                    cp1 = CtrlPts.row(12);
                    cp2 = CtrlPts.row(13);
                    double den = std::max(eps, u + v);
                    xij = (u * cp1 + v * cp2) / den;
                    D1xij = cp1 / den - (u * cp1 + v * cp2) / (den * den);
                    D2xij = cp2 / den - (u * cp1 + v * cp2) / (den * den);
                } else if (i == 1 && j == 2) {
                    cp1 = CtrlPts.row(18);
                    cp2 = CtrlPts.row(19);
                    double den = std::max(eps, u + 1.0 - v);
                    xij = (u * cp1 + (1.0 - v) * cp2) / den;
                    D1xij = cp1 / den - (u * cp1 + (1.0 - v) * cp2) / (den * den);
                    D2xij = -cp2 / den + (u * cp1 + (1.0 - v) * cp2) / (den * den);
                } else if (i == 2 && j == 1) {
                    cp1 = CtrlPts.row(14);
                    cp2 = CtrlPts.row(15);
                    double den = std::max(eps, v + 1.0 - u);
                    xij = ((1.0 - u) * cp1 + v * cp2) / den;
                    D1xij = -cp1 / den + ((1.0 - u) * cp1 + v * cp2) / (den * den);
                    D2xij = cp2 / den - ((1.0 - u) * cp1 + v * cp2) / (den * den);
                } else { // i == 2 && j == 2
                    cp1 = CtrlPts.row(16);
                    cp2 = CtrlPts.row(17);
                    double den = std::max(eps, 2.0 - u - v);
                    xij = ((1.0 - u) * cp1 + (1.0 - v) * cp2) / den;
                    D1xij = -cp1 / den + ((1.0 - u) * cp1 + (1.0 - v) * cp2) / (den * den);
                    D2xij = -cp2 / den + ((1.0 - u) * cp1 + (1.0 - v) * cp2) / (den * den);
                }
            } else {
                if (i == 0 && j == 0) xij = CtrlPts.row(0);
                else if (i == 3 && j == 0) xij = CtrlPts.row(1);
                else if (i == 3 && j == 3) xij = CtrlPts.row(2);
                else if (i == 0 && j == 3) xij = CtrlPts.row(3);
                else if (i == 1 && j == 0) xij = CtrlPts.row(4);
                else if (i == 2 && j == 0) xij = CtrlPts.row(5);
                else if (i == 3 && j == 1) xij = CtrlPts.row(6);
                else if (i == 3 && j == 2) xij = CtrlPts.row(7);
                else if (i == 2 && j == 3) xij = CtrlPts.row(8);
                else if (i == 1 && j == 3) xij = CtrlPts.row(9);
                else if (i == 0 && j == 2) xij = CtrlPts.row(10);
                else if (i == 0 && j == 1) xij = CtrlPts.row(11);
            }

            double Bi = Bu[i];
            double Bj = Bv[j];
            double D1Bi = dBu[i];
            double D2Bj = dBv[j];

            res.p += Bi * Bj * xij;
            res.D1p += D1Bi * Bj * xij + Bi * Bj * D1xij;
            res.D2p += Bi * D2Bj * xij + Bi * Bj * D2xij;
        }
    }
    return res;
}

// pybind wrapper for Grg_derivs
py::tuple Grg_derivs(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    GrgDerivsResult res = Grg_derivs_impl(CtrlPts, u, v, eps);
    return py::make_tuple(res.p, res.D1p, res.D2p);
}

// Internal implementation of Grg_derivs2 (no pybind types) – generic core
template <typename Derived>
GrgDerivs2Result Grg_derivs2_impl_generic(const Eigen::DenseBase<Derived> &CtrlPts_base, double u, double v, double eps) {
    const auto &CtrlPts = CtrlPts_base.derived();
    GrgDerivs2Result res;
    res.p.setZero();
    res.D1p.setZero();
    res.D2p.setZero();
    res.D1D1p.setZero();
    res.D1D2p.setZero();
    res.D2D2p.setZero();
    int n = 3;
    int m = 3;

    // Precompute cubic Bernstein basis and derivatives for u and v
    double Bu[4], Bv[4];
    double dBu[4], dBv[4];
    double ddBu[4], ddBv[4];
    cubic_bernstein(u, Bu);
    cubic_bernstein(v, Bv);
    cubic_bernstein_deriv(u, dBu);
    cubic_bernstein_deriv(v, dBv);
    cubic_bernstein_second_deriv(u, ddBu);
    cubic_bernstein_second_deriv(v, ddBv);

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            Eigen::Vector3d xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D1xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D2xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D11xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D12xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D22xij = Eigen::Vector3d::Zero();

            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                Eigen::Vector3d cp1, cp2;

                if (i == 1 && j == 1) {
                    cp1 = CtrlPts.row(12);
                    cp2 = CtrlPts.row(13);
                    double den = std::max(eps, u + v);
                    xij = (u * cp1 + v * cp2) / den;
                    D1xij = cp1 / den - (u * cp1 + v * cp2) / (den * den);
                    D2xij = cp2 / den - (u * cp1 + v * cp2) / (den * den);
                    D11xij = -2 * v * (cp1 - cp2) / (den * den * den);
                    D12xij = (u - v) * (cp1 - cp2) / (den * den * den);
                    D22xij = 2 * u * (cp1 - cp2) / (den * den * den);
                } else if (i == 1 && j == 2) {
                    cp1 = CtrlPts.row(18);
                    cp2 = CtrlPts.row(19);
                    double den = std::max(eps, u + 1.0 - v);
                    xij = (u * cp1 + (1.0 - v) * cp2) / den;
                    D1xij = cp1 / den - (u * cp1 + (1.0 - v) * cp2) / (den * den);
                    D2xij = -cp2 / den + (u * cp1 + (1.0 - v) * cp2) / (den * den);
                    D11xij = (2 * (-1 + v) * (cp1 - cp2)) / (den * den * den);
                    D12xij = -((-1 + u + v) * (cp1 - cp2)) / (den * den * den);
                    D22xij = (2 * u * (cp1 - cp2)) / (den * den * den);
                } else if (i == 2 && j == 1) {
                    cp1 = CtrlPts.row(14);
                    cp2 = CtrlPts.row(15);
                    double den = std::max(eps, v + 1.0 - u);
                    xij = ((1.0 - u) * cp1 + v * cp2) / den;
                    D1xij = -cp1 / den + ((1.0 - u) * cp1 + v * cp2) / (den * den);
                    D2xij = cp2 / den - ((1.0 - u) * cp1 + v * cp2) / (den * den);
                    D11xij = (2 * v * (cp1 - cp2)) / (-den * den * den);
                    D12xij = ((-1 + u + v) * (cp1 - cp2)) / (den * den * den);
                    D22xij = (2 * (-1 + u) * (cp1 - cp2)) / (-den * den * den);
                } else { // i == 2 && j == 2
                    cp1 = CtrlPts.row(16);
                    cp2 = CtrlPts.row(17);
                    double den = std::max(eps, 2.0 - u - v);
                    xij = ((1.0 - u) * cp1 + (1.0 - v) * cp2) / den;
                    D1xij = -cp1 / den + ((1.0 - u) * cp1 + (1.0 - v) * cp2) / (den * den);
                    D2xij = -cp2 / den + ((1.0 - u) * cp1 + (1.0 - v) * cp2) / (den * den);
                    D11xij = -((2 * (-1 + v) * (cp1 - cp2)) / (-den * den * den));
                    D12xij = ((u - v) * (cp1 - cp2)) / (-den * den * den);
                    D22xij = (2 * (-1 + u) * (cp1 - cp2)) / (-den * den * den);
                }
            } else {
                if (i == 0 && j == 0) xij = CtrlPts.row(0);
                else if (i == 3 && j == 0) xij = CtrlPts.row(1);
                else if (i == 3 && j == 3) xij = CtrlPts.row(2);
                else if (i == 0 && j == 3) xij = CtrlPts.row(3);
                else if (i == 1 && j == 0) xij = CtrlPts.row(4);
                else if (i == 2 && j == 0) xij = CtrlPts.row(5);
                else if (i == 3 && j == 1) xij = CtrlPts.row(6);
                else if (i == 3 && j == 2) xij = CtrlPts.row(7);
                else if (i == 2 && j == 3) xij = CtrlPts.row(8);
                else if (i == 1 && j == 3) xij = CtrlPts.row(9);
                else if (i == 0 && j == 2) xij = CtrlPts.row(10);
                else if (i == 0 && j == 1) xij = CtrlPts.row(11);
            }

            double Bi = Bu[i];
            double Bj = Bv[j];
            double D1Bi = dBu[i];
            double D2Bj = dBv[j];
            double DD1Bi = ddBu[i];
            double DD2Bj = ddBv[j];

            res.p      += Bi * Bj * xij;
            res.D1p    += D1Bi * Bj * xij + Bi * Bj * D1xij;
            res.D2p    += Bi * D2Bj * xij + Bi * Bj * D2xij;
            res.D1D1p  += (DD1Bi * xij + 2.0 * D1Bi * D1xij + Bi * D11xij) * Bj;
            res.D1D2p  += D1Bi * D2Bj * xij + D1Bi * Bj * D2xij + Bi * D2Bj * D1xij + Bi * Bj * D12xij;
            res.D2D2p  += (DD2Bj * xij + 2.0 * D2Bj * D2xij + Bj * D22xij) * Bi;
        }
    }

    return res;
}

// MatrixXd-specific wrapper used by existing callers / pybind
GrgDerivs2Result Grg_derivs2_impl(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    return Grg_derivs2_impl_generic(CtrlPts, u, v, eps);
}

// pybind wrapper for Grg_derivs2
py::tuple Grg_derivs2(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    GrgDerivs2Result res = Grg_derivs2_impl(CtrlPts, u, v, eps);
    return py::make_tuple(res.p, res.D1p, res.D2p, res.D1D1p, res.D1D2p, res.D2D2p);
}

// Micro-optimized MinDist - maintains exact Python logic with performance improvements
py::tuple MinDist(const Eigen::MatrixXd &CtrlPts, const Eigen::Vector3d &x, int seeding, double eps, 
                  double x0 = 0.0, double x1 = 1.0, double y0 = 0.0, double y1 = 1.0, 
                  bool recursive = false, int recursionLevel = 0, double prev_u = -1.0, double prev_v = -1.0) {
    double umin = 0.0;
    double vmin = 0.0;
    Eigen::Vector3d initial_point = CtrlPts.row(0);  
    double dmin = (x - initial_point).norm();

    // Handle recursive seeding - exact Python logic
    int actual_seeding = recursive ? ((seeding > 1) ? seeding : 4) : seeding;

    // Micro-optimization 1: Pre-compute step sizes
    double du = (x1 - x0) / actual_seeding;
    double dv = (y1 - y0) / actual_seeding;
    
    // Micro-optimization 2: Early termination threshold
    double very_close_threshold = 1e-12;
    
    // Exact Python grid search logic with micro-optimizations
    for (int i = 0; i <= actual_seeding; ++i) {
        double u = x0 + i * du;
        
        // Micro-optimization 3: Skip obviously out-of-bounds u values
        if (u < -0.01 || u > 1.01) continue;
        
        for (int j = 0; j <= actual_seeding; ++j) {
            double v = y0 + j * dv;
            
            // Micro-optimization 4: Skip obviously out-of-bounds v values  
            if (v < -0.01 || v > 1.01) continue;
            
            Eigen::Vector3d p = Grg(CtrlPts, u, v, eps);
            double d = (x - p).norm();
            
            if (d < dmin) {
                dmin = d;
                umin = u;
                vmin = v;
                
                // Micro-optimization 5: Early termination for very close points
                if (d < very_close_threshold) {
                    return py::make_tuple(umin, vmin);
                }
            }
        }
    }

    // Recursive refinement logic - exact Python translation
    if (recursive && recursionLevel < 8) {
        bool should_refine = (prev_u < 0.0 || prev_v < 0.0) || 
                            (std::abs(prev_u - umin) > 5e-3 || std::abs(prev_v - vmin) > 5e-3);
        
        if (should_refine) {
            double dx = x1 - x0;
            double dy = y1 - y0;
            
            // Exact Python formula for refined bounds
            double new_x0 = std::max(0.0, umin - 7*dx/(16*actual_seeding));
            double new_x1 = std::min(1.0, umin + 7*dx/(16*actual_seeding));
            double new_y0 = std::max(0.0, vmin - 7*dy/(16*actual_seeding));
            double new_y1 = std::min(1.0, vmin + 7*dy/(16*actual_seeding));
            
            return MinDist(CtrlPts, x, seeding, eps, new_x0, new_x1, new_y0, new_y1, 
                          recursive, recursionLevel + 1, umin, vmin);
        }
    }

    return py::make_tuple(umin, vmin);
}

// Wrapper for non-recursive calls
py::tuple MinDist(const Eigen::MatrixXd &CtrlPts, const Eigen::Vector3d &x, int seeding, double eps) {
    return MinDist(CtrlPts, x, seeding, eps, 0.0, 1.0, 0.0, 1.0, false, 0, -1.0, -1.0);
}

// D3Grg function - rigorous translation of Python logic
Eigen::Vector3d D3Grg(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps, bool normalize = true) {
    // Exact translation of Python: D1p, D2p = self.Grg(t, deriv = 1)[1].T
    py::tuple derivs = Grg_derivs(CtrlPts, u, v, eps);
    Eigen::Vector3d D1p = derivs[1].cast<Eigen::Vector3d>();
    Eigen::Vector3d D2p = derivs[2].cast<Eigen::Vector3d>();
    
    // Exact translation of Python: D3p = np.cross(D1p,D2p)
    Eigen::Vector3d D3p = D1p.cross(D2p);
    
    // Exact translation of Python: if norm(D3p) ==0: set_trace()
    double norm_D3p = D3p.norm();
    if (norm_D3p == 0.0) {
        // In C++, we can't set_trace(), but we should handle this case
        // Fallback or throw? For strict equivalency with "set_trace", throw.
        throw std::runtime_error("D3p norm is zero in D3Grg - would trigger set_trace in Python");
    }
    
    // Exact translation of Python: if normalize: D3p = D3p/norm(D3p)
    if (normalize) {
        D3p /= norm_D3p;
    }
    
    return D3p;
}

// Helper function for backward compatibility
Eigen::Vector3d D3Grg_helper(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps, bool normalize) {
    return D3Grg(CtrlPts, u, v, eps, normalize);
}

// D3Grg template helper for internal use (avoids pybind tuple overhead)
template <typename Derived>
Eigen::Vector3d D3Grg_internal(const Eigen::DenseBase<Derived> &CtrlPts_base, double u, double v, double eps, bool normalize = true) {
    GrgDerivsResult d = Grg_derivs_impl(CtrlPts_base.derived(), u, v, eps);
    Eigen::Vector3d D3p = d.D1p.cross(d.D2p);
    double norm_D3p = D3p.norm();
    if (norm_D3p == 0.0) {
         // handle singularity gracefully if possible, or throw
         return Eigen::Vector3d::Zero(); 
    }
    if (normalize) {
        D3p /= norm_D3p;
    }
    return D3p;
}

// Combined point and normal helper to avoid redundant evaluation:
// computes p(u,v) and unit normal n(u,v) in a single pass.
template <typename Derived>
void point_and_normal_internal(const Eigen::DenseBase<Derived> &CtrlPts_base,
                               double u,
                               double v,
                               double eps,
                               Eigen::Vector3d &p,
                               Eigen::Vector3d &normal) {
    GrgDerivsResult d = Grg_derivs_impl(CtrlPts_base.derived(), u, v, eps);
    Eigen::Vector3d D3p = d.D1p.cross(d.D2p);
    double norm_D3p = D3p.norm();
    if (norm_D3p == 0.0) {
        normal.setZero();
        p = d.p;
        return;
    }
    normal = D3p / norm_D3p;
    p = d.p;
}

// find_projection function
py::tuple find_projection(const Eigen::MatrixXd &CtrlPts, const Eigen::Vector3d &xs, py::tuple t_py, double bs_r, double eps) {
    Eigen::Vector2d t(t_py[0].cast<double>(), t_py[1].cast<double>());

    double tol = 1e-15; 
    double res = 1.0 + tol;
    int niter = 0;
    Eigen::Vector2d tcandidate = t;

    // Initialize candidate tracking 
    Eigen::Vector3d xc_candidate = Grg(CtrlPts, tcandidate.x(), tcandidate.y(), eps);
    double dist = (xs - xc_candidate).norm();

    double opa = 1e-2;  // Match original Python opa value

    // Get initial derivatives and f vector 
    py::tuple derivs2 = Grg_derivs2(CtrlPts, t.x(), t.y(), eps);
    Eigen::Vector3d xc = derivs2[0].cast<Eigen::Vector3d>();
    Eigen::Vector3d D1p = derivs2[1].cast<Eigen::Vector3d>();
    Eigen::Vector3d D2p = derivs2[2].cast<Eigen::Vector3d>();
    
    Eigen::Matrix<double, 3, 2> dxcdt;
    dxcdt.col(0) = D1p;
    dxcdt.col(1) = D2p;
    Eigen::Vector2d f = -2 * dxcdt.transpose() * (xs - xc);

    // Main Newton-Raphson iteration
    while (res > tol && (t.x() >= -opa && t.x() <= 1.0 + opa) && (t.y() >= -opa && t.y() <= 1.0 + opa)) {
        
        // Get second derivatives for this iteration
        py::tuple derivs2_iter = Grg_derivs2(CtrlPts, t.x(), t.y(), eps);
        Eigen::Vector3d xc_iter = derivs2_iter[0].cast<Eigen::Vector3d>();
        Eigen::Vector3d D1p_iter = derivs2_iter[1].cast<Eigen::Vector3d>();
        Eigen::Vector3d D2p_iter = derivs2_iter[2].cast<Eigen::Vector3d>();
        Eigen::Vector3d D1D1p = derivs2_iter[3].cast<Eigen::Vector3d>();
        Eigen::Vector3d D1D2p = derivs2_iter[4].cast<Eigen::Vector3d>();
        Eigen::Vector3d D2D2p = derivs2_iter[5].cast<Eigen::Vector3d>();

        // Build K matrix 
        Eigen::Matrix2d K;
        K(0,0) = 2.0 * (-(xs - xc_iter).dot(D1D1p) + D1p_iter.dot(D1p_iter));
        K(0,1) = 2.0 * (-(xs - xc_iter).dot(D1D2p) + D1p_iter.dot(D2p_iter));
        K(1,0) = K(0,1);
        K(1,1) = 2.0 * (-(xs - xc_iter).dot(D2D2p) + D2p_iter.dot(D2p_iter));

        Eigen::Vector2d dt = -K.inverse() * f;
        t += dt;
        
        // Update xc, dxcdt, and f for next iteration
        py::tuple derivs2_new = Grg_derivs2(CtrlPts, t.x(), t.y(), eps);
        xc = derivs2_new[0].cast<Eigen::Vector3d>();
        D1p = derivs2_new[1].cast<Eigen::Vector3d>();
        D2p = derivs2_new[2].cast<Eigen::Vector3d>();
        
        dxcdt.col(0) = D1p;
        dxcdt.col(1) = D2p;
        f = -2 * dxcdt.transpose() * (xs - xc);

        // Match original Python: res = np.linalg.norm(dt)
        res = dt.norm();

        // Match original Python convergence check exactly
        if (res < std::sqrt(tol) && !(t.x() > 0 && t.x() < 1 && t.y() > 0 && t.y() < 1)) {
            return py::make_tuple(-1.0, -1.0);
        }

        niter++;
        if (niter > 10) {
            // Match original Python: dist_new = norm(xs - xc)
            double dist_new = (xs - xc).norm();
            if (dist_new < dist) {
                dist = dist_new;
                tcandidate = t;
            }
            if (niter > 13) {
                // Return candidate like original Python
                return py::make_tuple(tcandidate.x(), tcandidate.y());
            }
        }
    }

    // Return t like original Python (before proj_final_check)
    return py::make_tuple(t.x(), t.y());
}

// Internal result struct for TR projection
struct TRProjResult {
    Eigen::Vector2d t;
    double m;
};

// Helper for TR subproblem (Steihaug-CG), ported from legacy backend
std::pair<Eigen::Vector2d, bool> tr_subproblem(const Eigen::Vector2d &f,
                                               const Eigen::Matrix2d &K,
                                               double delta) {
    Eigen::Vector2d r = -f;
    Eigen::Vector2d p = r;
    Eigen::Vector2d q = r;
    Eigen::Vector2d h = Eigen::Vector2d::Zero();

    // Early exit if gradient is already small
    if (r.norm() < std::max(1e-15, 1e-5 * f.norm())) {
        return std::make_pair(h, false);
    }

    int iter = 0;
    while (true) {
        if (iter++ > 10) break;  // Safety cap as in legacy implementation

        double pKp = p.dot(K * p);
        if (pKp <= 0.0) {
            // Negative curvature: step to boundary
            double a_coeff = p.squaredNorm();
            double b_coeff = 2.0 * h.dot(p);
            double c_coeff = h.squaredNorm() - delta * delta;
            double disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
            if (disc < 0.0) disc = 0.0;
            double alpha = (-b_coeff + std::sqrt(disc)) / (2.0 * a_coeff);
            h += alpha * p;
            return std::make_pair(h, true);
        }

        double alpha = r.dot(q) / pKp;

        // Check if proposed step hits boundary
        Eigen::Vector2d h_trial = h + alpha * p;
        if (h_trial.squaredNorm() >= delta * delta) {
            double a_coeff = p.squaredNorm();
            double b_coeff = 2.0 * h.dot(p);
            double c_coeff = h.squaredNorm() - delta * delta;
            double disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
            if (disc < 0.0) disc = 0.0;
            double alpha_bd = (-b_coeff + std::sqrt(disc)) / (2.0 * a_coeff);
            h += alpha_bd * p;
            return std::make_pair(h, true);
        }

        h = h_trial;
        double phi = r.dot(p);

        Eigen::Vector2d r_new = r - alpha * (K * p);
        if (r_new.norm() < std::max(1e-15, 1e-5 * f.norm())) {
            break;
        }

        double numerator = r_new.dot(r_new);
        double beta = numerator / phi;
        p = r_new + beta * p;

        r = r_new;
        q = r;
    }

    return std::make_pair(h, false);
}

// Small Newton refinement on m(t) = ||x(t) - xs||^2 in parameter space.
// Uses the same gradient/Hessian definitions as in TR and keeps (u,v) in [0,1]^2.
template <typename Derived>
void newton_refine_t_core(const Eigen::DenseBase<Derived> &CtrlPts_base,
                          const Eigen::Vector3d &xs,
                          double eps,
                          Eigen::Vector2d &t,
                          double &m) {
    const auto &CtrlPts = CtrlPts_base.derived();

    // If current m is not finite, do not attempt refinement
    if (!std::isfinite(m)) {
        return;
    }

    const int max_iter = 5000;
    const double grad_tol = 1e-12;
    const double step_tol = 1e-15;

    // Ensure we start inside the [0,1]^2 box
    t.x() = std::min(1.0, std::max(0.0, t.x()));
    t.y() = std::min(1.0, std::max(0.0, t.y()));

    // Initial objective value at starting point
    GrgDerivs2Result d2 = Grg_derivs2_impl_generic(CtrlPts, t.x(), t.y(), eps);
    Eigen::Vector3d diff = d2.p - xs;
    double m_old = diff.squaredNorm();

    Eigen::Matrix<double, 3, 2> dxcdt;
    Eigen::Vector2d g;
    Eigen::Matrix2d K;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Geometry and derivatives at current (u, v)
        d2 = Grg_derivs2_impl_generic(CtrlPts, t.x(), t.y(), eps);
        diff = d2.p - xs;

        dxcdt.col(0) = d2.D1p;
        dxcdt.col(1) = d2.D2p;

        // Gradient g = 2 * J^T * diff
        g = 2.0 * dxcdt.transpose() * diff;
        double gnorm = g.norm();
        if (gnorm < grad_tol) {
            break;
        }

        // Hessian H = 2 * (J^T J + sum_i diff_i * H_i)
        K(0,0) = 2.0 * (d2.D1p.dot(d2.D1p) + diff.dot(d2.D1D1p));
        K(0,1) = 2.0 * (d2.D1p.dot(d2.D2p) + diff.dot(d2.D1D2p));
        K(1,0) = K(0,1);
        K(1,1) = 2.0 * (d2.D2p.dot(d2.D2p) + diff.dot(d2.D2D2p));

        // Solve for Newton step: K * dt = -g
        Eigen::Vector2d dt = -K.ldlt().solve(g);
        double dt_norm = dt.norm();
        if (dt_norm < step_tol) {
            break;
        }

        // Backtracking line search in (u,v), enforcing [0,1]^2 and monotone m
        double alpha = 1.0;
        bool accepted = false;
        for (int ls = 0; ls < 6; ++ls) {
            Eigen::Vector2d t_trial = t + alpha * dt;
            t_trial.x() = std::min(1.0, std::max(0.0, t_trial.x()));
            t_trial.y() = std::min(1.0, std::max(0.0, t_trial.y()));

            Eigen::Vector3d xc_trial = Grg_impl(CtrlPts, t_trial.x(), t_trial.y(), eps);
            Eigen::Vector3d diff_trial = xc_trial - xs;
            double m_trial = diff_trial.squaredNorm();

            if (m_trial < m_old) {
                t = t_trial;
                m_old = m_trial;
                accepted = true;
                break;
            }

            alpha *= 0.5;
        }

        if (!accepted) {
            // No acceptable step found; stop refinement
            break;
        }
    }

    m = m_old;
}

// Core trust-region projection with 9 fixed seeds (t1t2_init) and TR-CG step
// Templated on the control-point storage to avoid copies (e.g. blocks of a bigger matrix)
template <typename Derived>
TRProjResult projection_tr_core(const Eigen::DenseBase<Derived> &CtrlPts_base,
                                const Eigen::Vector3d &xs,
                                double eps,
                                double TR_init,
                                double TR_min,
                                double TR_max)
{
    const auto &CtrlPts = CtrlPts_base.derived();
    // Fixed 3x3 grid seeds in parameter space
    static const double seeds[9][2] = {
        {0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0},
        {0.0, 0.5}, {0.5, 0.5}, {1.0, 0.5},
        {0.0, 1.0}, {0.5, 1.0}, {1.0, 1.0}
    };

    const double eps_min = 1e-15;   // parameter bounds tolerance (as in Python)

    double best_m = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_t(-1.0, -1.0);

    // Objects reused across iterations to avoid repeated allocations
    Eigen::Matrix<double, 3, 2> dxcdt;
    Eigen::Vector2d f;
    Eigen::Matrix2d K;

    // Loop over the 9 initial seeds
    for (int s = 0; s < 9; ++s) {
        Eigen::Vector2d t(seeds[s][0], seeds[s][1]);

        // Initial objective at seed
        Eigen::Vector3d xc0 = Grg(CtrlPts, t.x(), t.y(), eps);
        double m_new = (xs - xc0).squaredNorm();
        double m_old = 1.0e10;
        double TR_radius = TR_init;
        bool flag_new_u = true;
        int tr_iter = 0;
        const int max_tr_iter = 500;

        // Outer TR loop: stop when objective stops decreasing or trust region becomes too small/outside domain
        while (m_new < m_old && tr_iter < max_tr_iter) {
            ++tr_iter;
            if (flag_new_u) {
                // Compute xc, first and second derivatives using the pure C++ core (generic)
                GrgDerivs2Result d2 = Grg_derivs2_impl_generic(CtrlPts, t.x(), t.y(), eps);
                Eigen::Vector3d xc = d2.p;

                dxcdt.col(0) = d2.D1p;
                dxcdt.col(1) = d2.D2p;

                Eigen::Vector3d diff = xc - xs;

                // Gradient of m(t) = ||x(t) - xs||^2 in parameter space
                f = 2.0 * dxcdt.transpose() * diff;

                // If the gradient in parameter space is already very small, we are at
                // a stationary point of the distance functional on this patch.
                if (f.norm() < 1e-12) {
                    break;
                }

                // Hessian of m(t) as in Python: 2*(J^T J + sum_i diff_i * H_i)
                K(0,0) = 2.0 * (d2.D1p.dot(d2.D1p) + diff.dot(d2.D1D1p));
                K(0,1) = 2.0 * (d2.D1p.dot(d2.D2p) + diff.dot(d2.D1D2p));
                K(1,0) = K(0,1);
                K(1,1) = 2.0 * (d2.D2p.dot(d2.D2p) + diff.dot(d2.D2D2p));
            }

            // Trust-region CG step for this (f, K) via legacy Steihaug-CG helper
            auto sub_result = tr_subproblem(f, K, TR_radius);
            Eigen::Vector2d h = sub_result.first;
            bool flag_boundary_reached = sub_result.second;

            // Evaluate new objective at t + h, enforcing the [0,1]^2 box in (u,v)
            Eigen::Vector2d t_new = t + h;
            Eigen::Vector2d t_new_clamped = t_new;
            t_new_clamped.x() = std::min(1.0, std::max(0.0, t_new_clamped.x()));
            t_new_clamped.y() = std::min(1.0, std::max(0.0, t_new_clamped.y()));

            // Effective step actually taken after enforcing the box
            Eigen::Vector2d h_eff = t_new_clamped - t;

            Eigen::Vector3d xc_new = Grg_impl(CtrlPts, t_new_clamped.x(), t_new_clamped.y(), eps);
            double m_new_plus_h = (xs - xc_new).squaredNorm();

            // Predicted reduction from quadratic model
            double hKh = (h_eff.transpose() * K * h_eff)(0,0);
            double pred = -f.dot(h) - 0.5 * hKh;
            double ratio;
            if (std::abs(pred) < 1e-30) {
                ratio = 0.0;
            } else {
                ratio = (m_new - m_new_plus_h) / pred;
            }

            if (ratio < 0.25) {
                TR_radius *= 0.25;
                flag_new_u = false;
            } else {
                t = t_new_clamped;
                if (ratio > 0.75 && flag_boundary_reached) {
                    TR_radius = std::min(2.0 * TR_radius, TR_max);
                }
                flag_new_u = true;
            }

            // Stopping criteria: small TR radius or t too far from patch
            if (TR_radius < TR_min ||
                t.x() < -0.5 || t.x() > 1.5 ||
                t.y() < -0.5 || t.y() > 1.5) {
                break;
            }

            if (flag_new_u) {
                m_old = m_new;
                m_new = m_new_plus_h;
            }
        } // end outer TR loop for this seed

        // Keep best (within [eps_min, 1] for both parameters)
        if (t.x() >= eps_min && t.x() <= 1.0 &&
            t.y() >= eps_min && t.y() <= 1.0) {
            if (m_new < best_m) {
                best_m = m_new;
                best_t = t;
            }
        }
    } // end loop over seeds

    TRProjResult result;
    result.t = best_t;
    result.m = best_m;

    // Final local Newton refinement on the best patch, if we found a valid candidate
    if (best_t.x() >= eps_min && best_t.x() <= 1.0 &&
        best_t.y() >= eps_min && best_t.y() <= 1.0 &&
        std::isfinite(best_m)) {
        newton_refine_t_core(CtrlPts_base, xs, eps, result.t, result.m);
    }

    return result;
}

// Python-exposed wrapper for single-patch TR projection
py::tuple find_projection_tr(const Eigen::MatrixXd &CtrlPts,
                             const Eigen::Vector3d &xs,
                             double eps,
                             double TR_init,
                             double TR_min,
                             double TR_max)
{
    TRProjResult res = projection_tr_core(CtrlPts, xs, eps, TR_init, TR_min, TR_max);
    return py::make_tuple(res.t.x(), res.t.y(), res.m);
}

// Python-exposed wrapper for many patches / one point TR projection
// CtrlPtsAll: stacked CtrlPts for all patches, shape (npatches*20, 3)
// candidate_indices: 1D array of patch indices to consider
// radii: 1D array of patch bounding-sphere radii, length npatches
py::tuple find_projection_tr_multi(const Eigen::MatrixXd &CtrlPtsAll,
                                   const Eigen::Vector3d &xs,
                                   py::array_t<int> candidate_indices,
                                   py::array_t<double> radii,
                                   double eps,
                                   double TR_init,
                                   double TR_min,
                                   double TR_max)
{
    auto buf = candidate_indices.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("candidate_indices must be a 1D array");
    }
    int nCand = static_cast<int>(buf.shape[0]);
    int *idx_ptr = static_cast<int*>(buf.ptr);

    // Each patch contributes 20 control points
    const int rows_per_patch = 20;
    if (CtrlPtsAll.rows() % rows_per_patch != 0) {
        throw std::runtime_error("CtrlPtsAll.rows() must be a multiple of 20");
    }
    int nPatches = CtrlPtsAll.rows() / rows_per_patch;

    // Radii array
    auto rbuf = radii.request();
    if (rbuf.ndim != 1) {
        throw std::runtime_error("radii must be a 1D array");
    }
    if (static_cast<int>(rbuf.shape[0]) != nPatches) {
        throw std::runtime_error("radii length must match number of patches");
    }
    double *r_ptr = static_cast<double*>(rbuf.ptr);

    double best_m = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_t(-1.0, -1.0);
    int best_patch = -1;

    for (int k = 0; k < nCand; ++k) {
        int p_id = idx_ptr[k];
        if (p_id < 0 || p_id >= nPatches) {
            throw std::runtime_error("candidate index out of range");
        }
        double r = r_ptr[p_id];

        // View of CtrlPts for this patch: 20x3 block (no heap allocation)
        auto CtrlPts = CtrlPtsAll.block(p_id * rows_per_patch, 0, rows_per_patch, 3);
        TRProjResult res = projection_tr_core(
            CtrlPts,
            xs,
            eps,
            TR_init,
            TR_min,
            TR_max
        );

        Eigen::Vector2d t = res.t;
        double u = t.x();
        double v = t.y();

        // If TR core did not find a valid seed, skip this patch
        if (u < 0.0 || v < 0.0) {
            continue;
        }

        // Geometric final check equivalent to proj_final_check in Python
        bool inside = (u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0);
        if (!inside) {
            double uc = std::min(1.0, std::max(0.0, u));
            double vc = std::min(1.0, std::max(0.0, v));
            Eigen::Vector3d xc0, nor0;
            point_and_normal_internal(CtrlPts, uc, vc, eps, xc0, nor0);
            Eigen::Vector3d diff0 = xs - xc0;
            Eigen::Vector3d x_tang = diff0 - diff0.dot(nor0) * nor0;
            if (x_tang.norm() > 2.0 * r / 100.0) {
                // Equivalent to returning [-1,-1] and being discarded in Python
                continue;
            }
        }

        // Use squared distance returned by TR core at the accepted parameters
        double m_val = res.m;

        if (m_val < best_m) {
            best_m = m_val;
            best_t = t;
            best_patch = p_id;
        }
    }

    return py::make_tuple(best_patch, best_t.x(), best_t.y(), best_m);
}

// Function to find the signed distance and its gradient (normal)
// Reuses the efficient TR multi-patch projection logic
// Returns: (gn, nx, ny, nz, patch_id, u, v)
py::tuple find_signed_distance(const Eigen::MatrixXd &CtrlPtsAll,
                               const Eigen::Vector3d &xs,
                               py::array_t<int> candidate_indices,
                               py::array_t<double> radii,
                               double eps,
                               double TR_init,
                               double TR_min,
                               double TR_max)
{
    // Reuse the projection logic to find the closest point
    py::tuple proj = find_projection_tr_multi(CtrlPtsAll, xs, candidate_indices, radii, eps, TR_init, TR_min, TR_max);
    
    int best_patch = proj[0].cast<int>();
    double u = proj[1].cast<double>();
    double v = proj[2].cast<double>();
    
    // If no valid projection was found
    if (best_patch == -1) {
        double inf = std::numeric_limits<double>::infinity();
        return py::make_tuple(inf, 0.0, 0.0, 0.0, -1, -1.0, -1.0);
    }
    
    // Reconstruct geometry for the optimal patch
    const int rows_per_patch = 20;
    auto CtrlPts = CtrlPtsAll.block(best_patch * rows_per_patch, 0, rows_per_patch, 3);
    
    // Compute point and normal in a single pass
    Eigen::Vector3d p;
    Eigen::Vector3d normal;
    point_and_normal_internal(CtrlPts, u, v, eps, p, normal);
    
    // Compute Signed Distance
    // gn = (xs - p) . n
    double gn = (xs - p).dot(normal);
    
    return py::make_tuple(gn, normal.x(), normal.y(), normal.z(), best_patch, u, v);
}


// BoundingSphere ContainsNode function - rigorous translation of Python logic
bool ContainsNode(const Eigen::Vector3d &sphere_center, double sphere_radius, const Eigen::Vector3d &point) {
    // Exact translation of: return norm(xp-self.x) <= self.r
    return (point - sphere_center).norm() <= sphere_radius;
}

// Vectorized version for multiple points - could be even faster
py::array_t<bool> ContainsNodes(const Eigen::Vector3d &sphere_center, double sphere_radius, py::array_t<double> points) {
    auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Points array must be (N, 3) shape");
    }
    
    int n_points = buf.shape[0];
    auto result = py::array_t<bool>(n_points);
    auto result_buf = result.request();
    
    double *points_ptr = static_cast<double*>(buf.ptr);
    bool *result_ptr = static_cast<bool*>(result_buf.ptr);
    
    for (int i = 0; i < n_points; ++i) {
        Eigen::Vector3d point(points_ptr[i*3], points_ptr[i*3+1], points_ptr[i*3+2]);
        result_ptr[i] = (point - sphere_center).norm() <= sphere_radius;
    }
    
    return result;
}

// Helper: precompute NN^{-1} once for m_el_extra (depends only on Gauss points)
static const Eigen::Matrix<double, 8, 8> &precomputed_NN_inv() {
    static const Eigen::Matrix<double, 8, 8> NN_inv = []() {
        Eigen::Matrix<double, 8, 3> gauss_points;
        double gp = 1.0 / std::sqrt(3.0);
        gauss_points << -gp, -gp, -gp,
                         gp, -gp, -gp,
                         gp,  gp, -gp,
                        -gp,  gp, -gp,
                        -gp, -gp,  gp,
                         gp, -gp,  gp,
                         gp,  gp,  gp,
                        -gp,  gp,  gp;

        Eigen::Matrix<double, 8, 8> NN;
        for (int g_i = 0; g_i < 8; ++g_i) {
            double g1 = gauss_points(g_i, 0);
            double g2 = gauss_points(g_i, 1);
            double g3 = gauss_points(g_i, 2);

            NN(g_i, 0) = (1 - g1) * (1 - g2) * (1 - g3);
            NN(g_i, 1) = (1 + g1) * (1 - g2) * (1 - g3);
            NN(g_i, 2) = (1 + g1) * (1 + g2) * (1 - g3);
            NN(g_i, 3) = (1 - g1) * (1 + g2) * (1 - g3);
            NN(g_i, 4) = (1 - g1) * (1 - g2) * (1 + g3);
            NN(g_i, 5) = (1 + g1) * (1 - g2) * (1 + g3);
            NN(g_i, 6) = (1 + g1) * (1 + g2) * (1 + g3);
            NN(g_i, 7) = (1 - g1) * (1 + g2) * (1 + g3);
            NN.row(g_i) *= 1.0 / 8.0;
        }
        return NN.inverse();
    }();
    return NN_inv;
}

// FEAssembly m_el_extra function - rigorous translation of Python hyperelastic SED computation
// Use fixed-size Eigen types (8 nodes, 3 components) to avoid heap allocations.
Eigen::Matrix<double, 8, 1> m_el_extra(const Eigen::Matrix<double, 8, 3> &X_hexa,
                                       const Eigen::Matrix<double, 8, 3> &u_hexa,
                                       double Youngsmodulus,
                                       double Poissonsratio) {
    // X_hexa: 8x3 matrix of hexahedron node coordinates
    // u_hexa: 8x3 matrix of displacements (matches Python u[self.DoFs[hexa]] output)

    // Exact translation of Python material parameters
    double d1 = Youngsmodulus * Poissonsratio
                / (2.0 * (1.0 + Poissonsratio) * (1.0 - 2.0 * Poissonsratio));
    double c10 = Youngsmodulus / (4.0 * (1.0 + Poissonsratio));

    // Gauss points - exact translation of Python array
    Eigen::Matrix<double, 8, 3> gauss_points;
    double gp = 1.0 / std::sqrt(3.0);
    gauss_points << -gp, -gp, -gp,
                     gp, -gp, -gp,
                     gp,  gp, -gp,
                    -gp,  gp, -gp,
                    -gp, -gp,  gp,
                     gp, -gp,  gp,
                     gp,  gp,  gp,
                    -gp,  gp,  gp;

    Eigen::Matrix<double, 8, 1> SED;

    // Loop over 8 Gauss points - exact translation of Python logic
    for (int g_i = 0; g_i < 8; ++g_i) {
        double g1 = gauss_points(g_i, 0);
        double g2 = gauss_points(g_i, 1);
        double g3 = gauss_points(g_i, 2);

        // Shape function derivatives - exact translation of Python dNd_xi
        Eigen::Matrix<double, 8, 3> dNd_xi;
        dNd_xi << -(1 - g2) * (1 - g3), -(1 - g1) * (1 - g3), -(1 - g1) * (1 - g2),
                   (1 - g2) * (1 - g3), -(1 + g1) * (1 - g3), -(1 + g1) * (1 - g2),
                   (1 + g2) * (1 - g3),  (1 + g1) * (1 - g3), -(1 + g1) * (1 + g2),
                  -(1 + g2) * (1 - g3),  (1 - g1) * (1 - g3), -(1 - g1) * (1 + g2),
                  -(1 - g2) * (1 + g3), -(1 - g1) * (1 + g3),  (1 - g1) * (1 - g2),
                   (1 - g2) * (1 + g3), -(1 + g1) * (1 + g3),  (1 + g1) * (1 - g2),
                   (1 + g2) * (1 + g3),  (1 + g1) * (1 + g3),  (1 + g1) * (1 + g2),
                  -(1 + g2) * (1 + g3),  (1 - g1) * (1 + g3),  (1 - g1) * (1 + g2);
        dNd_xi *= 1.0 / 8.0;

        // Jacobian computation - exact translation: J = np.dot(dNd_xi.T, X)
        Eigen::Matrix3d J = dNd_xi.transpose() * X_hexa;

        // Jacobian inverse - exact translation: invJ = np.linalg.inv(J)
        Eigen::Matrix3d invJ = J.inverse();

        // Global derivatives - exact translation: dNdx = np.dot(dNd_xi, invJ.T)
        Eigen::Matrix<double, 8, 3> dNdx = dNd_xi * invJ.transpose();

        // Deformation gradient - exact translation: F = np.eye(len(dNdx.T)) + np.dot(dNdx.T, u).T
        Eigen::Matrix3d F =
            Eigen::Matrix3d::Identity() + (dNdx.transpose() * u_hexa).transpose();

        // Determinant - exact translation: detF = np.linalg.det(F)
        double detF = F.determinant();

        // SED computation - exact translation of Python hyperelastic formula
        // SED[g_i] = c10 * (np.trace(F.T @ F) - 3 - 2 * np.log(detF)) + d1 * (np.log(detF))**2
        double trace_FtF = (F.transpose() * F).trace();
        double log_detF = std::log(detF);
        SED(g_i) = c10 * (trace_FtF - 3.0 - 2.0 * log_detF) + d1 * log_detF * log_detF;

        // Shape functions NN are handled in precomputed_NN_inv()
    }

    // Final extrapolation - exact translation of: return SED @ np.linalg.inv(NN)
    // Python row-vector formulation y^T = SED^T * inv(NN)
    // Column-vector equivalent: y = inv(NN)^T * SED
    const Eigen::Matrix<double, 8, 8> &NN_inv = precomputed_NN_inv();
    Eigen::Matrix<double, 8, 1> nodal_sed = NN_inv.transpose() * SED;
    return nodal_sed;
}


PYBIND11_MODULE(gregory_patch_backend, m) {
    m.doc() = "C++ backend for Gregory patch calculations and BoundingSphere operations";
    m.def("Grg", &Grg, "A function that calculates a point on a Gregory patch");
    m.def("Grg_derivs", &Grg_derivs, "A function that calculates first derivatives of Grg");
    m.def("Grg_derivs2", &Grg_derivs2, "A function that calculates second derivatives of Grg");
    m.def("MinDist", py::overload_cast<const Eigen::MatrixXd&, const Eigen::Vector3d&, int, double>(&MinDist), 
          "A function that calculates the minimum distance from a point to a Gregory patch");
    m.def("MinDist", py::overload_cast<const Eigen::MatrixXd&, const Eigen::Vector3d&, int, double, double, double, double, double, bool, int, double, double>(&MinDist), 
          "A function that calculates the minimum distance with full parameters");
    m.def("find_projection", &find_projection, "A function that finds the projection of a point onto a Gregory patch");
    m.def("find_projection_tr", &find_projection_tr, "Trust-region projection of a point onto a Gregory patch");
    m.def("find_projection_tr_multi", &find_projection_tr_multi, "Trust-region projection for many patches and one point");
    m.def("find_signed_distance", &find_signed_distance, "Find signed distance, normal (gradient), and projection");
    m.def("D3Grg", &D3Grg, "Calculate the normal vector at (u,v) on Gregory patch", py::arg("CtrlPts"), py::arg("u"), py::arg("v"), py::arg("eps"), py::arg("normalize")=true);
    m.def("ContainsNode", &ContainsNode, "Check if a point is contained within a bounding sphere");
    m.def("ContainsNodes", &ContainsNodes, "Vectorized check if multiple points are contained within a bounding sphere");
    m.def("m_el_extra", &m_el_extra, "Compute strain energy density at nodes for hyperelastic material");
}
