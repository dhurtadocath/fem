#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <limits>

namespace py = pybind11;

// Bernstein basis functions
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


// Grg function
Eigen::Vector3d Grg(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    int n = 3; // Degree of Bernstein polynomial in u
    int m = 3; // Degree of Bernstein polynomial in v

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
            
            double Bi = Bernstein(n, i, u);
            double Bj = Bernstein(m, j, v);
            p += Bi * Bj * xij;
        }
    }
    return p;
}

// Grg_derivs function
py::tuple Grg_derivs(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D1p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D2p = Eigen::Vector3d::Zero();
    int n = 3;
    int m = 3;

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            Eigen::Vector3d xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D1xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D2xij = Eigen::Vector3d::Zero();

            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                Eigen::Vector3d cp1;
                Eigen::Vector3d cp2;

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

            double Bi = Bernstein(n, i, u);
            double Bj = Bernstein(m, j, v);
            double D1Bi = dnBernstein(n, i, u, 1);  // Match original Python exactly
            double D2Bj = dnBernstein(m, j, v, 1);  // Match original Python exactly

            p += Bi * Bj * xij;
            D1p += D1Bi * Bj * xij + Bi * Bj * D1xij;
            D2p += Bi * D2Bj * xij + Bi * Bj * D2xij;
        }
    }
    return py::make_tuple(p, D1p, D2p);
}

// Grg_derivs2 function
py::tuple Grg_derivs2(const Eigen::MatrixXd &CtrlPts, double u, double v, double eps) {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D1p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D2p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D1D1p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D1D2p = Eigen::Vector3d::Zero();
    Eigen::Vector3d D2D2p = Eigen::Vector3d::Zero();
    int n = 3;
    int m = 3;

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            Eigen::Vector3d xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D1xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D2xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D11xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D12xij = Eigen::Vector3d::Zero();
            Eigen::Vector3d D22xij = Eigen::Vector3d::Zero();

            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                Eigen::Vector3d cp1;
                Eigen::Vector3d cp2;

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

            double Bi = Bernstein(n, i, u);
            double Bj = Bernstein(m, j, v);
            double D1Bi = dnBernstein(n, i, u, 1);  // Match original Python exactly
            double D2Bj = dnBernstein(m, j, v, 1);  // Match original Python exactly
            double DD1Bi = dnBernstein(n, i, u, 2);  // Match original Python exactly
            double DD2Bj = dnBernstein(m, j, v, 2);  // Match original Python exactly

            p += Bi * Bj * xij;
            D1p += D1Bi * Bj * xij + Bi * Bj * D1xij;
            D2p += Bi * D2Bj * xij + Bi * Bj * D2xij;
            D1D1p += (DD1Bi * xij + 2 * D1Bi * D1xij + Bi * D11xij) * Bj;
            D1D2p += D1Bi * D2Bj * xij + D1Bi * Bj * D2xij + Bi * D2Bj * D1xij + Bi * Bj * D12xij;
            D2D2p += (DD2Bj * xij + 2 * D2Bj * D2xij + Bj * D22xij) * Bi;
        }
    }
    return py::make_tuple(p, D1p, D2p, D1D1p, D1D2p, D2D2p);
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

// Core trust-region projection with 9 fixed seeds (t1t2_init) and TR-CG step
TRProjResult projection_tr_core(const Eigen::MatrixXd &CtrlPts,
                                const Eigen::Vector3d &xs,
                                double eps,
                                double TR_init,
                                double TR_min,
                                double TR_max)
{
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
    Eigen::Vector2d r, p, q, h;

    // Loop over the 9 initial seeds
    for (int s = 0; s < 9; ++s) {
        Eigen::Vector2d t(seeds[s][0], seeds[s][1]);

        // Initial objective at seed
        Eigen::Vector3d xc0 = Grg(CtrlPts, t.x(), t.y(), eps);
        double m_new = (xs - xc0).squaredNorm();
        double m_old = 1.0e10;
        double TR_radius = TR_init;
        bool flag_new_u = true;

        // Outer TR loop: stop when objective stops decreasing or trust region becomes too small/outside domain
        while (m_new < m_old) {
            if (flag_new_u) {
                // Compute xc, first and second derivatives
                py::tuple derivs2 = Grg_derivs2(CtrlPts, t.x(), t.y(), eps);
                Eigen::Vector3d xc = derivs2[0].cast<Eigen::Vector3d>();
                Eigen::Vector3d D1p = derivs2[1].cast<Eigen::Vector3d>();
                Eigen::Vector3d D2p = derivs2[2].cast<Eigen::Vector3d>();
                Eigen::Vector3d D1D1p = derivs2[3].cast<Eigen::Vector3d>();
                Eigen::Vector3d D1D2p = derivs2[4].cast<Eigen::Vector3d>();
                Eigen::Vector3d D2D2p = derivs2[5].cast<Eigen::Vector3d>();

                dxcdt.col(0) = D1p;
                dxcdt.col(1) = D2p;

                Eigen::Vector3d diff = xc - xs;

                // Gradient of m(t) = ||x(t) - xs||^2 in parameter space
                f = 2.0 * dxcdt.transpose() * diff;

                // Hessian of m(t) as in Python: 2*(J^T J + sum_i diff_i * H_i)
                K(0,0) = 2.0 * (D1p.dot(D1p) + diff.dot(D1D1p));
                K(0,1) = 2.0 * (D1p.dot(D2p) + diff.dot(D1D2p));
                K(1,0) = K(0,1);
                K(1,1) = 2.0 * (D2p.dot(D2p) + diff.dot(D2D2p));
            }

            // Trust-region CG step for this (f, K)
            r = -f;
            p = r;
            q = p;
            h.setZero();
            bool flag_boundary_reached = false;

            // Inner CG loop
            while (true) {
                double pKp = p.transpose() * K * p;
                if (pKp <= 0.0) {
                    // Negative curvature: go to boundary in direction p
                    flag_boundary_reached = true;
                    double a = p.squaredNorm();
                    double b = 2.0 * p.dot(h);
                    double c = h.squaredNorm() - TR_radius * TR_radius;
                    double disc = b * b - 4.0 * a * c;
                    if (disc < 0.0) disc = 0.0;
                    double alpha = (-b + std::sqrt(disc)) / (2.0 * a);
                    h += alpha * p;
                    break;
                }

                double rq = r.dot(q);
                double alpha = rq / pKp;

                // Check if proposed step exceeds trust region
                Eigen::Vector2d h_trial = h + alpha * p;
                if (h_trial.squaredNorm() >= TR_radius * TR_radius) {
                    flag_boundary_reached = true;
                    double a = p.squaredNorm();
                    double b = 2.0 * p.dot(h);
                    double c = h.squaredNorm() - TR_radius * TR_radius;
                    double disc = b * b - 4.0 * a * c;
                    if (disc < 0.0) disc = 0.0;
                    double alpha_boundary = (-b + std::sqrt(disc)) / (2.0 * a);
                    h += alpha_boundary * p;
                    break;
                }

                h = h_trial;
                double phi = r.dot(p);
                r -= alpha * (K * p);

                double norm_r = r.norm();
                double norm_f = f.norm();
                double tol_r = std::max(1e-15, 1e-5 * norm_f);
                if (norm_r < tol_r) {
                    break;
                }

                q = r;
                double beta = r.dot(q) / phi;
                p = q + beta * p;
            } // end CG loop

            // Evaluate new objective at t + h
            Eigen::Vector2d t_new = t + h;
            Eigen::Vector3d xc_new = Grg(CtrlPts, t_new.x(), t_new.y(), eps);
            double m_new_plus_h = (xs - xc_new).squaredNorm();

            // Predicted reduction from quadratic model
            double hKh = (h.transpose() * K * h)(0,0);
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
                t = t_new;
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
py::tuple find_projection_tr_multi(const Eigen::MatrixXd &CtrlPtsAll,
                                   const Eigen::Vector3d &xs,
                                   py::array_t<int> candidate_indices,
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

    double best_m = std::numeric_limits<double>::infinity();
    Eigen::Vector2d best_t(-1.0, -1.0);
    int best_patch = -1;

    for (int k = 0; k < nCand; ++k) {
        int p_id = idx_ptr[k];
        if (p_id < 0 || p_id >= nPatches) {
            throw std::runtime_error("candidate index out of range");
        }
        // View of CtrlPts for this patch: 20x3 block
        Eigen::MatrixXd CtrlPts = CtrlPtsAll.block(p_id * rows_per_patch, 0, rows_per_patch, 3);
        TRProjResult res = projection_tr_core(CtrlPts, xs, eps, TR_init, TR_min, TR_max);
        if (res.m < best_m) {
            best_m = res.m;
            best_t = res.t;
            best_patch = p_id;
        }
    }

    return py::make_tuple(best_patch, best_t.x(), best_t.y(), best_m);
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

// FEAssembly m_el_extra function - rigorous translation of Python hyperelastic SED computation
Eigen::VectorXd m_el_extra(const Eigen::MatrixXd &X_hexa, const Eigen::MatrixXd &u_hexa, 
                          double Youngsmodulus, double Poissonsratio) {
    // X_hexa: 8x3 matrix of hexahedron node coordinates
    // u_hexa: 8x3 matrix of displacements (matches Python u[self.DoFs[hexa]] output)
    
    // Exact translation of Python material parameters
    double d1 = Youngsmodulus * Poissonsratio / (2 * (1 + Poissonsratio) * (1 - 2 * Poissonsratio));
    double c10 = Youngsmodulus / (4 * (1 + Poissonsratio));
    
    // Gauss points - exact translation of Python array
    Eigen::MatrixXd gauss_points(8, 3);
    double gp = 1.0 / std::sqrt(3.0);
    gauss_points << -gp, -gp, -gp,
                     gp, -gp, -gp,
                     gp,  gp, -gp,
                    -gp,  gp, -gp,
                    -gp, -gp,  gp,
                     gp, -gp,  gp,
                     gp,  gp,  gp,
                    -gp,  gp,  gp;
    
    Eigen::VectorXd SED(8);
    Eigen::MatrixXd NN(8, 8);
    
    // u_hexa is already in 8x3 format from Python u[self.DoFs[hexa]]
    
    // Loop over 8 Gauss points - exact translation of Python logic
    for (int g_i = 0; g_i < 8; ++g_i) {
        double g1 = gauss_points(g_i, 0);
        double g2 = gauss_points(g_i, 1);  
        double g3 = gauss_points(g_i, 2);
        
        // Shape function derivatives - exact translation of Python dNd_xi
        Eigen::MatrixXd dNd_xi(8, 3);
        dNd_xi << -(1 - g2) * (1 - g3), -(1 - g1) * (1 - g3), -(1 - g1) * (1 - g2),
                   (1 - g2) * (1 - g3), -(1 + g1) * (1 - g3), -(1 + g1) * (1 - g2),
                   (1 + g2) * (1 - g3),  (1 + g1) * (1 - g3), -(1 + g1) * (1 + g2),
                  -(1 + g2) * (1 - g3),  (1 - g1) * (1 - g3), -(1 - g1) * (1 + g2),
                  -(1 - g2) * (1 + g3), -(1 - g1) * (1 + g3),  (1 - g1) * (1 - g2),
                   (1 - g2) * (1 + g3), -(1 + g1) * (1 + g3),  (1 + g1) * (1 - g2),
                   (1 + g2) * (1 + g3),  (1 + g1) * (1 + g3),  (1 + g1) * (1 + g2),
                  -(1 + g2) * (1 + g3),  (1 - g1) * (1 + g3),  (1 - g1) * (1 + g2);
        dNd_xi *= 1.0/8.0;
        
        // Jacobian computation - exact translation: J = np.dot(dNd_xi.T, X)
        Eigen::Matrix3d J = dNd_xi.transpose() * X_hexa;
        
        // Jacobian inverse - exact translation: invJ = np.linalg.inv(J)
        Eigen::Matrix3d invJ = J.inverse();
        
        // Global derivatives - exact translation: dNdx = np.dot(dNd_xi, invJ.T)
        Eigen::MatrixXd dNdx = dNd_xi * invJ.transpose();
        
        // Deformation gradient - exact translation: F = np.eye(len(dNdx.T)) + np.dot(dNdx.T, u).T
        Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + (dNdx.transpose() * u_hexa).transpose();
        
        // Determinant - exact translation: detF = np.linalg.det(F)
        double detF = F.determinant();
        
        // SED computation - exact translation of Python hyperelastic formula
        // SED[g_i] = c10 * (np.trace(F.T @ F) - 3 - 2 * np.log(detF)) + d1 * (np.log(detF))**2
        double trace_FtF = (F.transpose() * F).trace();
        double log_detF = std::log(detF);
        SED(g_i) = c10 * (trace_FtF - 3.0 - 2.0 * log_detF) + d1 * log_detF * log_detF;
        
        // Shape functions for extrapolation - exact translation of Python NN[g_i]
        NN(g_i, 0) = (1 - g1) * (1 - g2) * (1 - g3);
        NN(g_i, 1) = (1 + g1) * (1 - g2) * (1 - g3);
        NN(g_i, 2) = (1 + g1) * (1 + g2) * (1 - g3);
        NN(g_i, 3) = (1 - g1) * (1 + g2) * (1 - g3);
        NN(g_i, 4) = (1 - g1) * (1 - g2) * (1 + g3);
        NN(g_i, 5) = (1 + g1) * (1 - g2) * (1 + g3);
        NN(g_i, 6) = (1 + g1) * (1 + g2) * (1 + g3);
        NN(g_i, 7) = (1 - g1) * (1 + g2) * (1 + g3);
        NN.row(g_i) *= 1.0/8.0;
    }
    
    // Final extrapolation - exact translation: return SED@np.linalg.inv(NN)
    return SED.transpose() * NN.inverse();
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
    m.def("D3Grg", &D3Grg, "Calculate the normal vector at (u,v) on Gregory patch", py::arg("CtrlPts"), py::arg("u"), py::arg("v"), py::arg("eps"), py::arg("normalize")=true);
    m.def("ContainsNode", &ContainsNode, "Check if a point is contained within a bounding sphere");
    m.def("ContainsNodes", &ContainsNodes, "Vectorized check if multiple points are contained within a bounding sphere");
    m.def("m_el_extra", &m_el_extra, "Compute strain energy density at nodes for hyperelastic material");
}
