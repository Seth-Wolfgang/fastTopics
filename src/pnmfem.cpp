#include <RcppParallel.h>
#include "misc.h"
#include "poismix.h"

using namespace arma;

// INLINE FUNCTION DEFINITIONS
// ---------------------------
// Perform one or several EM updates for a single column of the k x m
// factors matrix, Factors, when X is a dense matrix.
inline vec pnmfem_update_factor(const mat& X, const mat& Factors, const mat& L1,
                                const vec& u, mat& P, unsigned int j,
                                unsigned int numiter) {
    vec factor = Factors.col(j);
    poismixem(L1, u, X.col(j), factor, P, numiter);
    return factor;
}

// Perform one or several EM updates for a single column of the k x m
// factors matrix, Factors, when X is a sparse matrix.
inline vec pnmfem_update_factor_sparse(const sp_mat& X, const mat& Factors,
                                       const mat& L1, const vec& u,
                                       unsigned int j, unsigned int numiter) {
    vec          x = nonzeros(X.col(j));
    unsigned int n = x.n_elem;
    vec          factor = Factors.col(j);
    uvec         i(n);
    getcolnonzeros(X, i, j);
    poismixem(L1, u, x, i, factor, numiter);
    return factor;
}

// CLASS DEFINITIONS
// -----------------
// This class is used to implement multithreaded computation of the
// factor updates in pnmfem_update_factors_parallel_rcpp.
struct pnmfem_factor_updater : public RcppParallel::Worker {
    const mat& X;
    const mat& Factors;
    mat          L1;
    vec          u;
    mat& Fnew;
    const vec& j;
    unsigned int numiter;

    // This is used to create a pnmfem_factor_updater object.
    pnmfem_factor_updater(const mat& X, const mat& Factors, const mat& Loadings,
                          mat& Fnew, const vec& j, unsigned int numiter) : // This is a constructor
        X(X), Factors(Factors), L1(Loadings), u(Loadings.n_cols), Fnew(Fnew), j(j), numiter(numiter) {
        u = sum(Loadings, 0);
        normalizecols(L1);
    };

    // This function updates the factors for a given range of columns.
    void operator() (std::size_t begin, std::size_t end) {
        mat P = L1;
        for (unsigned int i = begin; i < end; i++)
            Fnew.col(j(i)) = pnmfem_update_factor(X, Factors, L1, u, P, j(i), numiter);
    }
};

// This class is used to implement multithreaded computation of the
// factor updates in pnmfem_update_factors_sparse_parallel_rcpp.
struct pnmfem_factor_updater_sparse : public RcppParallel::Worker {
    const sp_mat& X;
    const mat& Factors;
    mat           L1;
    vec           u;
    mat& Fnew;
    const vec& j;
    unsigned int  numiter;

    // This is used to create a pnmfem_factor_updater object.
    pnmfem_factor_updater_sparse(const sp_mat& X, const mat& Factors, const mat& Loadings,
                                 mat& Fnew, const vec& j,
                                 unsigned int numiter) : // This is a constructor
        X(X), Factors(Factors), L1(Loadings), u(Loadings.n_cols), Fnew(Fnew), j(j), numiter(numiter) {
        u = sum(Loadings, 0);
        normalizecols(L1);
    };

    // This function updates the factors for a given range of columns.
    void operator() (std::size_t begin, std::size_t end) {
        for (unsigned int i = begin; i < end; i++)
            Fnew.col(j(i)) = pnmfem_update_factor_sparse(X, Factors, L1, u, j(i), numiter);
    }
};

// FUNCTION DEFINITIONS
// --------------------
// Perform one or several EM updates for the factors matrix, Factors, in
// which the matrix X is approximated by Loadings*Factors. Input "k" specifies
// which columns of Factors are updated. Input "numiter" specifies the
// number of EM updates to perform.
//
// Note that, unlike most other functions implemented in this package,
// the factors matrix Factors should be a k x m matrix, where k is the
// number factors, or "topics". This is done for ease of
// implementation, and for speed, because the computation is performed
// on the m columns of Factors, and the matrix is stored columnwise.
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat pnmfem_update_factors_rcpp(const arma::mat& X, const arma::mat& Factors,
                                     const arma::mat& Loadings, const arma::vec& j,
                                     double numiter) {
    unsigned int n = j.n_elem;
    vec  u = sum(Loadings, 0);
    mat  L1 = Loadings;
    mat  P = Loadings;
    mat  Fnew = Factors;
    normalizecols(L1);
    for (unsigned int i = 0; i < n; i++)
        Fnew.col(j(i)) = pnmfem_update_factor(X, Factors, L1, u, P, j(i), numiter);
    return Fnew;
}

// This does the same thing as pnmfem_update_factors_rcpp, except that
// X is a sparse matrix. See pnmfem_update_factors_rcpp for details.
//
// [[Rcpp::export]]
arma::mat pnmfem_update_factors_sparse_rcpp(const arma::sp_mat& X,
                                            const arma::mat& Factors,
                                            const arma::mat& Loadings,
                                            const arma::vec& j,
                                            double numiter) {
    unsigned int n = j.n_elem;
    vec  u = sum(Loadings, 0);
    mat  L1 = Loadings;
    mat  Fnew = Factors;
    normalizecols(L1);
    for (unsigned int i = 0; i < n; i++)
        Fnew.col(j(i)) = pnmfem_update_factor_sparse(X, Factors, L1, u, j(i), numiter);
    return Fnew;
}

// This does the same thing as pnmfem_update_factors_rcpp, except that
// Intel Threading Building Blocks (TBB) are used to update the
// factors in parallel.
//
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::export]]
arma::mat pnmfem_update_factors_parallel_rcpp(const arma::mat& X,
                                              const arma::mat& Factors,
                                              const arma::mat& Loadings,
                                              const arma::vec& j,
                                              double numiter) {
    mat newFactors = Factors;
    pnmfem_factor_updater worker(X, Factors, Loadings, newFactors, j, numiter);
    parallelFor(0, j.n_elem, worker);
    return newFactors;
}

// This does the same thing as pnmfem_update_factors_sparse_rcpp,
// except that Intel Threading Building Blocks (TBB) are used to
// update the factors in parallel.
//
// [[Rcpp::export]]
arma::mat pnmfem_update_factors_sparse_parallel_rcpp(const arma::sp_mat& X,
                                                     const arma::mat& Factors,
                                                     const arma::mat& Loadings,
                                                     const arma::vec& j,
                                                     double numiter) {
    mat newFactors = Factors;
    pnmfem_factor_updater_sparse worker(X, Factors, Loadings, newFactors, j, numiter);
    parallelFor(0, j.n_elem, worker);
    return Fnew;
}
