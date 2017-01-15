#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
LogicalVector dataclamp(SEXP input) {
  LogicalVector allOk = {1};
  LogicalVector typeOk = {0};
  size_t nRows = 0, nCols = 0;

  if(is<NumericVector>(input)){
    typeOk[0] = TRUE;
    if(Rf_isMatrix(input)){
      NumericMatrix mat = as<NumericMatrix>(input);
      nRows = mat.nrow();
      nCols = mat.ncol();
      Rcout << "Matrix of size: " << nRows << " x " << nCols<< std::endl;
    }else{
      NumericVector vec = as<NumericVector>(input);
      nRows = 1;
      nCols = vec.length();
      Rcout << "Vector of length: " << nCols << std::endl;
    };
  }
  if(is<DataFrame>(input)){
    DataFrame df = as<DataFrame>(input);
    nRows = df.nrows();
    nCols = df.size();
    Rcout << "Dataframe of size: " << nRows << " x " << nCols << std::endl;
  }

  return allOk;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
dataclamp(42)
*/
