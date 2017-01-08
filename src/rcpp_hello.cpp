#include <Rcpp.h>
using namespace Rcpp;

// This is a simple function using Rcpp that creates an R list
// containing a character vector and a numeric vector.
//
// Learn more about how to use Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//
// and browse examples of code using Rcpp at:
//
//   http://gallery.rcpp.org/
//

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::export]]
List rcpp_hello(NumericVector a, NumericVector b) {
  LogicalVector allOk = {1};

  List z;

  if(a.length() != 1 || b.length() != 1){
    warning("Warning: one of your arguments is not of length 1");
    allOk[0] = FALSE;
  }

  if(allOk[0]==TRUE){
    CharacterVector x = CharacterVector::create("foo", "bar");
    NumericVector y   = NumericVector::create(0.0, 1.0);
    z.push_back(x);
    z.push_back(y);
  }else{
    NumericVector y   = NumericVector::create(2.0, 2.0);
    z.push_back(y);
  }
  CharacterVector w = CharacterVector::create("done");
  z.push_back(w);

  return z;
}
