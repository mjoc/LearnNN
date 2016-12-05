#include <iostream>
#include "rng.hpp"
#include <gsl/gsl_randist.h>

rng::rng(){
    _rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(_rng, 12345);
}

rng::rng(long seed){
    _rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(_rng, seed);
}

void rng::setSeed(long seed){
    gsl_rng_set(_rng, seed);
}

rng::~rng(){
    gsl_rng_free (_rng);
}

void rng::getUniformVector(std::vector<double>& dataToFill){
    for(std::vector<double>::iterator it = dataToFill.begin(); it != dataToFill.end(); ++it) {
	*it = gsl_rng_uniform(_rng);
    }
    return;
}

void rng::getGaussianVector(std::vector<double>& dataToFill, double stdev){
    for(std::vector<double>::iterator it = dataToFill.begin(); it != dataToFill.end(); ++it) {
	*it = gsl_ran_gaussian(_rng, stdev);
    }
    return;
}

void rng::getShuffled(std::vector<int>& indata){
  
  gsl_ran_shuffle (_rng, indata.data(), indata.size(), sizeof (int));
  
  return;
}
