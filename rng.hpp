#ifndef __MOC_Rng_
#define __MOC_Rng_


#include <vector>
#include <gsl/gsl_rng.h>

class rng{
    gsl_rng *_rng;
    
public:
    rng();
    rng(long seed);
    ~rng();
    void getUniformVector(std::vector<double>& dataToFill);
    void getGaussianVector(std::vector<double>& dataToFill, double stdev);
    void setSeed(long seed);
};

#endif
