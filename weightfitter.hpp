//
//  weightfitter.hpp
//  BasicNN
//
//  Created by Martin on 30/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef weightfitter_hpp
#define weightfitter_hpp


//#include <cstring>
#include "nnet.hpp"

class weightfitter{
  
protected:
  
  
  
public:
  weightfitter();
  ~weightfitter();
  
  
  
  
 
  
  virtual bool fitWeights(nnet netToFit) = 0;
};

#endif /* weightfitter_hpp */
