//
//  mat_ops.cpp
//  BasicNN
//
//  Created by Martin on 20/11/2016.
//  Copyright © 2016 Martin. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <gsl/gsl_cblas.h>
#include "mat_ops.hpp"

void mat_ops::matMul(size_t Arows, size_t Acols, std::vector<double>& A, size_t Bcols, std::vector<double>& B, std::vector<double>& C){
  double alpha = 1.0;
  double beta = 1.0;
  if(A.size() != Arows*Acols){
    std::cout << "***** A rows and columns misspecified\n";
  }
  if(B.size() != Acols*Bcols){
    std::cout << "***** B rows and columns misspecified\n";
  }
  if(C.size() != Arows*Bcols){
    std::cout << "***** C is of the wrong size\n";
  }
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)Arows, (int)Bcols, (int)Acols, alpha, A.data(), (int)Acols, B.data(), (int)Bcols, beta, C.data(), (int)Bcols);
}

void mat_ops::writeMatrix(std::string filename, std::vector<double> outMat, size_t nRows, size_t nCols){
  std::ofstream myfile;
  myfile.open (filename);
  for(int i = 0; i < nRows; i++){
    for(int j = 0; j < nCols; j++){
      myfile << outMat[i *nCols + j] ;
      if(j < nCols - 1){
        myfile << ", ";
      }
      
    }
    myfile << std::endl;
  }
  
  myfile.close();
  return;
  
}
