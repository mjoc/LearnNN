//
//  mat_ops.hpp
//  BasicNN
//
//  Created by Martin on 20/11/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef __MOC_mat_ops_hpp_
#define __MOC_mat_ops_hpp_

#include <gsl/gsl_matrix.h>

class mat_ops{
  static gsl_matrix* gsl_pca(const gsl_matrix* data, unsigned int L , gsl_matrix* eigenVectors);
  static gsl_matrix* gsl_pca_project(const gsl_matrix* data, unsigned int L, gsl_matrix* eigenVectors);
  
public:
  static void writeMatrix(std::string filePathAndName, std::vector<double> outMat, size_t nRows, size_t nCols);
  static void matMul(size_t Arows, size_t Acols, std::vector<double>& A, size_t Bcols, std::vector<double>& B, std::vector<double>& C);
  static void pca(std::vector<double>& inMat, size_t nInMatCols, size_t nPcaMatCols, std::vector<double>&eigenMat);
  static void pcaProject(std::vector<double>& inMat, size_t nInMatCols, size_t nPcaMatCols, std::vector<double>& eigenMat);
};

#endif /* mat_ops_hpp */
