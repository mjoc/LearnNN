//
//  mat_ops.cpp
//  BasicNN
//
//  Created by Martin on 20/11/2016.
//  Copyright © 2016 Martin. All rights reserved.
//
#include <vector>
#include <string>
#include <fstream>
#include <gsl/gsl_cblas.h>
#include <assert.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <iomanip>
#include <sstream>
#include "mat_ops.hpp"
#include "message.hpp"


bool mat_ops::matMul(size_t Arows, size_t Acols, std::vector<double>& A, size_t Bcols, std::vector<double>& B, std::vector<double>& C){
  bool allOk = true;
  std::ostringstream message;

  double alpha = 1.0;
  double beta = 1.0;
  size_t ldA = Arows, ldB = Acols, ldC = Arows;
  if(A.size() != Arows*Acols){
    message << "DGEMM matrix multiply A rows and columns misspecified. Size of A is: " << A.size() << " not " << Arows << " by " << Acols << std::endl;
    msg::error(message);
    allOk = false;
  }
  if(B.size() != Acols*Bcols){
    message << "DGEMM matrix multiply B rows and columns misspecified. Size of B is: " << B.size() << " not " << Acols << " by " << Bcols << std::endl;
    msg::error(message);
    allOk = false;
  }
  if(C.size() != Arows*Bcols){
    message << "DGEMM matrix multiply C rows and columns misspecified. Size of C is: " << C.size() << " not " << Arows << " by " << Bcols << std::endl;
    msg::error(message);
    allOk = false;
  }
/*
  message << "A: ";
  for(size_t i = 0; i <A.size(); i++ ){
    message << A[i] << " ";
  }
  message << std::endl;
  msg::info(message);

  message << "B: ";
  for(size_t i = 0; i <B.size(); i++ ){
    message << B[i] << " ";
  }
  message << std::endl;
  msg::info(message);

  message << "C: ";
  for(size_t i = 0; i <C.size(); i++ ){
    message << C[i] << " ";
  }
  message << std::endl;
  msg::info(message);

  message << "Arows: " << Arows << " Bcols " << Bcols << " Acols " << std::endl;
 msg::info(message);
*/

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              (int)Arows, (int)Bcols, (int)Acols, alpha, A.data(), (int)ldA, B.data(), (int)ldB, beta, C.data(), (int)ldC);

  return allOk;
}

void mat_ops::pca(std::vector<double>& inMat, size_t nInMatCols, size_t nPcaMatCols, std::vector<double>& eigenMat){
  size_t nRows = inMat.size()/nInMatCols;
  size_t nCols = nInMatCols;
  size_t nGslRows = nCols;
  size_t nGslCols = nRows;


  gsl_matrix *inMatrix = gsl_matrix_alloc(nGslRows , nGslCols);
  gsl_matrix *outMatrix;
  gsl_matrix* eigenvectors = gsl_matrix_alloc(nCols, nCols);


  for(size_t iRow = 0; iRow < nRows; iRow++){
    for (size_t iCol = 0; iCol < nCols; iCol++){
      gsl_matrix_set (inMatrix, iCol, iRow, inMat[(iCol * nRows) + iRow]);
    }
  }
  outMatrix = gsl_pca(inMatrix,(int)nPcaMatCols,eigenvectors);

  inMat.assign(nRows*nPcaMatCols,0.0);
  for(size_t iRow = 0; iRow < nRows; iRow++){
    for (size_t iCol = 0; iCol < nPcaMatCols; iCol++){
      inMat[(iCol*nRows) + iRow] = gsl_matrix_get (outMatrix, iCol, iRow);
    }
  }

  eigenMat.assign(nCols*nCols,0.0);
  for(size_t iRow = 0; iRow < nCols; iRow++){
    for (size_t iCol = 0; iCol < nCols; iCol++){
      eigenMat[(iCol*nCols) + iRow] = gsl_matrix_get (eigenvectors, iCol, iRow);
    }
  }


  gsl_matrix_free (inMatrix);
  gsl_matrix_free (outMatrix);
  gsl_matrix_free(eigenvectors);

}



void mat_ops::pcaProject(std::vector<double>& inMat, size_t nInMatCols, size_t nPcaMatCols, std::vector<double>& eigenMat){
  size_t nRows = inMat.size()/nInMatCols;
  size_t nCols = nInMatCols;
  size_t nGslRows = nCols;
  size_t nGslCols = nRows;


  gsl_matrix *inMatrix = gsl_matrix_alloc(nGslRows , nGslCols);
  gsl_matrix *outMatrix;
  gsl_matrix* eigenMatrix = gsl_matrix_alloc(nCols, nCols);


  for(size_t iRow = 0; iRow < nRows; iRow++){
    for (size_t iCol = 0; iCol < nCols; iCol++){
      gsl_matrix_set (inMatrix, iCol, iRow, inMat[(iCol * nRows) + iRow]);
    }
  }

  for(size_t iRow = 0; iRow < nCols; iRow++){
    for (size_t iCol = 0; iCol < nCols; iCol++){
      gsl_matrix_set (eigenMatrix, iCol, iRow, eigenMat[(iCol * nCols) + iRow]);
    }
  }

  outMatrix = gsl_pca_project(inMatrix,(int)nPcaMatCols,eigenMatrix);

  inMat.assign(nRows*nPcaMatCols,0.0);
  for(size_t iRow = 0; iRow < nRows; iRow++){
    for (size_t iCol = 0; iCol < nPcaMatCols; iCol++){
      inMat[(iCol * nRows) + iRow] = gsl_matrix_get (outMatrix, iCol, iRow);
    }
  }

  gsl_matrix_free (inMatrix);
  gsl_matrix_free (outMatrix);
  gsl_matrix_free(eigenMatrix);

}

gsl_matrix* mat_ops::gsl_pca(const gsl_matrix* data, unsigned int L, gsl_matrix* eigenvectors)
{
  /*
   @param data - matrix of data vectors, MxN matrix, each column is a data vector, M - dimension, N - data vector count
   @param L - dimension reduction
   */
  assert(data != NULL);
  assert(L > 0 && L < data->size2);
  unsigned int i;
  size_t rows = data->size1;
  size_t cols = data->size2;
  gsl_vector* mean = gsl_vector_alloc(rows);

  for(i = 0; i < rows; i++) {
    gsl_vector_set(mean, i, gsl_stats_mean(data->data + i * cols, 1, cols));
  }

  // Get mean-substracted data into matrix mean_substracted_data.
  gsl_matrix* mean_substracted_data = gsl_matrix_alloc(rows, cols);
  gsl_matrix_memcpy(mean_substracted_data, data);
  for(i = 0; i < cols; i++) {
    gsl_vector_view mean_substracted_point_view = gsl_matrix_column(mean_substracted_data, i);
    gsl_vector_sub(&mean_substracted_point_view.vector, mean);
  }
  gsl_vector_free(mean);

  // Compute Covariance matrix
  gsl_matrix* covariance_matrix = gsl_matrix_alloc(rows, rows);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0 / (double)(cols - 1), mean_substracted_data, mean_substracted_data, 0.0, covariance_matrix);
  gsl_matrix_free(mean_substracted_data);

  // Get eigenvectors, sort by eigenvalue.
  gsl_vector* eigenvalues = gsl_vector_alloc(rows);
  //gsl_matrix* eigenvectors = gsl_matrix_alloc(rows, rows);
  gsl_eigen_symmv_workspace* workspace = gsl_eigen_symmv_alloc(rows);
  gsl_eigen_symmv(covariance_matrix, eigenvalues, eigenvectors, workspace);
  gsl_eigen_symmv_free(workspace);
  gsl_matrix_free(covariance_matrix);

  // Sort the eigenvectors
  gsl_eigen_symmv_sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_ABS_DESC);
  gsl_vector_free(eigenvalues);

  // Project the original dataset
  gsl_matrix* result = gsl_matrix_alloc(L, cols);
  gsl_matrix_view L_eigenvectors = gsl_matrix_submatrix(eigenvectors, 0, 0, rows, L);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &L_eigenvectors.matrix, data, 0.0, result);
  //gsl_matrix_free(eigenvectors);

  // Result is n LxN matrix, each column is the original data vector with reduced dimension from M to L
  return result;
}


// This code was copied from microo8 to speed up development, I might go back and recode for fun
// Altered a bit to keep the eigenvector matrix for projetion of other data (e.g. test)
// PCA is basically the EigenDecompostion of covariance matrix
gsl_matrix* mat_ops::gsl_pca_project(const gsl_matrix* data, unsigned int L, gsl_matrix* eigenVectors)
{
  /*
   @param data - matrix of data vectors, MxN matrix, each column is a data vector, M - dimension, N - data vector count
   @param L - dimension reduction
   */
  assert(data != NULL);
  assert(L > 0 && L < data->size2);
  size_t rows = data->size1;
  size_t cols = data->size2;

  // Project the original dataset
  gsl_matrix* result = gsl_matrix_alloc(L, cols);
  gsl_matrix_view L_eigenvectors = gsl_matrix_submatrix(eigenVectors, 0, 0, rows, L);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &L_eigenvectors.matrix, data, 0.0, result);
  //gsl_matrix_free(eigenvectors);

  // Result is n LxN matrix, each column is the original data vector with reduced dimension from M to L
  return result;
}

void mat_ops::writeMatrix(std::string filename, std::vector<double> outMat, size_t nRows, size_t nCols){
  std::ofstream myfile;
  myfile.open (filename);
  for(size_t iRow = 0; iRow < nRows; iRow++){
    for(size_t iCol = 0; iCol < nCols; iCol++){
      myfile << outMat[(iCol *nRows) + iRow] ;
      if(iCol < nCols - 1){
        myfile << ", ";
      }

    }
    myfile << std::endl;
  }

  myfile.close();
  return;

}






