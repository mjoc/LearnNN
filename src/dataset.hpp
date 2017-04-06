//
//  dataset.hpp
//  BasicNN
//
//  Created by Martin on 31/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef __MOC_dataset_hpp_
#define __MOC_dataset_hpp_

#include <vector>

#ifndef IGNORE_THIS_RCPP_CODE
#include "Rcpp.h"

class Dataset; // fwd declarations
RCPP_EXPOSED_CLASS(Dataset);

#endif

class Dataset{
public:
  enum normDataType
  {
    DATA_NORM_NONE,
    DATA_STAN_NORM,
    DATA_RANGE_BOUND
  };

private:
  bool _dataLoaded;
  bool _labelsLoaded;

  size_t _nRecords;
  size_t _nFields;
  size_t _nLabelFields;
  size_t _nInputFields;

  bool _pcaDone;
  std::vector<double> _data;
  std::vector<double> _labels;

  // preprocessing/changes to the input data
  bool _pcaEigenMatLoaded;
  std::vector<double> _pcaEigenMat;
  size_t _nPcaDimensions;
  normDataType _normType;
  std::vector<double> _normParam1;
  std::vector<double> _normParam2;

  bool loadDataFromFile(const char *filename,
                        bool hasHeader,
                        char delim);
  bool loadLabelsFromFile(const char *filename,
                          bool hasHeader,
                          char delim);

  std::string _dataOriginFile;
  std::string _labelsOriginFile;
  // Writing out to file
  std::string _outputDir;


public:
  Dataset(const char *datafilename, const char *labelfilename, bool hasHeader, const char* delim);
  // For R
    ~Dataset();
  // Load data

  size_t nRecords() const;
  size_t nFields() const;
  size_t nLabelFields() const;
  bool dataLoaded() const;
  bool labelsLoaded() const;

  void analyseAndNorm(std::string normType);
  void normWithParams(std::string normType, const  std::vector<double>& params1, const std::vector<double>& params2);
  void normWithPrototype(Dataset& otherDataset);


  void doPca(size_t dimensions = 0);
  void doPcaProjection(std::vector<double> pcaEigenMat,
                  size_t nPcaDimension);

  void doPcaFromDataset(const Dataset& otherDataset);

  bool isPcaDone() const;

  void transformDataset(Dataset& otherDataset);

  void printData(int nRecords);

  void printLabels(int nRecords = 10);

  // Write to file
  void setOutputFolder(char *filename);
  void writeData();


  std::vector<double>* data();
  std::vector<double>* labels();
  std::vector<double> getPcaMatrix() const;

  std::string getNormType() const;
  std::vector<double> getNormParam1() const;
  std::vector<double> getNormParam2() const;

// Rcpp stuff
  #ifndef IGNORE_THIS_RCPP_CODE

  Dataset(Rcpp::String datafilename, Rcpp::String labelfilename, Rcpp::LogicalVector hasHeader, Rcpp::String delim);
  Dataset(Rcpp::NumericMatrix indata, Rcpp::Nullable<Rcpp::NumericMatrix> labels);
  
  SEXP getDataR();
  SEXP getLabelsR();
  SEXP getPcaMatrixR();
  SEXP getNormParamMat() const;

  #endif

};



#endif /* dataset_hpp */
