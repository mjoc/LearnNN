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
#include "Rcpp.h"

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

  std::string _dataSource;
  std::string _labelsSource;

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

  bool loadDataFromFile(char *filename,
                        bool hasHeader,
                        char delim);
  bool loadLabelsFromFile(char *filename,
                          bool hasHeader,
                          char delim);

  // Writing out to file
  std::string _outputDir;


public:
  Dataset(char *datafilename, char *labelfilename, bool hasHeader, char delim);
  // For R
  Dataset(Rcpp::NumericMatrix data, Rcpp::NumericMatrix labels);
  ~Dataset();
  // Load data

  size_t nRecords() const;
  size_t nFields() const;
  size_t nLabelFields() const;
  bool dataLoaded() const;
  bool labelsLoaded() const;

  void analyseAndNorm(normDataType normType);
  void normFromParams(const normDataType normType,const  std::vector<double>& params1, const std::vector<double>& params2);
  void normFromDataset(const Dataset& otherDataset);

  void doPca(size_t dimensions = 0);
  void doPcaProjection(std::vector<double> pcaEigenMat,
                  size_t nPcaDimension);

  void doPcaFromDataset(const Dataset& otherDataset);

  bool isPcaDone() const;

  void transformDataset(const Dataset& otherDataset);

  void printData(size_t nRecords = 0);

  void printLabels(size_t nRecords = 0);

  // Write to file
  void setOutputFolder(char *filename);
  void writeData();


  std::vector<double> data();
  std::vector<double> labels();
  std::vector<double> getPcaMatrix() const;

  normDataType getNormType() const;
  std::vector<double> getNormParam1() const;
  std::vector<double> getNormParam2() const;

};



#endif /* dataset_hpp */
