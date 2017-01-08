//
//  dataset.cpp
//  BasicNN
//
//  Created by Martin on 31/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#include "dataset.hpp"
#include "mat_ops.hpp"
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

#define MIN_DATA_RANGE 1e-4

dataset::dataset(char *dataFileName, char *labelsFileName, bool hasHeader, char delim){
  bool allOk = true;
  _dataLoaded = false;
  _labelsLoaded = false;
  _nRecords = 0;
  _nFields = 0;
  _nLabelFields = 0;
  _nInputFields = 0;
  
  _normType = DATA_NORM_NONE;
  _pcaDone = false;
  _pcaEigenMatLoaded = false;
  _nPcaDimensions = 0;
  allOk = loadDataFromFile(dataFileName, hasHeader, delim);
  if(allOk){
    allOk = loadLabelsFromFile(labelsFileName, hasHeader, delim);
  }
  if(allOk){
    _dataLoaded = true;
    _dataSource = std::string(dataFileName);
    _labelsSource = std::string(labelsFileName);
  }
  
  _outputDir = "~/";
}

dataset::~dataset(){
}

void dataset::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}

dataset::normDataType dataset::getNormType() const {
  return _normType;
}

std::vector<double> dataset::getNormParam1() const {
  return _normParam1;
}

std::vector<double> dataset::getNormParam2() const {
  return _normParam2;
}


bool dataset::dataLoaded() const{
  return _dataLoaded;
}

bool dataset::labelsLoaded() const{
  return _labelsLoaded;
}


bool dataset::loadDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  
  std::vector<double> indata;
  
  _labelsLoaded = false;
  _dataLoaded = false;
  
  _nRecords = 0;
  
  _normType = DATA_NORM_NONE;
  _pcaDone = false;
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
    std::cout << "Reading in data from file " << filename << std::endl;
    _data.resize(0);
    if(hasHeader){
      std::string headerline;
      std::getline(infile, headerline);
      nDelims = std::count(headerline.begin(), headerline.end(), delim);
      first = false;
    }
    // http://stackoverflow.com/questions/18818777/c-program-for-reading-an-unknown-size-csv-file-filled-only-with-floats-with
    
    for (std::string line; std::getline(infile, line); )
    {
      if(line.find_first_not_of(' ') == std::string::npos){
        break;
      }else{
        nRecords++;
        if(first){
          nDelims = std::count(line.begin(), line.end(), delim);
          first = false;
        }else{
          if(nDelims != std::count(line.begin(), line.end(), delim)){
            std::cout << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
            allOk = false;
          }
        }
        
        std::replace(line.begin(), line.end(), delim, ' ');
        std::istringstream in(line);
        
        
        if(allOk){
          std::vector<double> rowValues = std::vector<double>(std::istream_iterator<double>(in), std::istream_iterator<double>());
          for (std::vector<double>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
            indata.push_back(*it);
            
          }
          if((indata.size()% (nDelims+1)) != 0 ){
            std::cout << "Number of data is not an integer multiple of first line fields!" << std::endl;
            allOk = false;
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    _nFields = nDelims + 1;
    _nInputFields = _nFields;
    _nRecords = nRecords;
    
    _data.resize(indata.size());
    
    for(size_t iCol = 0; iCol < _nFields; iCol++){
      for(size_t iRow = 0; iRow < _nRecords; iRow++){
         //Switching from row major to column major
        _data[iCol*_nRecords + iRow] = indata[iRow*_nFields + iCol];
      }
    }
    _dataLoaded = true;
    
    
    std::cout << "Read " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool dataset::loadLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  size_t nRecords = 0;
  size_t nFields = 0;
  size_t nDelims = 0;
  std::vector<double> indata;
  
  _labelsLoaded = false;
  _labels.resize(0);
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if(!_dataLoaded){
    allOk = false;
  }
  
  if (!infile.is_open()){
    std::cout << "Cannot open file, giving up!\n";
    allOk = false;
  }
  
  if(allOk){
    std::cout << "Reading in data from file " << filename << std::endl;
    if(hasHeader){
      std::string headerline;
      std::getline(infile, headerline);
      nDelims = std::count(headerline.begin(), headerline.end(), delim);
      first = false;
    }
    // http://stackoverflow.com/questions/18818777/c-program-for-reading-an-unknown-size-csv-file-filled-only-with-floats-with
    
    for (std::string line; std::getline(infile, line); )
    {
      if(line.find_first_not_of(' ') == std::string::npos){
        break;
      }else{
        nRecords++;
        if(first){
          nDelims = std::count(line.begin(), line.end(), delim);
          first = false;
        }else{
          if(nDelims != std::count(line.begin(), line.end(), delim)){
            std::cout << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
            allOk = false;
          }
        }
        
        std::replace(line.begin(), line.end(), delim, ' ');
        std::istringstream in(line);
        
        if(allOk){
          std::vector<double> rowValues = std::vector<double>(std::istream_iterator<double>(in), std::istream_iterator<double>());
          for (std::vector<double>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
            indata.push_back(*it);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    if((indata.size()% (nDelims+1)) != 0 ){
      std::cout << "Number of data is not an integer multiple of first line fields!" << std::endl;
      allOk = false;
    }
    if(indata.size() != _nRecords * (nDelims+1)){
      std::cout << "Expected " << _nRecords * (nDelims+1) << " Labels, got " << indata.size() << "!" << std::endl;
      allOk = false;
    }
  }
  
  
  if(allOk){
    nFields = nDelims+1;
    _labels.resize(indata.size());
    
    for(size_t iCol = 0; iCol < nFields; iCol++){
      for(size_t iRow = 0; iRow < _nRecords; iRow++){
        // Switching from row major to column major
        _labels[iCol*_nRecords + iRow] = indata[(iRow * nFields)+iCol];
      }
    }
    
    _nLabelFields = nDelims + 1;
    _labelsLoaded = true;
    
    std::cout << "Read " << _labels.size() << " labels of " << nRecords << " by " << _nFields << std::endl;
  }
  return allOk;
};

size_t dataset::nRecords() const{
  return _nRecords;
}

size_t dataset::nFields() const{
  return _nFields;
}

size_t dataset::nLabelFields() const{
  return _nLabelFields;
}

std::vector<double> dataset::data(){
  return _data;
}
std::vector<double> dataset::labels(){
  return _labels;
}

bool dataset::isPcaDone() const {
  return _pcaDone;
}

void dataset::analyseAndNorm(normDataType normType){
  _normParam1.resize(_nFields,0.0);
  _normParam2.resize(_nFields,0.0);
  double delta;
  if(_dataLoaded && _normType == DATA_NORM_NONE){
    switch (normType) {
      case DATA_STAN_NORM:
        // Welfords Method for standard deviation
        for(int iCol = 0; iCol < _nFields; iCol++){
          _normParam1[iCol] =  _data[iCol * _nRecords];
          for(int iRow = 1; iRow < _nRecords; iRow++){
            delta = _data[(iCol*_nRecords) + iRow] - _normParam1[iCol];
            _normParam1[iCol] += delta/(iRow+1);
            _normParam2[iCol] += delta*(_data[(iCol*_nRecords) + iRow]-_normParam1[iCol]);
          }
          _normParam2[iCol] = sqrt(_normParam2[iCol]/(_nRecords-1));
        }

        for(int iCol = 0; iCol < _nFields; iCol++){
          for(int iRow = 0; iRow < _nRecords; iRow++){
            _data[(iCol*_nRecords) + iRow] -= _normParam1[iCol];
            if(_normParam2[iCol] > MIN_DATA_RANGE){
              _data[(iCol*_nRecords) + iRow] /= _normParam2[iCol];
            }
          }
        }

        _normType = DATA_STAN_NORM;

        break;
      case DATA_RANGE_BOUND:
        for(int iCol = 0; iCol < _nFields; iCol++){
          _normParam1[iCol] = _data[iCol];
          _normParam2[iCol] = _data[iCol];

          for(int iRow = 1; iRow < _nRecords; iRow++){
            _normParam1[iCol] = std::min(_data[(iCol*_nRecords) + iRow],_normParam1[iCol]);
            _normParam2[iCol] = std::max(_data[(iCol*_nRecords) + iRow],_normParam2[iCol]);
          }
        }
        for(int iCol = 0; iCol < _nFields; iCol++){
          for(int iRow = 0; iRow < _nRecords; iRow++){
            _data[(iCol*_nRecords) + iRow] -= _normParam1[iCol] + ((_normParam2[iCol]- _normParam1[iCol])/2);
            if((_normParam2[iCol]- _normParam1[iCol]) > MIN_DATA_RANGE){
              _data[(iCol*_nRecords) + iRow] /= (_normParam2[iCol]- _normParam1[iCol]);
            }
          }
        }
        _normType = DATA_RANGE_BOUND;
        break;
      default:
        std::cout << "Confused in data normalisation function, doing nothing"<< std::endl;
        break;
    }

  }else{
    if(! _dataLoaded){
      std::cout << "Data normalisation requested but no data loaded" << std::endl;
    }else{
      if(_normType != DATA_NORM_NONE){
        std::cout << "Data normalisation already applied" << std::endl;
      }
    }
  }
  return;
}

void dataset::normFromDataset(const dataset& otherDataset){
  if(_normType== DATA_NORM_NONE){
    normFromParams(otherDataset.getNormType(), otherDataset.getNormParam1(), otherDataset.getNormParam2());
  }else{
    std::cout <<"Already Normed some way so doing nothing!";
  }
}


void dataset::normFromParams(const normDataType normType, const std::vector<double>& params1, const std::vector<double>& params2){
  bool canDo = true;
  if(! _dataLoaded){
    std::cout << "No non train data loaded\n";
    canDo = false;
  }
  if(normType == DATA_NORM_NONE){
    std::cout << "Training data is not normed so cannot perform on other data\n";
    canDo = false;
  }
  if(params1.size() != _nFields || params2.size() != _nFields){
    std::cout << "Dimension mismatch between train data normalisation parameters and load data\n";
    canDo = false;
  }
  
  if(canDo){
    switch (normType) {
      case DATA_STAN_NORM:
        for(int iCol = 0; iCol < _nFields; iCol++){
          for(int iRow = 0; iRow < _nRecords; iRow++){
            _data[(iCol*_nRecords)+iRow] -= params1[iCol];
            if(params2[iCol] > MIN_DATA_RANGE){
              _data[(iCol*_nRecords)+iRow] /= params2[iCol];
            }
          }
        }
        _normType = DATA_STAN_NORM;
        break;
      case DATA_RANGE_BOUND:
        for(int iCol = 0; iCol < _nFields; iCol++){
          for(int iRow = 0; iRow < _nRecords; iRow++){
            _data[(iCol*_nRecords)+iRow] -= params1[iCol] + ((params2[iCol]- params1[iCol])/2);
            if((_normParam2[iCol]- params1[iCol]) > MIN_DATA_RANGE){
              _data[(iCol*_nRecords)+iRow] /= (params2[iCol]- params1[iCol]);
            }
          }
        }
        _normType = DATA_RANGE_BOUND;
        break;
      default:
        std::cout << "Confused in data normalisation function, doing nothing"<< std::endl;
        break;
    }
    
  }
  
  return;
  
}

void dataset::doPca(size_t nRetainedDimensions){
  if(_normType != DATA_NORM_NONE){
    std::cout << "Only do PCA on un-transformed data sets, giving up\n";
  }else{
    if(nRetainedDimensions < 1 || nRetainedDimensions > _nFields){
      nRetainedDimensions = _nFields;
    }
    
    if(_dataLoaded){
      mat_ops::pca(_data, _nFields, nRetainedDimensions, _pcaEigenMat);
    }
    _nFields = nRetainedDimensions;
    _nPcaDimensions = nRetainedDimensions;
    _pcaDone = true;
  }
}

void dataset::doPcaFromDataset(const dataset& otherDataset){
  bool allOk = true;
  if(! otherDataset.isPcaDone()){
    std::cout << "Other Dataset does not have PCA  \n";
    allOk = false;
  }
  if(_normType != DATA_NORM_NONE){
    std::cout << "Only do PCA on un-transformed data sets, giving up\n";
    allOk = false;
  }
  if(allOk){
    doPcaProjection(otherDataset.getPcaMatrix(), otherDataset.nFields());
  }
  return;
}

void dataset::doPcaProjection(std::vector<double> pcaEigenMat, size_t nPcaDimension){
  bool canDo = true;
  if(!_dataLoaded){
    canDo = false;
    std::cout << "Can't do PCA as no data loaded\n";
  }

  if(_nRecords == 0){
    canDo = false;
    std::cout << "Can't do PCA as no records found\n";
  }
  
  if(canDo){
    mat_ops::pcaProject(_data, _nFields ,nPcaDimension, pcaEigenMat);
    _nFields = nPcaDimension;
    _nPcaDimensions = nPcaDimension;
    _pcaEigenMat = pcaEigenMat;
    _pcaEigenMatLoaded = true;
    _pcaDone = true;
  }
  return;
}

std::vector<double> dataset::getPcaMatrix() const{
  if(_pcaDone){
    return _pcaEigenMat;
  }else{
    std::cout << "No PCA matrix available!\n";
    return std::vector<double>();
  }
}

void dataset::transformDataset(const dataset& otherDataset){
  if(otherDataset.isPcaDone()){
    doPcaProjection(otherDataset.getPcaMatrix(), otherDataset.nFields());
  }
  if(otherDataset.getNormType() != DATA_NORM_NONE){
    normFromDataset(otherDataset);
  }
}
void dataset::printData(size_t nRecords){
  if(_dataLoaded){
    if(nRecords == 0){
      nRecords = _nRecords;
    }else{
      nRecords = std::min(nRecords, _nRecords);
    }
    std::cout << "Printing Data" << std::endl;
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nFields; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _data[(iCol* _nRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded\n";
  }
}


void dataset::printLabels(size_t nRecords){
  if(_dataLoaded){
    if(nRecords == 0){
      nRecords = _nRecords;
    }else{
      nRecords = std::min(nRecords, _nRecords);
    }
    std::cout << "Printing Labels" << std::endl;
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << "Record " << iRow + 1 << ": ";
      for(int iCol = 0; iCol < _nLabelFields; iCol++){
        std::cout << _labels[(iCol* _nRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No labels Loaded\n";
  }
}

void dataset::writeData(){
  if(_dataLoaded){
    std::ostringstream oss;
    oss << _outputDir << "data.csv";
    std::cout << "Wrirting data to " << oss.str() << std::endl;
    mat_ops::writeMatrix(oss.str(), _data,_nRecords,_nFields);
  }else{
    std::cout << "No data loaded\n";
  }
  return;
}

