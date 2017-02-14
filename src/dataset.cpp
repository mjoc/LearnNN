//
//  dataset.cpp
//  BasicNN
//
//  Created by Martin on 31/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include "dataset.hpp"
#include "mat_ops.hpp"
#include "message.hpp"


#define MIN_DATA_RANGE 1e-4

Dataset::Dataset(const char *dataFileName, const char *labelsFileName, bool hasHeader, const char* delim){
  bool allOk = true;
  char delimiter = *delim;
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
  allOk = loadDataFromFile(dataFileName, hasHeader, delimiter);
  if(allOk){
    allOk = loadLabelsFromFile(labelsFileName, hasHeader, delimiter);
  }
  if(allOk){
    _dataLoaded = true;
  }

  _outputDir = "~/";
}


Dataset::~Dataset(){
}


void Dataset::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}

std::string Dataset::getNormType() const{
  std::ostringstream normType;

  switch (_normType) {
  case DATA_STAN_NORM:
    normType << "snorm";
    break;
  case DATA_RANGE_BOUND:
    normType << "range";
    break;
  case DATA_NORM_NONE:
    normType << "none";
    break;
  default:
    normType <<  "unknown";
  break;
  }

  return std::string(normType.str());

}

std::vector<double> Dataset::getNormParam1() const {
  return _normParam1;
}

std::vector<double> Dataset::getNormParam2() const {
  return _normParam2;
}


bool Dataset::dataLoaded() const{
  return _dataLoaded;
}

bool Dataset::labelsLoaded() const{
  return _labelsLoaded;
}


bool Dataset::loadDataFromFile(const char *filename,
                               bool hasHeader,
                               char delim){
  std::ostringstream message;
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
    message << "Reading in data from file " << filename << std::endl;
    msg::info(message);
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
            message << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
            msg::error(message);
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
            message << "Number of data is not an integer multiple of first line fields!" << std::endl;
            msg::error(message);
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

    message << "Read " << nRecords << " by " << nDelims + 1 << std::endl;
    msg::info(message);
  }
  return allOk;
};

bool Dataset::loadLabelsFromFile(const char *filename,
                                 bool hasHeader,
                                 char delim){
  std::ostringstream message;
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
    message << "Cannot open file, giving up!\n";
    msg::error(message);
    allOk = false;
  }

  if(allOk){
    message << "Reading in data from file " << filename << std::endl;
    msg::error(message);
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
            message << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
            msg::error(message);
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
      message << "Number of data is not an integer multiple of first line fields!" << std::endl;
      msg::error(message);
      allOk = false;
    }
    if(indata.size() != _nRecords * (nDelims+1)){
      message << "Expected " << _nRecords * (nDelims+1) << " Labels, got " << indata.size() << "!" << std::endl;
      msg::error(message);
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

    message << "Read " << _labels.size() << " labels of " << nRecords << " by " << _nFields << std::endl;
    msg::info(message);
  }
  return allOk;
};

size_t Dataset::nRecords() const{
  return _nRecords;
}

size_t Dataset::nFields() const{
  return _nFields;
}

size_t Dataset::nLabelFields() const{
  return _nLabelFields;
}

std::vector<double>* Dataset::data(){
  return &_data;
}
std::vector<double>* Dataset::labels(){
  return &_labels;
}

bool Dataset::isPcaDone() const {
  return _pcaDone;
}

void Dataset::analyseAndNorm(std::string normType){
  normDataType internalNormType = DATA_NORM_NONE;

  bool normTypeValid = false;
  if(normType == "snorm"){
    normTypeValid = true;
    internalNormType = Dataset::DATA_STAN_NORM;
  }
  if(normType == "range"){
    normTypeValid = true;
    internalNormType = Dataset::DATA_RANGE_BOUND;
  }

  if(!normTypeValid){
    msg::error(std::string("Invalid data normalisation, only 'snorm' or 'range'!\n"));
  }

  std::ostringstream message;
  _normParam1.resize(_nFields,0.0);
  _normParam2.resize(_nFields,0.0);
  double delta;
  if(_dataLoaded && _normType == DATA_NORM_NONE){
    switch (internalNormType) {
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
        message << "Confused in data normalisation function, doing nothing"<< std::endl;
        msg::error(message);
        break;
    }

  }else{
    if(! _dataLoaded){
      message << "Data normalisation requested but no data loaded" << std::endl;
      msg::warn(message);
    }else{
      if(_normType != DATA_NORM_NONE){
        message << "Data normalisation already applied" << std::endl;
        msg::warn(message);
      }
    }
  }
  return;
}

void Dataset::normWithPrototype(Dataset& otherDataset){
 std::ostringstream message;
  if(_normType== DATA_NORM_NONE){
    normWithParams(otherDataset.getNormType(), otherDataset.getNormParam1(), otherDataset.getNormParam2());
  }else{
    message <<"Already Normed some way so doing nothing!";
    msg::warn(message);
  }
}


void Dataset::normWithParams(std::string normType, const std::vector<double>& params1, const std::vector<double>& params2){
  std::ostringstream message;

  normDataType internalNormType = DATA_NORM_NONE;

  bool normTypeValid = false;
  if(normType == "snorm"){
    normTypeValid = true;
    internalNormType = Dataset::DATA_STAN_NORM;
  }
  if(normType == "range"){
    normTypeValid = true;
    internalNormType = Dataset::DATA_RANGE_BOUND;
  }

  if(!normTypeValid){
    msg::error(std::string("Invalid data normalisation, only 'snorm' or 'range'!\n"));
  }

  bool canDo = true;
  if(! _dataLoaded){
    message << "No non train data loaded\n";
    msg::warn(message);
    canDo = false;
  }
  if(internalNormType == DATA_NORM_NONE){
    message << "Training data is not normed so cannot perform on other data\n";
    msg::error(message);
    canDo = false;
  }
  if(params1.size() != _nFields || params2.size() != _nFields){
    message << "Dimension mismatch between train data normalisation parameters and load data\n";
    msg::error(message);
    canDo = false;
  }

  if(canDo){
    switch (internalNormType) {
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
        message << "Confused in data normalisation function, doing nothing"<< std::endl;
        msg::error(message);
        break;
    }
  }
  return;
}

void Dataset::doPca(size_t nRetainedDimensions){
  std::ostringstream message;
  if(_normType != DATA_NORM_NONE){
    message << "Only do PCA on un-transformed data sets, giving up\n";
    msg::error(message);
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

void Dataset::doPcaFromDataset(const Dataset& otherDataset){
  std::ostringstream message;
  bool allOk = true;
  if(! otherDataset.isPcaDone()){
    message << "Other Dataset does not have PCA  \n";
    msg::error(message);
    allOk = false;
  }
  if(_normType != DATA_NORM_NONE){
    message << "Only do PCA on un-transformed data sets, giving up\n";
    msg::error(message);
    allOk = false;
  }
  if(allOk){
    doPcaProjection(otherDataset.getPcaMatrix(), otherDataset.nFields());
  }
  return;
}

void Dataset::doPcaProjection(std::vector<double> pcaEigenMat, size_t nPcaDimension){
  std::ostringstream message;
  bool canDo = true;
  if(!_dataLoaded){
    canDo = false;
    message << "Can't do PCA as no data loaded\n";
    msg::error(message);
  }

  if(_nRecords == 0){
    canDo = false;
    message << "Can't do PCA as no records found\n";
    msg::error(message);
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

std::vector<double> Dataset::getPcaMatrix() const{
  std::ostringstream message;
  if(_pcaDone){
    return _pcaEigenMat;
  }else{
    message << "No PCA matrix available1!\n";
    msg::error(message);
    return std::vector<double>();
  }
}

void Dataset::transformDataset( Dataset& otherDataset){
  if(otherDataset.isPcaDone()){
    doPcaProjection(otherDataset.getPcaMatrix(), otherDataset.nFields());
  }
  if(otherDataset.getNormType() != "none" and otherDataset.getNormType() != "unknown" ){
    normWithPrototype(otherDataset);
  }
}

void Dataset::printData(int nRecordsToPrint = 10){
  std::ostringstream message;
  size_t startRow = 0, endRow = _nRecords-1;
  if(_dataLoaded){
    if(nRecordsToPrint == 0){
      nRecordsToPrint = _nRecords;
    }else{
      if(nRecordsToPrint < 0){
        nRecordsToPrint = -nRecordsToPrint;
        startRow = std::max(0,(int)_nRecords - nRecordsToPrint);
      }

      nRecordsToPrint = std::min(nRecordsToPrint, (int)_nRecords);
    }
    endRow = std::min(startRow + nRecordsToPrint-1, _nRecords-1);
    message << "Printing Data" << std::endl;
    msg::info(message);
    message << "From " << startRow << " to " << endRow << std::endl;
    msg::info(message);

    for(int iRow = startRow; iRow <= endRow; iRow++){
      message << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nFields; iCol++){
        message << std::fixed;
        message << std::setprecision(3) << _data[(iCol* _nRecords)+iRow] << " | ";
      }
      message << std::endl;
      msg::info(message);
    }
  }else{
    message << "No data loaded\n";
    msg::error(message);
  }
}


void Dataset::printLabels(int nRecordsToPrint){
  std::ostringstream message;
  int startRow = 0, endRow = _nRecords-1;
  if(_labelsLoaded){
    if(nRecordsToPrint == 0){
      nRecordsToPrint = _nRecords;
    }else{
      if(nRecordsToPrint < 0){
        nRecordsToPrint = -nRecordsToPrint;
        startRow = std::max(0,(int)_nRecords - nRecordsToPrint);
      }
      nRecordsToPrint = std::min(nRecordsToPrint, (int)_nRecords);
    }
    endRow = std::min(startRow + nRecordsToPrint-1, (int)_nRecords-1);
    message << "Printing Labels" << std::endl;
    msg::info(message);
    for(int iRow = startRow; iRow <= endRow; iRow++){
      message << "Record " << iRow + 1 << ": ";
      for(int iCol = 0; iCol < _nLabelFields; iCol++){
        message << _labels[(iCol* _nRecords)+iRow] << " | ";
      }
      message << std::endl;
    }
    msg::info(message);
  }else{
    message << "No labels Loaded\n";
    msg::error(message);
  }
}

void Dataset::writeData(){
  std::ostringstream message;
  if(_dataLoaded){
    std::ostringstream oss;
    oss << _outputDir << "data.csv";
    message << "Writing data to " << oss.str() << std::endl;
    msg::info(message);
    mat_ops::writeMatrix(oss.str(), _data,_nRecords,_nFields);
  }else{
    msg::error(std::string("No data loaded\n"));
  }
  return;
}


// For R interface
Dataset::Dataset(Rcpp::String dataFileName, Rcpp::String labelsFileName, Rcpp::LogicalVector hasHeader, Rcpp::String delim){
  bool allOk = true;
  const char *delimiter = delim.get_cstring();
  char delimiter2 = *delimiter;
  bool hasHeader2 = (hasHeader[0]==FALSE) ? false : true;

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
  if(hasHeader2){
    msg::info("Header\n");
  }else{
    msg::info("No Header\n");
  }
  allOk = loadDataFromFile(dataFileName.get_cstring(), hasHeader2, delimiter2);
  if(allOk){
    allOk = loadLabelsFromFile(labelsFileName.get_cstring(), hasHeader2, delimiter2);
  }
  if(allOk){
    _dataLoaded = true;
  }

  _outputDir = "~/";
}

SEXP Dataset::getDataR(){
  if(_dataLoaded){
    return Rcpp::NumericMatrix( _nRecords , _nFields , _data.begin());
  }else{
    return R_NilValue;
  }
}

SEXP Dataset::getLabelsR(){
  if(_labelsLoaded){
    return Rcpp::NumericMatrix( _nRecords , _nLabelFields , _labels.begin());
  }else{
    return R_NilValue;
  }
}

SEXP Dataset::getPcaMatrixR(){
  std::ostringstream message;
  if(_pcaDone){
    return Rcpp::NumericMatrix( _nPcaDimensions , _nPcaDimensions , _pcaEigenMat.begin());
  }else{
    message << "No PCA matrix available2!\n";
    msg::error(message);
    return R_NilValue;
  }
}

Rcpp::NumericMatrix Dataset::getNormParamMat() const {
  Rcpp::NumericMatrix params( 2 , _normParam1.size() );

  for(int iParam = 0; iParam < _normParam1.size(); iParam++){
    params(0,iParam) = _normParam1[iParam];
    params(1,iParam) = _normParam2[iParam];
  }
  return params;
}



// Dataset::Dataset(Rcpp::NumericMatrix data, Rcpp::NumericMatrix labels){
// _outputDir = "~/";
// bool allOk = true;
// _dataLoaded = false;
// _labelsLoaded = false;
// _nRecords = 0;
// _nFields = 0;
// _nLabelFields = 0;
// _nInputFields = 0;
//
// _normType = DATA_NORM_NONE;
// _pcaDone = false;
// _pcaEigenMatLoaded = false;
// _nPcaDimensions = 0;
//
// if(allOk){
// _nFields = data.ncol();
// _nInputFields = _nFields;
// _nRecords = data.nrow();
//
// _data.resize(_nInputFields*_nRecords);
//
// for(int iCol = 0; iCol < _nFields; iCol++){
// for(int iRow = 0; iRow < _nRecords; iRow++){
// //Switching from row major to column major
// _data[iCol*_nRecords + iRow] = data(iRow,iCol);
// }
// }
//
// if(labels.nrow() > 0){
// _nLabelFields = labels.ncol();
// _labels.resize(_nRecords*_nLabelFields);
//
// for(int iCol = 0; iCol < _nFields; iCol++){
// for(int iRow = 0; iRow < _nRecords; iRow++){
// _labels[iCol*_nRecords + iRow] = labels(iRow,iCol);
// }
// }
// _labelsLoaded = true;
// }else{
// _labels.resize(0);
// }
// _dataLoaded = true;
// _dataSource = std::string("R");
// _labelsSource = std::string("R");
// }
// }


