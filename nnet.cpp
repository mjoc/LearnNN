#include <fstream>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <gsl/gsl_cblas.h>
#include <vector>
#include "mat_ops.hpp"
#include "nnet.hpp"


#define BIAS_START 0.0
#define MIN_DATA_RANGE 1e-4


// http://cs231n.github.io/neural-networks-3/

nnet::nnet(){
  _outputDir = "/Users/Martin/Google Drive/Projects/NN/testdata/";
  _trainDataLoaded = false;
  _trainDataLabelsLoaded = false;
  _trainLabelsGenerated = false;
  _trainDataShuffled = false;
  _weightsInitialised = false;
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _nTrainDataRecords = 0;

  _outputType = nnet::LIN_OUT_TYPE;
  _lossType = nnet::MSE_LOSS_TYPE;
  _activationType = nnet::LIN_ACT_TYPE;
  
  _trainDataNormType = nnet::DATA_NORM_NONE;
  _nonTrainDataNormType = nnet::DATA_NORM_NONE;
  _rng = new rng;

};

nnet::~nnet(){
  delete _rng;
};

bool nnet::loadTrainDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  //std::vector<double> *indata_ptr = new std::vector<double>;
  
  _trainDataLabelsLoaded = false;
  _trainDataLoaded = false;
  
  _nInputUnits = 0;
  _nTrainDataRecords = 0;

  _trainDataNormType = nnet::DATA_NORM_NONE;
  _trainDataPCA = false;
  _trainLabelsGenerated = false;
  _weightsInitialised = false;
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
    std::cout << "Reading in data from file " << filename << std::endl;
    _trainDataFeedForwardValues.resize(1);
    _trainDataFeedForwardValues[0].resize(0);
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
            dataValue = *it;
            _trainDataFeedForwardValues[0].push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    //_trainDataFeedForwardValues.push_back(*indata_ptr);
    _trainDataLoaded = true;
    _trainDataShuffled = false;
    _nTrainDataRecords = nRecords;
    _nInputUnits = nDelims + 1;
    std::cout << "Read " << _trainDataFeedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool nnet::loadTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  
  _trainDataLabelsLoaded = false;
  _trainDataLabels.resize(0);
  _nOutputUnits = 0;
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
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
          std::vector<int> rowValues = std::vector<int>(std::istream_iterator<int>(in), std::istream_iterator<int>());
          for (std::vector<int>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
            dataValue = *it;
            _trainDataLabels.push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    _trainDataLabelsLoaded = true;
    _nOutputUnits = nDelims + 1;
    std::cout << "Read " << _trainDataLabels.size() << " class labels of " << nRecords << " by " << _nOutputUnits << std::endl;
  }
  return allOk;
};

bool nnet::loadNonTrainDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  
  _nonTrainDataLabelsLoaded = false;
  _nonTrainDataLoaded = false;
  
  _nNonTrainDataRecords = 0;
  _trainDataNormType = nnet::DATA_NORM_NONE;
  _nonTrainDataPCA = false;
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
    std::cout << "Reading in data from file " << filename << std::endl;
    _nonTrainDataFeedForwardValues.resize(1);
    _nonTrainDataFeedForwardValues[0].resize(0);
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
            dataValue = *it;
            _nonTrainDataFeedForwardValues[0].push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    //_testFeedForwardValues.push_back(*indata_ptr);
    _nonTrainDataLoaded = true;
    _nNonTrainDataRecords = nRecords;
    _nNonTrainDataInputUnits = nDelims + 1;
    std::cout << "Read " << _nonTrainDataFeedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool nnet::loadNonTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  
  _nonTrainDataLabelsLoaded = false;
  _nonTrainDataLabels.resize(0);
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
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
          std::vector<int> rowValues = std::vector<int>(std::istream_iterator<int>(in), std::istream_iterator<int>());
          for (std::vector<int>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
            dataValue = *it;
            _nonTrainDataLabels.push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    _nonTrainDataLabelsLoaded = true;
    _nNonTrainDataOutputUnits = nDelims + 1;
    std::cout << "Read " << _nonTrainDataLabels.size() << " class labels of " << nRecords << " by " << _nOutputUnits << std::endl;
  }
  return allOk;
};

void nnet::setHiddenLayerSizes(const std::vector<int>& layerSizes){
  _hiddenLayerSizes.resize(0);
  for(std::vector<int>::const_iterator it = layerSizes.begin(); it != layerSizes.end(); ++it) {
    _hiddenLayerSizes.push_back(*it);
  }
  return;
}

void nnet::setActivationType(activationType activationType){
  _activationType = activationType;
  return;
}

void nnet::setOutputType(outputType outputType){
  _outputType = outputType;
  return;
}

void nnet::setLossType(lossType lossType){
  _lossType = lossType;
  return;
}

void nnet::initialiseWeights(double stdev){
  size_t nTotalWeights = 0;
  size_t nCurrentInputWidth = _nInputUnits;
  
  // std::cout << "Initialising weights" << std::endl;
  _hiddenWeights.resize(0);
  _hiddenBiases.resize(0);
  
  if(_hiddenLayerSizes.size() > 0){
    nTotalWeights = 0;
    for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
      nTotalWeights += nCurrentInputWidth * _hiddenLayerSizes[i];
      _hiddenWeights.resize(i+1);
      _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i]);
      _hiddenBiases.resize(i+1);
      _hiddenBiases[i].resize(_hiddenLayerSizes[i], BIAS_START);
      
      _rng->getGaussianVector(_hiddenWeights[i], stdev);
      nCurrentInputWidth = _hiddenLayerSizes[i];
    }
  }
  
  if(_nOutputUnits > 0){
    nTotalWeights += nCurrentInputWidth * _nOutputUnits;
    _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 1);
    _outputBiases.resize(_nOutputUnits, BIAS_START);
    _rng->getGaussianVector(_outputWeights, stdev);
    nCurrentInputWidth = _nOutputUnits;
  }
  // std::cout << "Total Number of Weights: " << nTotalWeights << std::endl;
//  if(_hiddenLayerSizes.size() > 0){
//    std::cout << "Hidden Layer Weights: ";
//    for(std::vector<std::vector<double> >::const_iterator it = _hiddenWeights.begin(); it != _hiddenWeights.end(); ++it)	{
//      std::cout << it->size() << " ";
//    }
//    std::cout << std::endl;
//  }
//  
//  if(_nOutputUnits > 0){
//    std::cout << "Output Layer Weights: ";
//    std::cout << _outputWeights.size();
//    std::cout << std::endl;
//  }
  _weightsInitialised = true;
}

void nnet::normTrainData(normDataType normType){
  _trainDataNormParam1.resize(_nInputUnits,0.0);
  _trainDataNormParam2.resize(_nInputUnits,0.0);
  double delta;
  if(_trainDataLoaded && _trainDataNormType == nnet::DATA_NORM_NONE){
    switch (normType) {
      case nnet::DATA_STAN_NORM:
        // Welfords Method for standard deviation
        for(int i = 0; i < _nInputUnits; i++){
          _trainDataNormParam1[i] =  _trainDataFeedForwardValues[0][i];
          for(int j = 1; j < _nTrainDataRecords; j++){
            delta = _trainDataFeedForwardValues[0][(j*_nInputUnits)+i] - _trainDataNormParam1[i];
            _trainDataNormParam1[i] += delta/(j+1);
            _trainDataNormParam2[i] += delta*(_trainDataFeedForwardValues[0][(j*_nInputUnits)+i]-_trainDataNormParam1[i]);
          }
          _trainDataNormParam2[i] = sqrt(_trainDataNormParam2[i]/(_nTrainDataRecords-1));
        }
        
        for(int i = 0; i < _nInputUnits; i++){
          for(int j = 0; j < _nTrainDataRecords; j++){
            _trainDataFeedForwardValues[0][(j*_nInputUnits)+i] -= _trainDataNormParam1[i];
            if(_trainDataNormParam2[i] > MIN_DATA_RANGE){
              _trainDataFeedForwardValues[0][(j*_nInputUnits)+i] /= _trainDataNormParam2[i];
            }
          }
        }
        
        _trainDataNormType = nnet::DATA_STAN_NORM;

        break;
      case nnet::DATA_RANGE_BOUND:
        for(int i = 0; i < _nInputUnits; i++){
          _trainDataNormParam1[i] = _trainDataFeedForwardValues[0][i];
          _trainDataNormParam2[i] = _trainDataFeedForwardValues[0][i];

          for(int j = 1; j < _nTrainDataRecords; j++){
            _trainDataNormParam1[i] = std::min(_trainDataFeedForwardValues[0][(j*_nInputUnits)+i],_trainDataNormParam1[i]);
            _trainDataNormParam2[i] = std::max(_trainDataFeedForwardValues[0][(j*_nInputUnits)+i],_trainDataNormParam2[i]);
          }
        }
        for(int iInputUnit = 0; iInputUnit < _nInputUnits; iInputUnit++){
          for(int iIndataRecord = 0; iIndataRecord < _nTrainDataRecords; iIndataRecord++){
            _trainDataFeedForwardValues[0][(iIndataRecord*_nInputUnits)+iInputUnit] -= _trainDataNormParam1[iInputUnit] + ((_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit])/2);
            if((_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit]) > MIN_DATA_RANGE){
            _trainDataFeedForwardValues[0][(iIndataRecord*_nInputUnits)+iInputUnit] /= (_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit]);
            }
          }
        }
        _trainDataNormType = nnet::DATA_RANGE_BOUND;
        break;
      default:
        std::cout << "Confused in data normalisation function, doing nothing"<< std::endl;
        break;
    }
    
  }else{
    if(! _trainDataLoaded){
      std::cout << "Data normalisation requested but no data loaded" << std::endl;
    }else{
      if(_trainDataNormType != nnet::DATA_NORM_NONE){
          std::cout << "Data normalisation already applied" << std::endl;
      }
    }
  }
  return;
  
}

void nnet::normNonTrainData(normDataType normType){
  bool canDo = true;
  if(! _nonTrainDataLoaded){
    std::cout << "No non train data loaded\n";
    canDo = false;
  }
  if(_trainDataNormType == nnet::DATA_NORM_NONE){
    std::cout << "Training data is not normed so cannot perform on other data\n";
    canDo = false;
  }
  if(!(_trainDataNormType == normType)){
    std::cout << "Training data norm type is different \n";
    canDo = false;
  }
  if(_trainDataNormParam1.size() != _nNonTrainDataInputUnits){
    std::cout << "Dinemsion mismatch between train data normalisation parameters and load data\n";
    canDo = false;
  }
  
  if(canDo){
    switch (normType) {
      case nnet::DATA_STAN_NORM:
        for(int i = 0; i < _nNonTrainDataInputUnits; i++){
          for(int j = 0; j < _nNonTrainDataRecords; j++){
            _nonTrainDataFeedForwardValues[0][(j*_nNonTrainDataInputUnits)+i] -= _trainDataNormParam1[i];
            if(_trainDataNormParam2[i] > MIN_DATA_RANGE){
              _nonTrainDataFeedForwardValues[0][(j*_nNonTrainDataInputUnits)+i] /= _trainDataNormParam2[i];
            }
          }
        }
        _nonTrainDataNormType = nnet::DATA_STAN_NORM;
        break;
      case nnet::DATA_RANGE_BOUND:
        for(int iInputUnit = 0; iInputUnit < _nNonTrainDataInputUnits; iInputUnit++){
          for(int iIndataRecord = 0; iIndataRecord < _nNonTrainDataRecords; iIndataRecord++){
            _nonTrainDataFeedForwardValues[0][(iIndataRecord*_nNonTrainDataInputUnits)+iInputUnit] -= _trainDataNormParam1[iInputUnit] + ((_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit])/2);
            if((_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit]) > MIN_DATA_RANGE){
              _nonTrainDataFeedForwardValues[0][(iIndataRecord*_nNonTrainDataInputUnits)+iInputUnit] /= (_trainDataNormParam2[iInputUnit]- _trainDataNormParam1[iInputUnit]);
            }
          }
        }
        _nonTrainDataNormType = nnet::DATA_RANGE_BOUND;
        break;
      default:
        std::cout << "Confused in data normalisation function, doing nothing"<< std::endl;
        break;
    }
    
  }
  
  return;
  
}


void nnet::shuffleTrainData(){
  std::vector<std::vector<int> > indataShuffle;
  std::vector<int> iClassShuffle;
  indataShuffle.resize(_nOutputUnits);
  iClassShuffle.resize(_nOutputUnits);
  
  for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
    indataShuffle[iLabelClass].resize(0);
    iClassShuffle[iLabelClass] = iLabelClass;
  }
  if(_trainDataLoaded){
    for(int iLabel = 0; iLabel < _nTrainDataRecords; iLabel++ ){
      for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
        if(_trainDataLabels[iLabel * _nOutputUnits + iLabelClass] > 0.8){
          indataShuffle[iLabelClass].push_back(iLabel);
          break;
        }
      }
    }
    for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
      if(indataShuffle[iLabelClass].size() > 1){
        _rng->getShuffled(indataShuffle[iLabelClass]);
    }
  }
    if(_nOutputUnits > 1){
      _rng->getShuffled(iClassShuffle);
    }
  size_t maxClass = indataShuffle[0].size();
  for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
    maxClass = indataShuffle[0].size() > maxClass ? indataShuffle[0].size() : maxClass;
  }
    
  _trainDataShuffleIndex.resize(0);
  for(int iLabel = 0; iLabel < maxClass; iLabel++ ){
    for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
      if(iLabel < indataShuffle[iClassShuffle[iLabelClass]].size()){
        _trainDataShuffleIndex.push_back(indataShuffle[iClassShuffle[iLabelClass]][iLabel]);
      }
    }
  }
  _trainDataShuffled = true;
  }
}

void nnet::pcaTrainData(size_t nRetainedDimensions){
  if(nRetainedDimensions < 1 || nRetainedDimensions > _nInputUnits){
    nRetainedDimensions = _nInputUnits;
  }
  
  if(_trainDataLoaded){
    mat_ops::pca(_trainDataFeedForwardValues[0], _nInputUnits, nRetainedDimensions, _pcaEigenMat);
  }
  _nInputUnits = nRetainedDimensions;
  _weightsInitialised = false;
  _trainDataPCA = nRetainedDimensions;
}

void nnet::pcaNonTrainData(){
  bool canDo = true;
  if(!_trainDataPCA){
    canDo = false;
    std::cout << "Can't do PCA on non-train data as it was not performed on the training train\n";
  }
  if(!_nonTrainDataLoaded){
    canDo = false;
    std::cout << "Can't do PCA on non-train data as it is not loaded\n";
    
  }
  if(_nNonTrainDataRecords == 0){
    canDo = false;
    std::cout << "Can't do PCA on non-train data as it has no records\n";

  }
  
  if(canDo){
    mat_ops::pcaProject(_nonTrainDataFeedForwardValues[0], _nNonTrainDataInputUnits ,_nInputUnits, _pcaEigenMat);
    _nNonTrainDataInputUnits = _nInputUnits;
    _nonTrainDataPCA = _trainDataPCA;
  }
  return;
}

void nnet::activateUnits(std::vector<double>& values){
  switch (_activationType){
    case nnet::TANH_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        
        values[i] =  tanh(values[i]);
      }
      break;
    case LIN_ACT_TYPE:
      // Do nothing
      break;
    case RELU_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        values[i] = fmax(0.0,values[i]);
      }
      break;
  }
  return;
};

void nnet::activateUnitsAndCalcGradients(std::vector<double>& values, std::vector<double>& gradients){
  if(values.size() != gradients.size()){
    std::cout << "****  Problem with mismatch value <-> gradient sizes\n";
  }
  switch (_activationType){
    case nnet::TANH_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        
        values[i] =  tanh(values[i]);
        gradients[i] =  (1-pow(values[i],2)) ;
      }
      break;
    case LIN_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        gradients[i] =  values[i];

      }
      break;
    case RELU_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        values[i] = fmax(0.0,values[i]);
        gradients[i] = (values[i]) > 0.0 ? 1 : 0.0;
      }
      break;
  }
  return;
};



void nnet::activateOutput(std::vector<double>& values){
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case SMAX_OUT_TYPE:
      // Big help from: http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      // std::cout << "Activate Softmax final layer" << std::endl;
      double denominator = 0.0;
      double max = 0.0;
      
      std::vector<double> numerator(_nTrainDataRecords);
      for(int i = 0; i < _nTrainDataRecords; i++){
        for(int j = 0; j < _nOutputUnits; j++){
          if(j == 0){
            max = values[(i*_nOutputUnits)+j];
          }else{
            if(values[(i*_nOutputUnits)+j]>max){
              max = values[(i*_nOutputUnits)+j];
            }
          }
        }
        for(int j = 0; j < _nOutputUnits; j++){
          values[(i*_nOutputUnits)+j] -= max;
        }
        
        denominator = 0.0;
        for(int j = 0; j < _nOutputUnits; j++){
          numerator[j] = exp(values[(i*_nOutputUnits)+j]);
          denominator  += numerator[j];
        }
        for(int j = 0; j < _nOutputUnits; j++){
          values[(i*_nOutputUnits)+j] = numerator[j]/denominator;
        }
      }
      break;
  }
  
}



bool nnet::dataLoaded(){
  return _trainDataLoaded;
}

double nnet::getTrainDataCost(){
  double cost = 0.0;
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iData = 0; iData < _nTrainDataRecords; ++iData){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          if(_trainDataLabels[(iData*_nOutputUnits)+iUnit] > 0.5){
            cost -= log( _trainGeneratedLabels[(iData*_nOutputUnits)+iUnit]);
          }
        }
      }
  }
  return cost;
}

double nnet::getTrainDataAccuracy(){
  double accuracy = 0.0;
  double maxProb = 0.0;
  bool correct = false;
  for(int iData = 0; iData < _nTrainDataRecords; ++iData){
    for(int iOutput = 0; iOutput < _nOutputUnits; ++iOutput){
      if(iOutput == 0){
        maxProb = _trainGeneratedLabels[(iData*_nOutputUnits)+iOutput];
        if(_trainDataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(_trainGeneratedLabels[(iData*_nOutputUnits)+iOutput] > maxProb){
          maxProb = _trainGeneratedLabels[(iData*_nOutputUnits)+iOutput];
          if(_trainDataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
            correct = true;
          }else{
            correct = false;
          }
        }
      }
      
    }
    if(correct){accuracy += 1.0;}
  }
  accuracy /= _nTrainDataRecords;
  return accuracy;
}

double nnet::getNonTrainDataCost(){
  bool canDo = true;
  if(!_nonTrainDataLoaded){
    std::cout << "Non Train Data no loaded";
    canDo = false;
  }
  if(!_nonTrainDataLabelsLoaded){
    std::cout << "Non Train Data Labels not loaded";
    canDo = false;
  }

  double cost = 0.0;
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iData = 0; iData < _nNonTrainDataRecords; ++iData){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          if(_nonTrainDataLabels[(iData*_nOutputUnits)+iUnit] > 0.5){
            cost -= log( _nonTrainGeneratedLabels[(iData*_nOutputUnits)+iUnit]);
          }
        }
      }
  }
  return cost;
}

double nnet::getNonTrainDataAccuracy(){
  double accuracy = 0.0;
  double maxProb = 0.0;
  bool correct = false;
  for(int iData = 0; iData < _nNonTrainDataRecords; ++iData){
    for(int iOutput = 0; iOutput < _nOutputUnits; ++iOutput){
      if(iOutput == 0){
        maxProb = _nonTrainGeneratedLabels[(iData*_nOutputUnits)+iOutput];
        if(_nonTrainDataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(_nonTrainGeneratedLabels[(iData*_nOutputUnits)+iOutput] > maxProb){
          maxProb = _nonTrainGeneratedLabels[(iData*_nOutputUnits)+iOutput];
          if(_nonTrainDataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
            correct = true;
          }else{
            correct = false;
          }
        }
      }
      
    }
    if(correct){accuracy += 1.0;}
  }
  accuracy /= _nNonTrainDataRecords;
  return accuracy;
}



void nnet::feedForwardTrainData(){
  size_t nInputRows, nInputCols, nWeightsCols;
  std::vector<double> tempForBiasInAndMatmulOut;
  
  if(_trainDataLoaded && _weightsInitialised){
    
    _trainLabelsGenerated = false;
    nInputRows = _nTrainDataRecords;
    nInputCols = _nInputUnits;
    if( _trainDataFeedForwardValues.size() > 1){
      
      _trainDataFeedForwardValues.resize(1);
    }
    //for(int iLayer = 0; iLayer < _hiddenLayerSizes.size(); iLayer++)
    //{
    //  std::cout << "Layer " << iLayer << " size: " << _hiddenLayerSizes[iLayer] << std::endl;
    //}
    
    for(int iLayer = 0; iLayer < _hiddenLayerSizes.size(); iLayer++)
    {
      //std::cout << "Feed forward: calculating layer " << iLayer << " size: " << _hiddenLayerSizes[iLayer] << std::endl;
      //tempForBiasInAndMatmulOut.resize(0);
      nWeightsCols =  _hiddenLayerSizes[iLayer];
      
      tempForBiasInAndMatmulOut.assign(nInputRows*_hiddenBiases[iLayer].size(),0.0);
      for(int i = 0; i < nInputRows; ++i){
        for(int j = 0; j <  _hiddenBiases[iLayer].size(); j++){
          tempForBiasInAndMatmulOut[(i*_hiddenBiases[iLayer].size())+j] = _hiddenBiases[iLayer][j];
        }
      }
      
      //    std::cout << "x values" << std::endl;
      //    for(int i = 0; i < nInputRows; i++){
      //      for(int j = 0; j < nInputCols; j++){
      //        std::cout << _trainDataFeedForwardValues[iLayer][(i*nInputCols) + j] << "|" ;
      //      }
      //      std::cout << std::endl;
      //    }
      //    std::cout << "Check: " << nInputCols << " : " << nWeightsCols << std::endl;
      //    std::cout << "y values" << std::endl;
      //    for(int i = 0; i < nInputCols; i++){
      //      for(int j = 0; j < nWeightsCols; j++){
      //        std::cout << _hiddenWeights[iLayer][(i*nWeightsCols) + j] << "|" ;
      //      }
      //      std::cout << std::endl;
      //    }
      
      
      //    std::cout << nInputRows << " x " << nInputCols << " by " << nInputCols << " x " << nWeightsCols << " bias " << tempForBiasInAndMatmulOut.size() << std::endl;
      //   std::cout << "Tempbias\n";
      //    for(int i = 0; i < nInputRows; i++){
      //      for(int j = 0; j < nWeightsCols; j++){
      //        std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
      //      }
      //      std::cout << std::endl;
      //    }
      
      mat_ops::matMul(nInputRows, nInputCols, _trainDataFeedForwardValues[iLayer], nWeightsCols, _hiddenWeights[iLayer], tempForBiasInAndMatmulOut);
      
      //    std::cout << "Matmul" << std::endl;
      //    for(int i = 0; i < nInputRows; i++){
      //      for(int j = 0; j < nWeightsCols; j++){
      //        std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
      //      }
      //      std::cout << std::endl;
      //    }
      _trainDataFeedForwardValues.resize(iLayer+2);
      _trainDataFeedForwardValues[iLayer+1] = tempForBiasInAndMatmulOut;
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(tempForBiasInAndMatmulOut.size(),0.0);
      //    std::cout << "Check " << _trainDataFeedForwardValues.size() << " " << tempForBiasInAndMatmulOut.size() << " \n";
      
      
      
      activateUnitsAndCalcGradients(_trainDataFeedForwardValues[iLayer+1],_hiddenGradients[iLayer]);
      //Ninputrows doesn't change
      nInputCols = nWeightsCols;
    }
    
    
    //  std::cout << "Feed forward: calculating output layer " << std::endl;
    nWeightsCols =  _nOutputUnits;
    tempForBiasInAndMatmulOut.resize(nInputRows*_outputBiases.size());
    for(int i = 0; i < nInputRows; ++i){
      for(int j = 0; j <  _outputBiases.size(); j++){
        tempForBiasInAndMatmulOut[(i*_outputBiases.size()) + j] = _outputBiases[j];
      }
    }
    
    
    //std::cout << nInputRows << " x " << nInputCols << " by " << nInputCols << " x " << nWeightsCols << " bias " << tempForBiasInAndMatmulOut.size() << std::endl;
    //std::cout << "Tempbias\n";
    //  for(int i = 0; i < nInputRows; i++){
    //    for(int j = 0; j < nWeightsCols; j++){
    //      std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
    //    }
    //    std::cout << std::endl;
    //  }
    mat_ops::matMul(nInputRows, nInputCols , _trainDataFeedForwardValues[_trainDataFeedForwardValues.size()-1] , nWeightsCols, _outputWeights, tempForBiasInAndMatmulOut);
    //
    //      std::cout << "> Matmul" << std::endl;
    //      for(int i = 0; i < nInputRows; i++){
    //        for(int j = 0; j < nWeightsCols; j++){
    //          std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
    //        }
    //        std::cout << std::endl;
    //      }
    
    
    _trainGeneratedLabels = tempForBiasInAndMatmulOut;
    activateOutput(_trainGeneratedLabels);
    _trainLabelsGenerated = true;
  }else{
    if (!_trainDataLoaded){
      std::cout << "No data loaded\n";
    }
    if (!_weightsInitialised){
      std::cout << "Weights not initialised\n";
    }
  }
  //printOutValues();
  return;
}

void nnet::feedForwardNonTrainData(){
  std::vector<double> tempMatrixForLabels;
  bool dataCorrect = true;
  
  if(!_nonTrainDataLoaded){
    std::cout << "No 'non-train' data loaded!";
    dataCorrect = false;
  }
  if(_nNonTrainDataRecords == 0){
    std::cout << "No 'non-train' data loaded!";
    dataCorrect = false;
  }
  if(_trainDataNormType == nnet::DATA_NORM_NONE && _nonTrainDataNormType != nnet::DATA_NORM_NONE){
    std::cout << "Train data was normalised, but 'non-train' was not!";
    dataCorrect = false;
  }
  if(_nonTrainDataNormType != nnet::DATA_NORM_NONE && _nonTrainDataNormType == nnet::DATA_NORM_NONE ){
    std::cout << "Non-train data was normalised, but train was not!";
    dataCorrect = false;
  }
  if(_trainDataPCA & !_nonTrainDataPCA){
    std::cout << "Train data was PCA'd but 'non-train' was not!";
    dataCorrect = false;
  }
  if(_nonTrainDataPCA & !_trainDataPCA ){
    std::cout << "Non-train data was PCA'd, but train was not!";
    dataCorrect = false;
  }
  if(_nNonTrainDataInputUnits != _nInputUnits){
    std::cout << "Differnet in dimension of Train (" << _nInputUnits<< ") and non-Train (" << _nNonTrainDataInputUnits << ")";
    dataCorrect = false;
  }
  if(_nNonTrainDataInputUnits != _nInputUnits){
    std::cout << "Different in dimension of Train Labels (" << _nOutputUnits<< ") and non-Train Labels (" << _nNonTrainDataOutputUnits << ")";
    dataCorrect = false;
    
  }
  if(dataCorrect){
    _nonTrainLabelsGenerated = false;
    
    flowDataThroughNetwork(_nonTrainDataFeedForwardValues, tempMatrixForLabels, false);
    
    _nonTrainGeneratedLabels = tempMatrixForLabels;
    
    _nonTrainLabelsGenerated = true;
  }else{
    if (!_trainDataLoaded){
      std::cout << "No data loaded\n";
    }
    if (!_weightsInitialised){
      std::cout << "Weights not initialised\n";
    }
  }
  //printOutValues();
  return;
}

void nnet::flowDataThroughNetwork(std::vector<std::vector<double> > dataflowStages, std::vector<double>& dataflowMatrix, bool calcTrainGradients){
  size_t nInputRows, nInputCols, nWeightsCols;
  
  
  nInputCols = _nInputUnits;
  nInputRows = dataflowStages[0].size()/nInputCols;
  
  if( dataflowStages.size() > 1){
    dataflowStages.resize(1);
  }
  
  for(int iLayer = 0; iLayer < _hiddenLayerSizes.size(); iLayer++)
  {
    nWeightsCols =  _hiddenLayerSizes[iLayer];
    
    dataflowMatrix.assign(nInputRows*_hiddenBiases[iLayer].size(),0.0);
    for(int i = 0; i < nInputRows; ++i){
      for(int j = 0; j <  _hiddenBiases[iLayer].size(); j++){
        dataflowMatrix[(i*_hiddenBiases[iLayer].size())+j] = _hiddenBiases[iLayer][j];
      }
    }
    
    mat_ops::matMul(nInputRows, nInputCols, dataflowStages[iLayer], nWeightsCols, _hiddenWeights[iLayer], dataflowMatrix);
    
    dataflowStages.resize(iLayer+2);
    dataflowStages[iLayer+1] = dataflowMatrix;
    
    if(calcTrainGradients){
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(dataflowMatrix.size(),0.0);
      activateUnitsAndCalcGradients(dataflowStages[iLayer+1],_hiddenGradients[iLayer]);
    }else{
      activateUnits(dataflowStages[iLayer+1]);
    }
    
    
    //Ninputrows doesn't change
    nInputCols = nWeightsCols;
  }
  
  
  //  std::cout << "Feed forward: calculating output layer " << std::endl;
  nWeightsCols =  _nOutputUnits;
  dataflowMatrix.resize(0);
  dataflowMatrix = _outputBiases;
  dataflowMatrix.resize(nInputRows*_outputBiases.size());
  for(int i = 0; i < nInputRows; ++i){
    for(int j = 0; j <  _outputBiases.size(); j++){
      dataflowMatrix[(i*_outputBiases.size()) + j] = _outputBiases[j];
    }
  }
  
  mat_ops::matMul(nInputRows,
                   nInputCols ,
                   dataflowStages[dataflowStages.size()-1] ,
                   nWeightsCols,
                   _outputWeights,
                   dataflowMatrix);
  
  activateOutput(dataflowMatrix);
  
  return;
}



void nnet::backProp(size_t nBatchIndicator,
                    double wgtLearnRate,
                    double biasLearnRate,
                    size_t nEpoch,
                    bool doMomentum,
                    double mom_mu,
                    double mom_decay,
                    size_t mom_decay_schedule,
                    double mom_final,
                    bool doTestCost){
  size_t nInputs, nOutputs, iDataStart, iDataStop;
  
  size_t nBatchSize;
  double initialCost = 0.0, cost = 0.0, testCost = 0.0;
  double momentum_mu = mom_mu;
  //std::vector<double> errorProgression;
  std::ostringstream oss;
  
  _epochTrainCostUpdates.resize(0);
  _epochTestCostUpdates.resize(0);
  
  
  if (nBatchIndicator < 1){
    nBatchSize = _nTrainDataRecords;
  }else{
    nBatchSize = nBatchIndicator;
  }
  
  if(doTestCost){
    if (!_nonTrainDataLoaded){
      doTestCost = false;
      std::cout << "There is no Non-Train data load, cannot calculate test error\n";
    }
    if(!_trainDataLabelsLoaded){
      doTestCost = false;
      std::cout << "There is no Non-Train data Labels loaded, cannot calculate test error\n";
    }
  }
  
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case  nnet::SMAX_OUT_TYPE:
      //http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      //http://neuralnetworksanddeeplearning.com/chap3.html#softmax
      size_t nOutputOutputs;
      std::vector<std::vector<double> > hiddenWeightsUpdate;
      std::vector<std::vector<double> > hiddenBiasesUpdate;
      
      std::vector<std::vector<double> > hiddenWeightsMomentum;
      std::vector<std::vector<double> > hiddenBiasesMomentum;
      
      std::vector<double> outputWeightsUpdate(_outputWeights.size(), 0.0);
      std::vector<double> outputBiasesUpdate(_outputBiases.size(), 0.0);
      
      std::vector<double> outputWeightsMomentum;
      std::vector<double> outputBiasesMomentum;
      
      std::vector<double> *inData, *outData, *forwardWeightsUpdate; //, *biases, *gradients;
      
      
      hiddenWeightsUpdate.resize(_hiddenWeights.size());
      hiddenBiasesUpdate.resize(_hiddenBiases.size());
      
      for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
        hiddenWeightsUpdate[iWgtCount].resize(_hiddenWeights[iWgtCount].size(),0.0);
        hiddenBiasesUpdate[iWgtCount].resize(_hiddenBiases[iWgtCount].size(),0.0);
      }
      
      if(doMomentum){
        hiddenWeightsMomentum.resize(_hiddenWeights.size());
        hiddenBiasesMomentum.resize(_hiddenBiases.size());
        
        for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
          hiddenWeightsMomentum[iWgtCount].resize(_hiddenWeights[iWgtCount].size(),0.0);
          hiddenBiasesMomentum[iWgtCount].resize(_hiddenBiases[iWgtCount].size(),0.0);
        }
        
        outputWeightsMomentum.resize(_outputWeights.size(), 0.0);
        outputBiasesMomentum.resize(_outputBiases.size(), 0.0);
        
      }
      
      feedForwardTrainData();
 
      outData = &_trainGeneratedLabels;
      initialCost = getTrainDataCost();
      _epochTrainCostUpdates.push_back(initialCost);
      std::cout << "Initial Cost: " << initialCost <<  std::endl;
      if(doTestCost){
        feedForwardNonTrainData();
        testCost = getNonTrainDataCost();
        _epochTestCostUpdates.push_back(testCost);
        std::cout << "Initial Test Cost: " << testCost <<  std::endl;
      }
      
      size_t nIterations =  _nTrainDataRecords/nBatchSize;
      if (nIterations*nBatchSize < _nTrainDataRecords){
        nIterations++;
      }
      for(int iEpoch = 0; iEpoch < nEpoch; iEpoch++){
        if(doMomentum && iEpoch > 0){
          if(mom_decay_schedule > 0){
            if(iEpoch% mom_decay_schedule == 0){
              momentum_mu += mom_decay * (mom_final - momentum_mu);
              std::cout << "Momentum decay parameter" << momentum_mu << std::endl;
            }
          }
        }
        
        for(int iIteration = 0; iIteration < nIterations; iIteration++){
          /***********************/
          /** Calculate updates **/
          /***********************/
          
          iDataStart = iIteration * nBatchSize;
          iDataStop = (iIteration + 1) * nBatchSize;
          iDataStop = ((iDataStop <= _nTrainDataRecords) ? iDataStop : _nTrainDataRecords);
          
          //std::cout << "+";
          inData = &_trainDataFeedForwardValues[_trainDataFeedForwardValues.size()-1];
          feedForwardTrainData();
          if(iEpoch == 0 & iDataStart == 0){
            initialCost = getTrainDataCost();
            std::cout << "Initial Cost Check: " << initialCost <<  std::endl;
          }
          
          // Zeroing the update matricies
          std::fill(outputWeightsUpdate.begin(),outputWeightsUpdate.end(),0.0);
          std::fill(outputBiasesUpdate.begin(),outputBiasesUpdate.end(),0.0);
          
          for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
            std::fill(hiddenWeightsUpdate[iWgtCount].begin(), hiddenWeightsUpdate[iWgtCount].end(), 0.0);
            std::fill(hiddenBiasesUpdate[iWgtCount].begin(), hiddenBiasesUpdate[iWgtCount].end(), 0.0);
          }
          // Starting with the output layer what is the output size and input size
          nInputs = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
          nOutputs = _nOutputUnits;

          for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
            size_t iDataInBatch;
            if(_trainDataShuffled){
              iDataInBatch = _trainDataShuffleIndex[iDataIndex];
            }else{
              iDataInBatch = iDataIndex;
            }
            // Softmax with cross entrophy has a simple derviative form for the weights on the input to the output units
            for(int iInput = 0; iInput < _hiddenLayerSizes[_hiddenLayerSizes.size()-1]; ++iInput){
              for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                outputWeightsUpdate[(iInput*nOutputs)+iOutput] += (*inData)[(iDataInBatch*nInputs)+iInput] * (_trainGeneratedLabels[(iDataInBatch*nOutputs)+iOutput]-_trainDataLabels[(iDataInBatch*nOutputs)+iOutput]);
              }
            }
            // Bias units have a 1 for the input weights of the unit, otherwise same as above
            for(int iBias = 0; iBias < nOutputs; ++iBias){
              outputBiasesUpdate[iBias] += (_trainGeneratedLabels[(iDataInBatch*_nOutputUnits)+iBias]-_trainDataLabels[(iDataInBatch*_nOutputUnits)+iBias]);
            }
          }
          forwardWeightsUpdate = &outputWeightsUpdate;
          nOutputOutputs = nOutputs;
          nOutputs = nInputs;
          
          // We are travelling backwards through the Weight matricies
          for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
            // As we go backwards find the correct weights to update
            size_t iWgtMat = _hiddenWeights.size() - 1 - iWgtCount;
            if(iWgtMat == 0){
              nInputs = _nInputUnits;
            }else{
              nInputs = _hiddenLayerSizes[iWgtMat-1];
            }
            
            inData = &_trainDataFeedForwardValues[iWgtMat];
            
            for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
              size_t iDataInBatch;
              if(_trainDataShuffled){
                iDataInBatch = _trainDataShuffleIndex[iDataIndex];
              }else{
                iDataInBatch = iDataIndex;
              }
              // We need the derivative of the weight multiplication on the input; by the activation dervi;
              // all chained with the derivatives of the weights on the output side of the units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                  for(int iNext = 0; iNext < nOutputOutputs; iNext++){
                    hiddenWeightsUpdate[iWgtMat][(iInput*nOutputs)+iOutput]  += (*inData)[(iDataInBatch*nInputs)+iInput]* _hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nOutputOutputs)+iNext];
                  }
                }
              }
              // Biases have 1 for the weight multiplicaiton on input, otherwise the same
              for(int iInput = 0; iInput < 1; ++iInput){
                for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                  for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                    hiddenBiasesUpdate[iWgtMat][iOutput] +=  _hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nOutputOutputs)+iNext];
                  }
                }
              }
            }
            
            // Move back to the next weights, be need to keep the info from the "forward" which is one beyond the output weights
            forwardWeightsUpdate = &hiddenWeightsUpdate[iWgtMat];
            nOutputOutputs = nOutputs;
            nOutputs = nInputs;
          }
          
          /*********************/
          /** Perform updates **/
          /*********************/
          // First output weights and biases and then move back
          for(int iWgt = 0; iWgt < _outputWeights.size(); iWgt++){
            if(doMomentum){
              outputWeightsMomentum[iWgt] = momentum_mu * outputWeightsMomentum[iWgt] - wgtLearnRate * outputWeightsUpdate[iWgt];
              _outputWeights[iWgt] += outputWeightsMomentum[iWgt];
            }else{
              _outputWeights[iWgt] -= wgtLearnRate*outputWeightsUpdate[iWgt];
            }
          }
          for(int iBias = 0; iBias < _outputBiases.size(); iBias++){
            if(doMomentum){
              outputBiasesMomentum[iBias] = momentum_mu * outputBiasesMomentum[iBias] - biasLearnRate * outputBiasesUpdate[iBias];
              _outputBiases[iBias] += outputWeightsMomentum[iBias];
            }else{
              _outputBiases[iBias] -= biasLearnRate*outputBiasesUpdate[iBias];
            }
          }
          // Back through the hidden layers updating the weights
          for(int iWgtMat = 0; iWgtMat < _hiddenWeights.size(); iWgtMat++){
            for(int iWgt = 0; iWgt < _hiddenWeights[iWgtMat].size(); iWgt++){
              if(doMomentum){
                hiddenWeightsMomentum[iWgtMat][iWgt] = momentum_mu * hiddenWeightsMomentum[iWgtMat][iWgt] - wgtLearnRate * hiddenWeightsUpdate[iWgtMat][iWgt];
                _hiddenWeights[iWgtMat][iWgt] += hiddenWeightsMomentum[iWgtMat][iWgt] ;
              }else{
                _hiddenWeights[iWgtMat][iWgt] -= wgtLearnRate*hiddenWeightsUpdate[iWgtMat][iWgt];
                
              }
            }
            for(int iWgt = 0; iWgt < _hiddenBiases[iWgtMat].size(); iWgt++){
              if(doMomentum){
                hiddenBiasesMomentum[iWgtMat][iWgt] = momentum_mu * hiddenBiasesMomentum[iWgtMat][iWgt] - biasLearnRate * hiddenBiasesUpdate[iWgtMat][iWgt];
                _hiddenBiases[iWgtMat][iWgt] += hiddenBiasesMomentum[iWgtMat][iWgt] ;
              }else{
                _hiddenBiases[iWgtMat][iWgt] -= biasLearnRate*hiddenBiasesUpdate[iWgtMat][iWgt];
              }
            }
          }
        }
        
        //feedForwardTrainData();
        cost = getTrainDataCost();
        _epochTrainCostUpdates.push_back(cost);
        std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << "-- Accuracy " << getTrainDataAccuracy() << "  ("  << getTrainDataAccuracy() * _nTrainDataRecords << ")" << std::endl;
        
        if(doTestCost){
          feedForwardNonTrainData();
          testCost = getNonTrainDataCost();
          _epochTestCostUpdates.push_back(testCost);
          std::cout << "Test Cost: " << testCost <<  std::endl;
        }
        
      }
      break;
  }
  //std::cout "Last feedforward\n";
  feedForwardTrainData();
  
  cost = getTrainDataCost();
  std::cout <<  " Cost went from " << initialCost << " to " <<  cost << std::endl;

  return;
}

void nnet::writeWeightValues(){
  size_t nRows = 0, nCols = 0;
  
  for(int i = 0; i < _hiddenWeights.size(); i++){
    if(i == 0){
      nRows = _nInputUnits;
    }else{
      nRows = _hiddenLayerSizes[i -1];
    }
    nCols = _hiddenLayerSizes[i -1];
    std::ostringstream oss;
    oss << _outputDir << "Weights " << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _hiddenWeights[i],nRows,nCols);
    oss << _outputDir << "Bias " << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _hiddenBiases[i],1,nCols);
  }
  return;
}

void nnet::writeFeedForwardValues(){
  size_t nCols = 0;
  
  for(int i = 0; i < _trainDataFeedForwardValues.size(); i++){
    if(i == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[i -1];
    }
    
    std::ostringstream oss;
    oss << _outputDir << "feedforward" << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _trainDataFeedForwardValues[i],_nTrainDataRecords,nCols);
    
  }
  return;
}

void nnet::writeOutValues(){
  std::ostringstream oss;
  if(_trainLabelsGenerated){
    oss << _outputDir << "outvalues.csv";
    mat_ops::writeMatrix(oss.str(), _trainGeneratedLabels,_nTrainDataRecords,_nOutputUnits);
  }else{
    std::cout << "No output data created\n";
  }
  return;
}

void nnet::writeEpochTrainCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTrainCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTrainCostUpdates, _epochTrainCostUpdates.size(),1);
  return;
}

void nnet::writeEpochTestCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTestCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTestCostUpdates, _epochTestCostUpdates.size(),1);
  return;
}


void nnet::printUnitType(){
  switch (_activationType){
    case nnet::TANH_ACT_TYPE:
      std::cout << "Unit: TANH" << std::endl;
      break;
    case LIN_ACT_TYPE:
      std::cout << "Unit: LINEAR" << std::endl;
      break;
    case RELU_ACT_TYPE:
      std::cout << "Unit: RELU" << std::endl;
      break;
 	}
  
}

void nnet::printOutputType(){
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      std::cout << "Output: LINEAR" << std::endl;
      break;
    case SMAX_OUT_TYPE:
      std::cout << "Output: SOFTMAX" << std::endl;
      break;
  }
  
}

void nnet::printGeometry(){
  int iLayer = 2;
  std::cout << "Input units: " << _nInputUnits << std::endl;
  for(std::vector<int>::iterator it = _hiddenLayerSizes.begin(); it != _hiddenLayerSizes.end(); ++it) {
    std::cout << "Layer " << iLayer++ << " units: " << *it << std::endl;
  }
  std::cout << "Output units: " << _nOutputUnits << std::endl;
}

void nnet::printLabels(){
  if(_trainDataLabelsLoaded){
    std::cout << "Printing Labels" << std::endl;
    for(int i = 0; i < _nTrainDataRecords; i++){
      std::cout << "Record " << i + 1 << ": ";
      for(int j = 0; j < _nOutputUnits; j++){
        std::cout << _trainDataLabels[(i*_nOutputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No Labels Loaded" << std::endl;
  }
}

void nnet::printWeights(int iWeightsIndex){
  size_t nRows = 0;
  size_t nCols = 0;
  if(iWeightsIndex < _hiddenWeights.size()){
    std::cout << "Weights " << iWeightsIndex << " ";
    if(iWeightsIndex == 0){
      nRows = _nInputUnits;
      nCols = _hiddenLayerSizes[0];
    }else{
      nRows = _hiddenLayerSizes[iWeightsIndex-1];
      nCols = _hiddenLayerSizes[iWeightsIndex];
    }
    if(nRows * nCols ==  _hiddenWeights[iWeightsIndex].size()){
      std::cout << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        std::cout << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          std::cout << std::fixed;
          std::cout << std::setprecision(2) << _hiddenWeights[iWeightsIndex][iRow*nCols+iCol] << " | ";
        }
        std::cout << std::endl;
      }
    }else{
      std::cout << "Error printing weights " << nRows << " by " << nCols << " as it is actually " << _hiddenWeights[iWeightsIndex].size() << std::endl;
    }
    if(iWeightsIndex < _hiddenBiases.size()){
      std::cout << "Bias Vector: " << std::endl;
      for(std::vector<double>::const_iterator it = _hiddenBiases[iWeightsIndex].begin(); it !=  _hiddenBiases[iWeightsIndex].end(); ++it){
        if(it == _hiddenBiases[iWeightsIndex].begin()){
          std::cout << *it;
        }else{
          std::cout <<  " : " << *it;
        }
      }
      std::cout << std::endl;
    }else{
      std::cout << "Bias index!" << std::endl;
    }
  }else{
    std::cout << "Invalid layer index!" << std::endl;
  }
}

void nnet::printOutputWeights(){
  size_t nRows = 0;
  size_t nCols = 0;
  if( _nOutputUnits > 0){
    std::cout << "Output Weights ";
    if(_hiddenLayerSizes.size() == 0){
      nRows = _nInputUnits;
      nCols = _nOutputUnits;
    }else{
      nRows = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
      nCols = _nOutputUnits;
    }
    
    std::cout << "( " << nRows << " x " << nCols << " )" << std::endl;
    for(int iRow = 0; iRow < nRows; iRow++){
      std::cout << "| " ;
      for(int iCol = 0; iCol < nCols; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(2) << _outputWeights[iRow*nCols+iCol] << " | ";
      }
      std::cout << std::endl;
    }
    std::cout << "Bias Vector: " << std::endl;
    for(std::vector<double>::const_iterator it = _outputBiases.begin(); it !=  _outputBiases.end(); ++it){
      if(it == _outputBiases.begin()){
        std::cout << *it;
      }else{
        std::cout <<  " : " << *it;
      }
    }
    std::cout << std::endl;
  }
  return;
}

void nnet::printGradients(int iWeightsIndex){
  size_t nRows = 0;
  size_t nCols = 0;
  if(iWeightsIndex <= _hiddenGradients.size()){
    std::cout << "Gradients " << iWeightsIndex << " ";
    if(iWeightsIndex == 0){
      nRows = _nInputUnits;
      nCols = _hiddenLayerSizes[0];
    }else{
      if(iWeightsIndex == _hiddenGradients.size()-1){
        nRows = _nOutputUnits;
        nCols = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
      }else{
        nRows = _hiddenLayerSizes[iWeightsIndex-1];
        nCols = _hiddenLayerSizes[iWeightsIndex];
      }
    }
    if(nRows * nCols ==  _hiddenGradients[iWeightsIndex].size()){
      std::cout << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        std::cout << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          std::cout << std::fixed;
          std::cout << std::setprecision(2) << _hiddenGradients[iWeightsIndex][iRow*nCols+iCol] << " | ";
        }
        std::cout << std::endl;
      }
      
    }else{
      std::cout << "Error printing weights " << nRows << " by " << nCols << " as it is actually " << _hiddenWeights[iWeightsIndex].size() << std::endl;
    }
    if(iWeightsIndex < _hiddenBiases.size()){
      std::cout << "Bias Vector: " << std::endl;
      for(std::vector<double>::const_iterator it = _hiddenBiases[iWeightsIndex].begin(); it !=  _hiddenBiases[iWeightsIndex].end(); ++it){
        if(it == _hiddenBiases[iWeightsIndex].begin()){
          std::cout << *it;
        }else{
          std::cout <<  " : " << *it;
        }
      }
      std::cout << std::endl;
    }else{
      std::cout << "Bias index!" << std::endl;
    }
  }else{
    std::cout << "Invalid layer index!" << std::endl;
  }
}

void nnet::printTrainData(){
  if(_trainDataLoaded){
    std::cout << "Printing data" << std::endl;
    for(int i = 0; i < _nTrainDataRecords; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nInputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _trainDataFeedForwardValues[0][(i*_nInputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded" <<std::endl;
  }
}

void nnet::printTrainData(size_t rows){
  if(_trainDataLoaded){
    size_t rowsToPrint = std::min(_nTrainDataRecords,rows);
    std::cout << "Printing data" << std::endl;
    for(int i = 0; i < rowsToPrint; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nInputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _trainDataFeedForwardValues[0][(i*_nInputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded" <<std::endl;
  }
}

void nnet::printNonTrainData(){
  if(_nonTrainDataLoaded){
    std::cout << "Printing data" << std::endl;
    for(int i = 0; i < _nNonTrainDataRecords; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nNonTrainDataInputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _nonTrainDataFeedForwardValues[0][(i*_nNonTrainDataInputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No non-train data loaded" <<std::endl;
  }
}

void nnet::printNonTrainData(size_t rows){
  if(_nonTrainDataLoaded){
    size_t rowsToPrint = std::min(_nNonTrainDataRecords,rows);
    std::cout << "Printing data" << std::endl;
    for(int i = 0; i < rowsToPrint; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nNonTrainDataInputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _nonTrainDataFeedForwardValues[0][(i*_nNonTrainDataInputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No non-train data loaded" <<std::endl;
  }
}


void nnet::printOutValues(){
  std::cout << "Printing the Output Unit Values" << std::endl;
  if(_trainLabelsGenerated){
    for(int i = 0; i < _nTrainDataRecords; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nOutputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(2) << _trainGeneratedLabels[(i*_nOutputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data created" <<std::endl;
  }
}

void nnet::printFeedForwardValues(int iIndex){
  size_t nRows = 0;
  size_t nCols = 0;
  if(iIndex < _trainDataFeedForwardValues.size()){
    std::cout << "FF " << iIndex << " ";;
    nRows = _nTrainDataRecords;
    if(iIndex == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex -1];
    }
    if(nRows * nCols ==  _trainDataFeedForwardValues[iIndex].size()){
      std::cout << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        std::cout << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          std::cout << std::fixed;
          std::cout << std::setprecision(2) << _trainDataFeedForwardValues[iIndex][iRow*nCols+iCol] << " | ";
        }
        std::cout << std::endl;
      }
    }else{
      std::cout << "Error printing FF values " << nRows << " by " << nCols << " as it is actually " << _trainDataFeedForwardValues[iIndex].size() << std::endl;
    }
  }else{
    std::cout << "Invalid FF index!" << std::endl;
  }
}


