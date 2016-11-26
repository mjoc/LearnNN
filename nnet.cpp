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

nnet::nnet(){
  _outputDir = "/Users/Martin/Google Drive/Projects/NN/testdata/";
  _indataLoaded = false;
  _indataLabelsLoaded = false;
  _outdataCreated = false;
  _indataNormed = false;
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _nIndataRecords = 0;

  _outputType = nnet::LIN_OUT_TYPE;
  _lossType = nnet::MSE_LOSS_TYPE;
  _activationType = nnet::LIN_ACT_TYPE;
  _rng = new rng;
  
  _firstFeedforward = true;
};
nnet::~nnet(){
  delete _rng;
};

void nnet::initialiseWeights(double stdev){
  size_t nTotalWeights = 0;
  size_t nCurrentInputWidth = _nInputUnits;

  std::cout << "Initialising weights" << std::endl;
  _hiddenWeights.resize(0);
  _hiddenBiases.resize(0);
  
  if(_hiddenLayerSizes.size() > 0){
    nTotalWeights = 0;
    for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
      nTotalWeights += nCurrentInputWidth * _hiddenLayerSizes[i];
      _hiddenWeights.resize(i+1);
      _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i], 0.1);
      _hiddenBiases.resize(i+1);
      _hiddenBiases[i].resize(_hiddenLayerSizes[i], BIAS_START);
      
      _rng->getGaussianVector(_hiddenWeights[i], stdev);
      nCurrentInputWidth = _hiddenLayerSizes[i];
    }
  }
  
  if(_nOutputUnits > 0){
    nTotalWeights += nCurrentInputWidth * _nOutputUnits;
    _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 0.1);
    _outputBiases.resize(_nOutputUnits, BIAS_START);
    _rng->getGaussianVector(_outputWeights, stdev);
    nCurrentInputWidth = _nOutputUnits;
  }
  std::cout << "Total Number of Weights: " << nTotalWeights << std::endl;
  if(_hiddenLayerSizes.size() > 0){
    std::cout << "Hidden Layer Weights: ";
    for(std::vector<std::vector<double> >::const_iterator it = _hiddenWeights.begin(); it != _hiddenWeights.end(); ++it)	{
      std::cout << it->size() << " ";
    }
    std::cout << std::endl;
  }
  
  if(_nOutputUnits > 0){
    std::cout << "Output Layer Weights: ";
    std::cout << _outputWeights.size();
    std::cout << std::endl;
  }
}


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

void nnet::printGeometry()
{
  int iLayer = 2;
  std::cout << "Input units: " << _nInputUnits << std::endl;
  for(std::vector<int>::iterator it = _hiddenLayerSizes.begin(); it != _hiddenLayerSizes.end(); ++it) {
    std::cout << "Layer " << iLayer++ << " units: " << *it << std::endl;
  }
  std::cout << "Output units: " << _nOutputUnits << std::endl;
}

bool nnet::loadDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  //std::vector<double> *indata_ptr = new std::vector<double>;
  
  _indataLabelsLoaded = false;
  _indataLoaded = false;
  
  _nInputUnits = 0;
  _nIndataRecords = 0;
  _indataNormed = false;
  
  std::ifstream infile(filename, std::ios_base::in);
  
  if (!infile.is_open()){
    allOk = false;
  }else{
    std::cout << "Reading in data from file " << filename << std::endl;
    _feedForwardValues.resize(1);
    _feedForwardValues[0].resize(0);
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
            _feedForwardValues[0].push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    //_feedForwardValues.push_back(*indata_ptr);
    _indataLoaded = true;
    _nIndataRecords = nRecords;
    _nInputUnits = nDelims + 1;
    std::cout << "Read " << _feedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool nnet::loadLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  double dataValue;
  
  _indataLabelsLoaded = false;
  _indataLabels.resize(0);
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
            _indataLabels.push_back(dataValue);
          }
        }else{
          break;
        }
      }
    }
  }
  if(allOk){
    _indataLabelsLoaded = true;
    _nOutputUnits = nDelims + 1;
    std::cout << "Read " << _indataLabels.size() << " class labels of " << nRecords << " by " << _nOutputUnits << std::endl;
  }
  return allOk;
};

void nnet::normIndata(){
  std::vector<double> means(_nInputUnits,0.0);
  std::vector<double> stds(_nInputUnits,0.0);
  double delta;
  if(_indataLoaded){
    // Welfords Method
    for(int i = 0; i < _nInputUnits; i++){
      means[i] =  _feedForwardValues[0][i];
      for(int j = 1; j < _nIndataRecords; j++){
        delta = _feedForwardValues[0][(j*_nInputUnits)+i] - means[i];
        means[i] += delta/(j+1);
        stds[i] += delta*(_feedForwardValues[0][(j*_nInputUnits)+i]-means[i]);
      }
      stds[i] = sqrt(stds[i]/(_nIndataRecords-1));
    }
    
    for(int i = 0; i < _nInputUnits; i++){
      for(int j = 0; j < _nIndataRecords; j++){
        _feedForwardValues[0][(j*_nInputUnits)+i] = (_feedForwardValues[0][(j*_nInputUnits)+i]- means[i])/stds[i];
      }
    }
    
    for(int j = 0; j < _nInputUnits; j++){
      std::cout << "Variable 1: Mean = " << means[j] << "std = " << stds[j] << std::endl;
    }
    
  }else{
    std::cout << "Data normalisation requested but no data loaded" << std::endl;
  }
  _indataNormed = true;
  return;
  
}


void nnet::printLabels(){
  if(_indataLabelsLoaded){
    std::cout << "Printing Labels" << std::endl;
    for(int i = 0; i < _nIndataRecords; i++){
      std::cout << "Record " << i + 1 << ": ";
      for(int j = 0; j < _nOutputUnits; j++){
        std::cout << _indataLabels[(i*_nOutputUnits)+j] << " | ";
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


void nnet::printIndata(){
  if(_indataLoaded){
    std::cout << "Printing data" << std::endl;
    for(int i = 0; i < _nIndataRecords; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nInputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _feedForwardValues[0][(i*_nInputUnits)+j] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded" <<std::endl;
  }
}

void nnet::printOutValues(){
  std::cout << "Printing the Output Unit Values" << std::endl;
  if(_outdataCreated){
    for(int i = 0; i < _nIndataRecords; i++){
      std::cout << " Record " << i + 1 << ": \t";
      for(int j = 0; j < _nOutputUnits; j++){
        std::cout << std::fixed;
        std::cout << std::setprecision(2) << _outData[(i*_nOutputUnits)+j] << " | ";
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
  if(iIndex < _feedForwardValues.size()){
    std::cout << "FF " << iIndex << " ";;
    nRows = _nIndataRecords;
    if(iIndex == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex -1];
    }
    if(nRows * nCols ==  _feedForwardValues[iIndex].size()){
      std::cout << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        std::cout << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          std::cout << std::fixed;
          std::cout << std::setprecision(2) << _feedForwardValues[iIndex][iRow*nCols+iCol] << " | ";
        }
        std::cout << std::endl;
      }
    }else{
      std::cout << "Error printing FF values " << nRows << " by " << nCols << " as it is actually " << _feedForwardValues[iIndex].size() << std::endl;
    }
      }else{
    std::cout << "Invalid FF index!" << std::endl;
  }
}

bool nnet::dataLoaded(){
  return _indataLoaded;
}

void nnet::activateUnits(std::vector<double>& values, std::vector<double>& gradients){
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
      // Do nothing
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

void nnet::activateOutput(std::vector<double>& values, std::vector<double>& gradients){
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case SMAX_OUT_TYPE:
      // Big help from: http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      // std::cout << "Activate Softmax final layer" << std::endl;
      double denominator = 0.0;
      double max = 0.0;
      
      std::vector<double> numerator(_nIndataRecords);
      for(int i = 0; i < _nIndataRecords; i++){
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

double nnet::getCost(){
  double cost = 0.0;
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iData = 0; iData < _nIndataRecords; ++iData){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          if(_indataLabels[(iData*_nOutputUnits)+iUnit] > 0.5){
            cost -= log( _outData[(iData*_nOutputUnits)+iUnit]);
          }
        }
      }
  }
  return cost;
}

double nnet::getAccuracy(){
  double accuracy = 0.0;
  double maxProb = 0.0;
  bool correct = false;
  for(int iData = 0; iData < _nIndataRecords; ++iData){
    for(int iOutput = 0; iOutput < _nOutputUnits; ++iOutput){
      if(iOutput == 0){
        maxProb = _outData[(iData*_nOutputUnits)+iOutput];
        if(_indataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(_outData[(iData*_nOutputUnits)+iOutput] > maxProb){
          maxProb = _outData[(iData*_nOutputUnits)+iOutput];
          if(_indataLabels[(iData*_nOutputUnits)+iOutput] > 0.5){
            correct = true;
          }else{
            correct = false;
          }
        }
      }
      
    }
    if(correct){accuracy += 1.0;}
  }
  accuracy /= _nIndataRecords;
  return accuracy;
}

void nnet::feedForward(){
  size_t nInputRows, nInputCols, nWeightsCols;
  std::vector<double> tempForBiasInAndMatmulOut;
  _outdataCreated = false;
  nInputRows = _nIndataRecords;
  nInputCols = _nInputUnits;
  if( _feedForwardValues.size() > 1){
    
    _feedForwardValues.resize(1);
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
//        std::cout << _feedForwardValues[iLayer][(i*nInputCols) + j] << "|" ;
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
    
    mat_ops::matMul(nInputRows, nInputCols, _feedForwardValues[iLayer], nWeightsCols, _hiddenWeights[iLayer], tempForBiasInAndMatmulOut);
    
//    std::cout << "Matmul" << std::endl;
//    for(int i = 0; i < nInputRows; i++){
//      for(int j = 0; j < nWeightsCols; j++){
//        std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//      }
//      std::cout << std::endl;
//    }
    _feedForwardValues.resize(iLayer+2);
    _feedForwardValues[iLayer+1] = tempForBiasInAndMatmulOut;
    _hiddenGradients.resize(iLayer+1);
    _hiddenGradients[iLayer].assign(tempForBiasInAndMatmulOut.size(),0.0);
//    std::cout << "Check " << _feedForwardValues.size() << " " << tempForBiasInAndMatmulOut.size() << " \n";
    
    
    
    activateUnits(_feedForwardValues[iLayer+1],_hiddenGradients[iLayer]);
    //Ninputrows doesn't change
    nInputCols = nWeightsCols;
  }
  
  
//  std::cout << "Feed forward: calculating output layer " << std::endl;
  nWeightsCols =  _nOutputUnits;
  tempForBiasInAndMatmulOut.resize(0);
  tempForBiasInAndMatmulOut = _outputBiases;
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
  mat_ops::matMul(nInputRows, nInputCols , _feedForwardValues[_feedForwardValues.size()-1] , nWeightsCols, _outputWeights, tempForBiasInAndMatmulOut);
//  
//      std::cout << "> Matmul" << std::endl;
//      for(int i = 0; i < nInputRows; i++){
//        for(int j = 0; j < nWeightsCols; j++){
//          std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//        }
//        std::cout << std::endl;
//      }

  
  _outData = tempForBiasInAndMatmulOut;
  _outputGradients.assign(tempForBiasInAndMatmulOut.size(),0.0);
  activateOutput(_outData,_outputGradients);
  _outdataCreated = true;

  if(_firstFeedforward){
    //writeFeedForwardValues();
    //writeOutValues();
    _firstFeedforward = false;
  }
  
  //printOutValues();
  return;
}

void nnet::backProp(size_t nBatchIndicator, double wgtLearnRate, double biasLearnRate, size_t nEpoch){
  size_t nInputs, nOutputs, iData = -1;
  
  size_t nBatchSize;
  double initialCost = 0.0, cost = 0.0;
  std::vector<double> errorProgression;
  std::ostringstream oss;
  
  if (nBatchIndicator < 1){
    nBatchSize = _nIndataRecords;
  }else{
    nBatchSize = nBatchIndicator;
  }
  
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case  nnet::SMAX_OUT_TYPE:
      //http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      //http://neuralnetworksanddeeplearning.com/chap3.html#softmax
      size_t nForwardWeightsOutput;
      std::vector<std::vector<double> > hiddenWeightsUpdate;
      std::vector<std::vector<double> > hiddenBiasesUpdate;
      std::vector<double> outputWeightsUpdate(_outputWeights.size(), 0.0);
      std::vector<double> outputBiasesUpdate(_outputBiases.size(), 0.0);
      std::vector<double> *inData, *outData, *forwardWeightsUpdate; //, *biases, *gradients;
      
      feedForward();
      
      //printFeedForwardValues(0);
      //printFeedForwardValues(1);
      
      
      outData = &_outData;
      
      errorProgression.resize(0);
      
      initialCost = getCost();
      errorProgression.push_back(initialCost);
      std::cout << "Initial Cost: " << initialCost <<  std::endl;
      for(int iEpoch = 0; iEpoch < nEpoch; iEpoch++){
        
        for(int iDataInBatch = 0; iDataInBatch < nBatchSize; iDataInBatch++){
          iData = (iData + 1) % _nIndataRecords;
          //std::cout << "+";
          inData = &_feedForwardValues[_feedForwardValues.size()-1];
          feedForward();
          if(iEpoch == 0 & iData == 0){
            initialCost = getCost();
            std::cout << "Initial Cost Check: " << initialCost <<  std::endl;
          }
          // Starting with the output layer what is the output size and input size
          nInputs = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
          nOutputs = _nOutputUnits;
          
          
          // Softmax with cross entrophy has a simple derviative form for the weights on the input to the output units
          for(int iInput = 0; iInput < _hiddenLayerSizes[_hiddenLayerSizes.size()-1]; ++iInput){
            for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
              outputWeightsUpdate[(iInput*nOutputs)+iOutput] = (*inData)[(iData*nInputs)+iInput] * (_outData[(iData*nOutputs)+iOutput]-_indataLabels[(iData*nOutputs)+iOutput]);
            }
          }
          
//          if(iEpoch == 0){
//            oss.clear();
//            oss.str("");
//            oss << _outputDir << "outputWeightsUpdate.csv";
//            mat_ops::writeMatrix(oss.str(), outputWeightsUpdate,_hiddenLayerSizes[_hiddenLayerSizes.size()-1],_nOutputUnits);
//            std::cout << "Wrote to " << oss.str() << std::endl;
//          }
          // Bias units have a 1 for the input weights of the unit, otherwise same as above
          for(int iBias = 0; iBias < nOutputs; ++iBias){
            outputBiasesUpdate[iBias] = (_outData[(iData*_nOutputUnits)+iBias]-_indataLabels[(iData*_nOutputUnits)+iBias]);
          }
//          if(iEpoch == 0){
//            oss.clear();
//            oss.str("");
//            oss << _outputDir << "outputBiasUpdate.csv";
//            mat_ops::writeMatrix(oss.str(), outputBiasesUpdate,1,nOutputs);
//            std::cout << "Wrote to " << oss.str() << std::endl;
//          }
          
          //std::cout << "Here" << std::endl;
          //inData = &_feedForwardValues[_feedForwardValues.size()-2];
          //biases = &_outputBiases;
          
          hiddenWeightsUpdate.resize(_hiddenWeights.size());
          hiddenBiasesUpdate.resize(_hiddenBiases.size());
          
          forwardWeightsUpdate = &outputWeightsUpdate;
          nForwardWeightsOutput = nOutputs;
          nOutputs = nInputs;
          
          for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
            // We are travelling backwards through the Weight matricies
            
            size_t iWgtMat = _hiddenWeights.size() - 1 - iWgtCount;
            if(iWgtMat == 0){
              nInputs = _nInputUnits;
            }else{
              nInputs = _hiddenLayerSizes[iWgtMat-1];
            }
            
            inData = &_feedForwardValues[iWgtMat];
            //gradients = &_hiddenGradients[i-1];
            
            hiddenWeightsUpdate[iWgtMat].resize(_hiddenWeights[iWgtMat].size(),0.0);
            //biases = &_hiddenBiases[i-1];
            hiddenBiasesUpdate[iWgtMat].resize(_hiddenBiases[iWgtMat].size(),0.0);
            // We need the derivative of the weight multiplication on the input; by the activation dervi; all chained with the derivatives of the weights on the output side of the units
            for(int iInput = 0; iInput < nInputs; ++iInput){
              for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                for(int iNext = 0; iNext < nForwardWeightsOutput; iNext++){
//                  if(iInput == 1 & iOutput == 0){
//                    std::cout << "*" << iWgtMat << ": " << (*inData)[(iData*nInputs)+iInput] << " | " << _hiddenGradients[iWgtMat][iData*nOutputs+iOutput] << " | " << (*forwardWeightsUpdate)[(iOutput*nForwardWeightsOutput)+iNext] << std::endl;
//                  }
                  hiddenWeightsUpdate[iWgtMat][(iInput*nOutputs)+iOutput]  += (*inData)[(iData*nInputs)+iInput]* _hiddenGradients[iWgtMat][iData*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nForwardWeightsOutput)+iNext];
                }
              }
            }
//            if(iEpoch == 0){
//              oss.clear();
//              oss.str("");
//              oss << _outputDir << "hiddenWeightsUpdate" << iWgtMat << ".csv";
//              mat_ops::writeMatrix(oss.str(), hiddenWeightsUpdate[iWgtMat],nInputs,nOutputs);
//              std::cout << "Wrote to " << oss.str() << std::endl;
//            }
            // Biases have 1 for the weight multiplicaiton on input, otherwise the same
            for(int iInput = 0; iInput < 1; ++iInput){
              for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                for(int iNext = 0; iNext< nForwardWeightsOutput; iNext++){
                  //std::cout << _hiddenGradients[iWgtMat][iData*nOutputs+iOutput] << " | " << (*forwardWeightsUpdate)[(iOutput*nForwardWeightsOutput)+iNext] <<std::endl;
                  hiddenBiasesUpdate[iWgtMat][iOutput] +=  _hiddenGradients[iWgtMat][iData*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nForwardWeightsOutput)+iNext];
                }
              }
            }
//            if(iEpoch == 0){
//              oss.clear();
//              oss.str("");
//              oss << _outputDir << "hiddenBiasesUpdate" << iWgtMat << ".csv";
//              mat_ops::writeMatrix(oss.str(), hiddenBiasesUpdate[iWgtMat],1,nOutputs);
//              std::cout << "Wrote to " << oss.str() << std::endl;
//            }
            forwardWeightsUpdate = &hiddenWeightsUpdate[iWgtMat];
            nForwardWeightsOutput = nOutputs;
            nOutputs = nInputs;
            
          }
          for(int iWgt = 0; iWgt <  _outputWeights.size() ; iWgt++){
            _outputWeights[iWgt] -= wgtLearnRate*outputWeightsUpdate[iWgt];
          }
//          std::cout << "Output Weights Update\n";
//          for(int iRow = 0; iRow <  _hiddenLayerSizes[_hiddenLayerSizes.size()-1] ; iRow++){
//            std::cout << "| ";
//            for(int iCol = 0; iCol < _nOutputUnits; iCol++){
//              std::cout << outputWeightsUpdate[iRow*_nOutputUnits + iCol] << " | ";
//            }
//            std::cout << std::endl;
//          }
          
          for(int iBias = 0; iBias < _outputBiases.size(); iBias++){
            _outputBiases[iBias] -= biasLearnRate*outputBiasesUpdate[iBias];
          }
//          std::cout << "Output Biases Update\n|";
//          for(int iBias = 0; iBias < _outputBiases.size(); iBias++){
//            std::cout << outputBiasesUpdate[iBias] << " | ";
//          }
//          std::cout << std::endl;
          
          for(int iWgtMat = 0; iWgtMat < _hiddenWeights.size(); iWgtMat++){
            for(int iWgt = 0; iWgt < _hiddenWeights[iWgtMat].size(); iWgt++){
              _hiddenWeights[iWgtMat][iWgt] -= wgtLearnRate*hiddenWeightsUpdate[iWgtMat][iWgt];
            }
            for(int iWgt = 0; iWgt < _hiddenBiases[iWgtMat].size(); iWgt++){
              _hiddenBiases[iWgtMat][iWgt] -= biasLearnRate*hiddenBiasesUpdate[iWgtMat][iWgt];
            }
          }
        }
        std::cout << std::endl;
        if ( iEpoch % 1 == 0 ){
          cost = getCost();
          errorProgression.push_back(cost);
          std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << "-- Accuracy " << getAccuracy() << std::endl;
        }
      }
      writeEpochCostUpdates(errorProgression);
      break;
  }
  //std::cout "Last feedforward\n";
  feedForward();
  
  cost = getCost();
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
  
  for(int i = 0; i < _feedForwardValues.size(); i++){
    if(i == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[i -1];
    }
    
    std::ostringstream oss;
    oss << _outputDir << "feedforward" << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _feedForwardValues[i],_nIndataRecords,nCols);
    
  }
  return;
}

void nnet::writeOutValues(){
  std::ostringstream oss;
  oss << _outputDir << "outvalues.csv";
  mat_ops::writeMatrix(oss.str(), _outData,_nIndataRecords,_nOutputUnits);
  return;
}

void nnet::writeEpochCostUpdates(std::vector<double> epochCostUpdates){
  std::ostringstream oss;
  oss << _outputDir << "epochCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), epochCostUpdates, epochCostUpdates.size(),1);
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



