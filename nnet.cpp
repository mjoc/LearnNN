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
#define ROW_MAJOR true
#define MIN_DATA_RANGE 1e-4


// http://cs231n.github.io/neural-networks-3/

nnet::nnet(){
  _outputDir = "~/";
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
  
  _doDropout = false;
  _inputDropoutRate = 0.0;
  _hiddenDropoutRate = 0.0;
  _doMomentum =false;
  _momMu = 0.0;
  _momDecay = 0.0;
  _momDecaySchedule = 0;
  _momFinal = 0.0;
  
  _rng = new rng;

};


nnet::~nnet(){
  delete _rng;
};

void nnet::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
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

void nnet::doTestCost(bool doTestCost){
  _doTestCost = doTestCost;
}

void nnet::doDropout(bool doDropout){
   _doDropout = doDropout;
}

void nnet::setDropoutRates(double inputDropout, double hiddenDropout){
  _inputDropoutRate = inputDropout;
  _hiddenDropoutRate = hiddenDropout;
  return;
}

void nnet::setMomentum(bool doMomentum,
                 double momMu,
                 double momDecay,
                 size_t momDecaySchedule,
                 double momFinal){
  _doMomentum = doMomentum;
  _momMu = momMu;
  _momDecay = momDecay;
  _momDecaySchedule = momDecaySchedule;
  _momFinal = momFinal;
}


void nnet::initialiseWeights(initialiseType initialtype, double param1){
  size_t nCurrentInputWidth = _nInputUnits;
  
  // std::cout << "Initialising weights" << std::endl;
  _hiddenWeights.resize(0);
  _hiddenBiases.resize(0);
  
  switch (initialtype) {
    case nnet::INIT_CONST_TYPE:
      if(_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
          _hiddenWeights.resize(i+1);
          _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i], param1);
          _hiddenBiases.resize(i+1);
          _hiddenBiases[i].resize(_hiddenLayerSizes[i], BIAS_START);
          nCurrentInputWidth = _hiddenLayerSizes[i];
        }
      }
      
      if(_nOutputUnits > 0){
        _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 1);
        _outputBiases.resize(_nOutputUnits, BIAS_START);
      }
    break;
    case nnet::INIT_GAUSS_TYPE:
      if(_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
          _hiddenWeights.resize(i+1);
          _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i]);
          _hiddenBiases.resize(i+1);
          _hiddenBiases[i].resize(_hiddenLayerSizes[i], BIAS_START);
          _rng->getGaussianVector(_hiddenWeights[i], param1);
          nCurrentInputWidth = _hiddenLayerSizes[i];
        }
      }
      
      if(_nOutputUnits > 0){
        _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 1);
        _outputBiases.resize(_nOutputUnits, BIAS_START);
        _rng->getGaussianVector(_outputWeights, param1);
      }
      break;
    default:
      std::cout << "Don't know this type for weight initialisation!\n";
      break;
  }
  
  
  
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
        for(int iCol = 0; iCol < _nInputUnits; iCol++){
          _trainDataNormParam1[iCol] =  _trainDataFeedForwardValues[0][iCol * _nTrainDataRecords];
          for(int iRow = 1; iRow < _nTrainDataRecords; iRow++){
            delta = _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow] - _trainDataNormParam1[iCol];
            _trainDataNormParam1[iCol] += delta/(iRow+1);
            _trainDataNormParam2[iCol] += delta*(_trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow]-_trainDataNormParam1[iCol]);
          }
          _trainDataNormParam2[iCol] = sqrt(_trainDataNormParam2[iCol]/(_nTrainDataRecords-1));
        }
        
        for(int iCol = 0; iCol < _nInputUnits; iCol++){
          for(int iRow = 0; iRow < _nTrainDataRecords; iRow++){
            _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow] -= _trainDataNormParam1[iCol];
            if(_trainDataNormParam2[iCol] > MIN_DATA_RANGE){
              _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow] /= _trainDataNormParam2[iCol];
            }
          }
        }
        
        _trainDataNormType = nnet::DATA_STAN_NORM;

        break;
      case nnet::DATA_RANGE_BOUND:
        for(int iCol = 0; iCol < _nInputUnits; iCol++){
          _trainDataNormParam1[iCol] = _trainDataFeedForwardValues[0][iCol];
          _trainDataNormParam2[iCol] = _trainDataFeedForwardValues[0][iCol];

          for(int iRow = 1; iRow < _nTrainDataRecords; iRow++){
            _trainDataNormParam1[iCol] = std::min(_trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow],_trainDataNormParam1[iCol]);
            _trainDataNormParam2[iCol] = std::max(_trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow],_trainDataNormParam2[iCol]);
          }
        }
        for(int iCol = 0; iCol < _nInputUnits; iCol++){
          for(int iRow = 0; iRow < _nTrainDataRecords; iRow++){
            _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow] -= _trainDataNormParam1[iCol] + ((_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol])/2);
            if((_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol]) > MIN_DATA_RANGE){
            _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords) + iRow] /= (_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol]);
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
        for(int iCol = 0; iCol < _nNonTrainDataInputUnits; iCol++){
          for(int iRow = 0; iRow < _nNonTrainDataRecords; iRow++){
            _nonTrainDataFeedForwardValues[0][(iCol*_nNonTrainDataRecords)+iRow] -= _trainDataNormParam1[iCol];
            if(_trainDataNormParam2[iCol] > MIN_DATA_RANGE){
              _nonTrainDataFeedForwardValues[0][(iCol*_nNonTrainDataRecords)+iRow] /= _trainDataNormParam2[iCol];
            }
          }
        }
        _nonTrainDataNormType = nnet::DATA_STAN_NORM;
        break;
      case nnet::DATA_RANGE_BOUND:
        for(int iCol = 0; iCol < _nNonTrainDataInputUnits; iCol++){
          for(int iRow = 0; iRow < _nNonTrainDataRecords; iRow++){
            _nonTrainDataFeedForwardValues[0][(iCol*_nNonTrainDataRecords)+iRow] -= _trainDataNormParam1[iCol] + ((_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol])/2);
            if((_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol]) > MIN_DATA_RANGE){
              _nonTrainDataFeedForwardValues[0][(iCol*_nNonTrainDataRecords)+iRow] /= (_trainDataNormParam2[iCol]- _trainDataNormParam1[iCol]);
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


// We are not physically shuffling the data, but creating an index of ordering
// for use in batch optimisation
void nnet::shuffleTrainData(){
  std::vector<std::vector<int> > indataShuffle;
  std::vector<int> iClassShuffle;
  indataShuffle.resize(_nOutputUnits);
  iClassShuffle.resize(_nOutputUnits);
  if(_trainDataLoaded){
    if(_nOutputUnits > 1){
      for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
        indataShuffle[iLabelClass].resize(0);
        iClassShuffle[iLabelClass] = iLabelClass;
      }
      for(int iLabel = 0; iLabel < _nTrainDataRecords; iLabel++ ){
        for(int iLabelClass = 0; iLabelClass < _nOutputUnits; iLabelClass++){
          if(_trainDataLabels[iLabelClass * _nTrainDataRecords +  iLabel ] > 0.8){
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
    }else{
      _trainDataShuffleIndex.resize(_nTrainDataRecords);
      for(int iRecord = 0; iRecord < _nTrainDataRecords; iRecord++){
        _trainDataShuffleIndex[iRecord] = iRecord;
      }
      _rng->getShuffled(_trainDataShuffleIndex);
    }
    _trainDataShuffled = true;
  }else{
    std::cout << "Shuffling but no train data loaded!\n";
  }
  return;
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

void nnet::activateUnitsAndCalcGradients(std::vector<double>& values,
                                         std::vector<double>& gradients,
                                         double nestorovNudge){
  if(values.size() != gradients.size()){
    std::cout << "****  Problem with mismatch value <-> gradient sizes\n";
  }
  switch (_activationType){
    case nnet::TANH_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        
        values[i] =  tanh(values[i]);
        gradients[i] =  (1-pow(values[i]+nestorovNudge,2)) ;
      }
      break;
    case LIN_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        gradients[i] =  values[i] + nestorovNudge;
      }
      break;
    case RELU_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        values[i] = fmax(0.0,values[i]);
        gradients[i] = (values[i]+nestorovNudge) > 0.0 ? 1 : 0.0;
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
      
      std::vector<double> numerator(_nOutputUnits);
      for(int iRow = 0; iRow < _nTrainDataRecords; iRow++){
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          if(iCol == 0){
            max = values[(iCol*_nTrainDataRecords)+iRow];
          }else{
            if(values[(iCol*_nTrainDataRecords)+iRow]>max){
              max = values[(iCol*_nTrainDataRecords)+iRow];
            }
          }
        }
        // An adjustment to avoid overflow
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          values[(iCol*_nTrainDataRecords)+iRow] -= max;
        }
        
        denominator = 0.0;
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          numerator[iCol] = exp(values[(iCol*_nTrainDataRecords)+iRow]);
          denominator  += numerator[iCol];
        }
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          values[(iCol*_nTrainDataRecords)+iRow] = numerator[iCol]/denominator;
        }
      }
      break;
  }
  
}

bool nnet::dataLoaded(){
  return _trainDataLoaded;
}


size_t nnet::getNTrainData(){
  return _nTrainDataRecords;
}

double nnet::getTrainDataCost(){
  double cost = 0.0;
  switch (_lossType){
    case nnet::MSE_LOSS_TYPE:
      cost = 0.0;
      for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
        for(int iRecord = 0; iRecord < _nTrainDataRecords; ++iRecord){
          cost += pow(_trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord] - _trainDataLabels[(iUnit*_nTrainDataRecords)+iRecord],2.0)/_nTrainDataRecords;
        }
      }

      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iRecord = 0; iRecord < _nTrainDataRecords; ++iRecord){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          if(_trainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
            cost -= log( _trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord]);
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
  for(int iRecord = 0; iRecord < _nTrainDataRecords; ++iRecord){
    for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
      if(iUnit == 0){
        maxProb = _trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord];
        if(_trainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(_trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord] > maxProb){
          maxProb = _trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord];
          // These should be 0 or 1, but using a condition on 0.5
          if(_trainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
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
  switch (_lossType){
    case nnet::MSE_LOSS_TYPE:
      cost = 0.0;
      for(int iRecord = 0; iRecord < _nTrainDataRecords; ++iRecord){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          cost += pow(_trainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord] - _trainDataLabels[(iUnit*_nTrainDataRecords)+iRecord],2.0)/_nTrainDataRecords;
        }
      }
      
      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iRecord = 0; iRecord < _nNonTrainDataRecords; ++iRecord){
        for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
          if(_nonTrainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
            cost -= log( _nonTrainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord]);
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
  for(int iRecord = 0; iRecord < _nNonTrainDataRecords; ++iRecord){
    for(int iUnit = 0; iUnit < _nOutputUnits; ++iUnit){
      if(iUnit == 0){
        maxProb = _nonTrainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord];
        if(_nonTrainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(_nonTrainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord] > maxProb){
          maxProb = _nonTrainGeneratedLabels[(iUnit*_nTrainDataRecords)+iRecord];
          if(_nonTrainDataLabels[(iUnit*_nTrainDataRecords)+iRecord] > 0.5){
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

void nnet::feedForwardTrainData(bool calcGradients,
                                double nestorovAdj){
  std::vector<double> tempMatrixForLabels;
  
  if(_trainDataLoaded && _weightsInitialised){
    _nonTrainLabelsGenerated = false;
    
    flowDataThroughNetwork(_trainDataFeedForwardValues,
                           tempMatrixForLabels,
                           calcGradients,
                           nestorovAdj);
    
    _trainGeneratedLabels = tempMatrixForLabels;
    
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

void nnet::feedForwardTrainData(bool calcGradients,
                                double nestorovAdj,
                                std::vector<std::vector<int> >& dropouts){
  std::vector<double> tempMatrixForLabels;
  
  if(_trainDataLoaded && _weightsInitialised){
    _nonTrainLabelsGenerated = false;
    
    flowDataThroughNetwork(_trainDataFeedForwardValues,
                           tempMatrixForLabels,
                           calcGradients,
                           nestorovAdj);
    
    _trainGeneratedLabels = tempMatrixForLabels;
    
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



//void nnet::feedForwardTrainData_old(){
//  size_t nInputRows, nInputCols, nWeightsCols;
//  std::vector<double> tempForBiasInAndMatmulOut;
//  
//  if(_trainDataLoaded && _weightsInitialised){
//    
//    _trainLabelsGenerated = false;
//    nInputRows = _nTrainDataRecords;
//    nInputCols = _nInputUnits;
//    if( _trainDataFeedForwardValues.size() > 1){
//      
//      _trainDataFeedForwardValues.resize(1);
//    }
//    //for(int iLayer = 0; iLayer < _hiddenLayerSizes.size(); iLayer++)
//    //{
//    //  std::cout << "Layer " << iLayer << " size: " << _hiddenLayerSizes[iLayer] << std::endl;
//    //}
//    
//    for(int iLayer = 0; iLayer < _hiddenLayerSizes.size(); iLayer++)
//    {
//      //std::cout << "Feed forward: calculating layer " << iLayer << " size: " << _hiddenLayerSizes[iLayer] << std::endl;
//      //tempForBiasInAndMatmulOut.resize(0);
//      nWeightsCols =  _hiddenLayerSizes[iLayer];
//      
//      tempForBiasInAndMatmulOut.assign(nInputRows*_hiddenBiases[iLayer].size(),0.0);
//      for(int i = 0; i < nInputRows; ++i){
//        for(int j = 0; j <  _hiddenBiases[iLayer].size(); j++){
//          tempForBiasInAndMatmulOut[(i*_hiddenBiases[iLayer].size())+j] = _hiddenBiases[iLayer][j];
//        }
//      }
//      
//      //    std::cout << "x values" << std::endl;
//      //    for(int i = 0; i < nInputRows; i++){
//      //      for(int j = 0; j < nInputCols; j++){
//      //        std::cout << _trainDataFeedForwardValues[iLayer][(i*nInputCols) + j] << "|" ;
//      //      }
//      //      std::cout << std::endl;
//      //    }
//      //    std::cout << "Check: " << nInputCols << " : " << nWeightsCols << std::endl;
//      //    std::cout << "y values" << std::endl;
//      //    for(int i = 0; i < nInputCols; i++){
//      //      for(int j = 0; j < nWeightsCols; j++){
//      //        std::cout << _hiddenWeights[iLayer][(i*nWeightsCols) + j] << "|" ;
//      //      }
//      //      std::cout << std::endl;
//      //    }
//      
//      
//      //    std::cout << nInputRows << " x " << nInputCols << " by " << nInputCols << " x " << nWeightsCols << " bias " << tempForBiasInAndMatmulOut.size() << std::endl;
//      //   std::cout << "Tempbias\n";
//      //    for(int i = 0; i < nInputRows; i++){
//      //      for(int j = 0; j < nWeightsCols; j++){
//      //        std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//      //      }
//      //      std::cout << std::endl;
//      //    }
//      
//      mat_ops::matMul(nInputRows, nInputCols, _trainDataFeedForwardValues[iLayer], nWeightsCols, _hiddenWeights[iLayer], tempForBiasInAndMatmulOut);
//      
//      //    std::cout << "Matmul" << std::endl;
//      //    for(int i = 0; i < nInputRows; i++){
//      //      for(int j = 0; j < nWeightsCols; j++){
//      //        std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//      //      }
//      //      std::cout << std::endl;
//      //    }
//      _trainDataFeedForwardValues.resize(iLayer+2);
//      _trainDataFeedForwardValues[iLayer+1] = tempForBiasInAndMatmulOut;
//      _hiddenGradients.resize(iLayer+1);
//      _hiddenGradients[iLayer].assign(tempForBiasInAndMatmulOut.size(),0.0);
//      //    std::cout << "Check " << _trainDataFeedForwardValues.size() << " " << tempForBiasInAndMatmulOut.size() << " \n";
//      
//      
//      
//      activateUnitsAndCalcGradients(_trainDataFeedForwardValues[iLayer+1],_hiddenGradients[iLayer]);
//      //Ninputrows doesn't change
//      nInputCols = nWeightsCols;
//    }
//    
//    
//    //  std::cout << "Feed forward: calculating output layer " << std::endl;
//    nWeightsCols =  _nOutputUnits;
//    tempForBiasInAndMatmulOut.resize(nInputRows*_outputBiases.size());
//    for(int i = 0; i < nInputRows; ++i){
//      for(int j = 0; j <  _outputBiases.size(); j++){
//        tempForBiasInAndMatmulOut[(i*_outputBiases.size()) + j] = _outputBiases[j];
//      }
//    }
//    
//    
//    //std::cout << nInputRows << " x " << nInputCols << " by " << nInputCols << " x " << nWeightsCols << " bias " << tempForBiasInAndMatmulOut.size() << std::endl;
//    //std::cout << "Tempbias\n";
//    //  for(int i = 0; i < nInputRows; i++){
//    //    for(int j = 0; j < nWeightsCols; j++){
//    //      std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//    //    }
//    //    std::cout << std::endl;
//    //  }
//    mat_ops::matMul(nInputRows, nInputCols , _trainDataFeedForwardValues[_trainDataFeedForwardValues.size()-1] , nWeightsCols, _outputWeights, tempForBiasInAndMatmulOut);
//    //
//    //      std::cout << "> Matmul" << std::endl;
//    //      for(int i = 0; i < nInputRows; i++){
//    //        for(int j = 0; j < nWeightsCols; j++){
//    //          std::cout << tempForBiasInAndMatmulOut[(i*nWeightsCols) + j] << "|" ;
//    //        }
//    //        std::cout << std::endl;
//    //      }
//    
//    
//    _trainGeneratedLabels = tempForBiasInAndMatmulOut;
//    activateOutput(_trainGeneratedLabels);
//    _trainLabelsGenerated = true;
//  }else{
//    if (!_trainDataLoaded){
//      std::cout << "No data loaded\n";
//    }
//    if (!_weightsInitialised){
//      std::cout << "Weights not initialised\n";
//    }
//  }
//  //printOutValues();
//  return;
//}

void nnet::feedForwardNonTrainData(){
  std::vector<double> tempMatrixForLabels;
  bool dataCorrect = true;
  
  if(!_nonTrainDataLoaded){
    std::cout << "No 'non-train' data loaded!\n";
    dataCorrect = false;
  }else{
    if(_nNonTrainDataRecords == 0){
      std::cout << "No 'non-train' data loaded!\n";
      dataCorrect = false;
    }
  }
  if(_trainDataNormType == nnet::DATA_NORM_NONE && _nonTrainDataNormType != nnet::DATA_NORM_NONE){
    std::cout << "Train data was normalised, but 'non-train' was not!\n";
    dataCorrect = false;
  }
  if(_nonTrainDataNormType != nnet::DATA_NORM_NONE && _nonTrainDataNormType == nnet::DATA_NORM_NONE ){
    std::cout << "Non-train data was normalised, but train was not!\n";
    dataCorrect = false;
  }
  if(_trainDataPCA & !_nonTrainDataPCA){
    std::cout << "Train data was PCA'd but 'non-train' was not!\n";
    dataCorrect = false;
  }
  if(_nonTrainDataPCA & !_trainDataPCA ){
    std::cout << "Non-train data was PCA'd, but train was not!\n";
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
    
    flowDataThroughNetwork(_nonTrainDataFeedForwardValues, tempMatrixForLabels, false, 0.0);
    
    _nonTrainGeneratedLabels = tempMatrixForLabels;
    
    _nonTrainLabelsGenerated = true;
  }else{
    if (!_nonTrainDataLoaded){
      std::cout << "No data loaded\n";
    }
    if (!_weightsInitialised){
      std::cout << "Weights not initialised\n";
    }
  }
  //printOutValues();
  return;
}

void nnet::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                                  std::vector<double>& dataflowMatrix,
                                  bool calcTrainGradients,
                                  double nesterovAdj){
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
    for(int iRow = 0; iRow < nInputRows; ++iRow){
      for(int iCol = 0; iCol <  _hiddenBiases[iLayer].size(); iCol++){
        dataflowMatrix[(iCol*nInputRows)+iRow] = _hiddenBiases[iLayer][iCol];
      }
    }
    
    mat_ops::matMul(nInputRows, nInputCols, dataflowStages[iLayer], nWeightsCols, _hiddenWeights[iLayer], dataflowMatrix);
    
    dataflowStages.resize(iLayer+2);
    dataflowStages[iLayer+1] = dataflowMatrix;
    
    if(calcTrainGradients){
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(dataflowMatrix.size(),0.0);
      activateUnitsAndCalcGradients(dataflowStages[iLayer+1],
                                    _hiddenGradients[iLayer],
                                    nesterovAdj);
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
  for(int iRow = 0; iRow < nInputRows; ++iRow){
    for(int iCol = 0; iCol <  _outputBiases.size(); iCol++){
      dataflowMatrix[(iCol*nInputRows) + iRow] = _outputBiases[iCol];
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



void nnet::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                                  std::vector<double>& dataflowMatrix,
                                  bool calcTrainGradients,
                                  double nesterovAdj,
                                  std::vector<std::vector<int> >& dropouts){
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
      activateUnitsAndCalcGradients(dataflowStages[iLayer+1],
                                    _hiddenGradients[iLayer],
                                    nesterovAdj);
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

bool nnet::backProp(size_t nBatchIndicator,
                    double wgtLearnRate,
                    double biasLearnRate,
                    size_t nEpoch){
  bool allOk = true;
  size_t nInputs, nOutputs, iDataStart, iDataStop;
  
  size_t nBatchSize;
  double initialCost = 0.0, cost = 0.0, testCost = 0.0;
  double momentum_mu = _momMu;
  
  std::ostringstream oss;
  
  _epochTrainCostUpdates.resize(0);
  _epochTestCostUpdates.resize(0);
  
  if (nBatchIndicator < 1){
    nBatchSize = _nTrainDataRecords;
  }else{
    nBatchSize = nBatchIndicator;
  }
  
  if(_doTestCost){
    if (!_nonTrainDataLoaded){
      _doTestCost = false;
      std::cout << "There is no Non-Train data load, cannot calculate test error\n";
    }
    if(!_trainDataLabelsLoaded){
      _doTestCost = false;
      std::cout << "There is no Non-Train data Labels loaded, cannot calculate test error\n";
    }
  }
  
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
  
  std::vector<std::vector<int> > dropoutMask;
  size_t iDropoutLayerIndex = 0;
  
  std::vector<double> *inData, *outData, *forwardWeightsUpdate; //, *biases, *gradients;
  
  hiddenWeightsUpdate.resize(_hiddenWeights.size());
  hiddenBiasesUpdate.resize(_hiddenBiases.size());
  
  for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
    hiddenWeightsUpdate[iWgtCount].resize(_hiddenWeights[iWgtCount].size(),0.0);
    hiddenBiasesUpdate[iWgtCount].resize(_hiddenBiases[iWgtCount].size(),0.0);
  }
  
  if(_doDropout){
    dropoutMask.resize(_hiddenLayerSizes.size());
    for(int iLayer = 0; iLayer < _hiddenLayerSizes.size();  ++iLayer) {
      dropoutMask[iLayer].resize(_hiddenLayerSizes[iLayer]);
    }
  }
  
  if(_doMomentum){
    hiddenWeightsMomentum.resize(_hiddenWeights.size());
    hiddenBiasesMomentum.resize(_hiddenBiases.size());
    
    for(size_t iWgtCount= 0; iWgtCount < _hiddenWeights.size() ; iWgtCount++){
      hiddenWeightsMomentum[iWgtCount].resize(_hiddenWeights[iWgtCount].size(),0.0);
      hiddenBiasesMomentum[iWgtCount].resize(_hiddenBiases[iWgtCount].size(),0.0);
    }
    
    outputWeightsMomentum.resize(_outputWeights.size(), 0.0);
    outputBiasesMomentum.resize(_outputBiases.size(), 0.0);
    
  }
  
  feedForwardTrainData(true,0.0);
  
  outData = &_trainGeneratedLabels;
  initialCost = getTrainDataCost();
  _epochTrainCostUpdates.push_back(initialCost);
  std::cout << "Initial Cost: " << initialCost <<  std::endl;
  if(_doTestCost){
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
    if(_doDropout){
      for(int iLayer = 0; iLayer < _hiddenLayerSizes.size();  ++iLayer) {
        if(iLayer == 0){
          _rng->getBernoulliVector(dropoutMask[iLayer],_inputDropoutRate);
        }else{
          _rng->getBernoulliVector(dropoutMask[iLayer],_hiddenDropoutRate);
        }
      }
      
         }
    if(_doMomentum && iEpoch > 0){
      if(_momDecaySchedule > 0){
        if(iEpoch% _momDecaySchedule == 0){
          momentum_mu += _momDecay * (_momFinal - momentum_mu);
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
      
      if(_doDropout){
        feedForwardTrainData(true, 0.0, dropoutMask);
        iDropoutLayerIndex = _hiddenLayerSizes.size()-1;
      }else{
        feedForwardTrainData(true, 0.0);
      }
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
      if(_hiddenLayerSizes.size() > 0){
        nInputs = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
      }else{
        nInputs = _nInputUnits;
      }
      nOutputs = _nOutputUnits;
      
      for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
        size_t iDataInBatch = 0;
        if(_trainDataShuffled){
          iDataInBatch = _trainDataShuffleIndex[iDataIndex];
        }else{
          iDataInBatch = iDataIndex;
        }
        switch (_outputType){
          case  nnet::LIN_OUT_TYPE:
          case  nnet::SMAX_OUT_TYPE:
            if(_doDropout){
              // Softmax with cross entropy has a simple derviative form for the weights on the input to the output units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                if(dropoutMask[iDropoutLayerIndex][iInput] == 0){
                  for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                    outputWeightsUpdate[(iOutput *nInputs) +  iInput] += (*inData)[(iInput*_nTrainDataRecords) + iDataInBatch] * (_trainGeneratedLabels[(iOutput*_nTrainDataRecords) + iDataInBatch]-_trainDataLabels[(iOutput*_nTrainDataRecords) + iDataInBatch]);
                  }
                }
              }
              // Bias units have a 1 for the input weights of the unit, otherwise same as above
              for(int iBias = 0; iBias < nOutputs; ++iBias){
                 outputBiasesUpdate[iBias] += (_trainGeneratedLabels[(iBias *_nTrainDataRecords)  + iDataInBatch]-_trainDataLabels[(iBias *_nTrainDataRecords)  + iDataInBatch]);
              }
            }else{
              // Softmax with cross entropy has a simple derviative form for the weights on the input to the output units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                  outputWeightsUpdate[(iOutput *nInputs) +  iInput] += (*inData)[(iInput*_nTrainDataRecords) + iDataInBatch] * (_trainGeneratedLabels[(iOutput*_nTrainDataRecords) + iDataInBatch]-_trainDataLabels[(iOutput*_nTrainDataRecords) + iDataInBatch]);
                  
                }
              }
              // Bias units have a 1 for the input weights of the unit, otherwise same as above
              for(int iBias = 0; iBias < nOutputs; ++iBias){
                outputBiasesUpdate[iBias] += (_trainGeneratedLabels[(iBias *_nTrainDataRecords) + iDataInBatch] - _trainDataLabels[(iBias *_nTrainDataRecords)  + iDataInBatch]);
              }
            }
            break;
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
          switch (_outputType){
            case  nnet::LIN_OUT_TYPE:
            case  nnet::SMAX_OUT_TYPE:
              if(_doDropout){
                iDropoutLayerIndex--;
                // We need the derivative of the weight multiplication on the input; by the activation dervi;
                // all chained with the derivatives of the weights on the output side of the units
                for(int iInput = 0; iInput < nInputs; ++iInput){
                  if(dropoutMask[iDropoutLayerIndex][iInput] == 0){
                    for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                      if(dropoutMask[iDropoutLayerIndex][iOutput] == 0){
                        for(int iNext = 0; iNext < nOutputOutputs; iNext++){
                          //hiddenWeightsUpdate[iWgtMat][(iInput*nOutputs)+iOutput]  += (1.0/nInputs)*(*inData)[(iDataInBatch*nInputs)+iInput]* _hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nOutputOutputs)+iNext];
                          hiddenWeightsUpdate[iWgtMat][(iOutput*nInputs) + iInput]  += (*inData)[(iInput * _nTrainDataRecords) +iDataInBatch]* _hiddenGradients[iWgtMat][(iOutput*_nTrainDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)[(iNext*nOutputs) + iOutput];
                        }
                      }
                    }
                  }
                }
                
                // Biases have 1 for the weight multiplicaiton on input, otherwise the same
                for(int iInput = 0; iInput < 1; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                    if(dropoutMask[iDropoutLayerIndex][iOutput] == 0){
                      for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                        // hiddenBiasesUpdate[iWgtMat][iOutput] += (1.0/nInputs)*_hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)  [(iOutput*nOutputOutputs)+iNext];
                        hiddenBiasesUpdate[iWgtMat][iOutput] += _hiddenGradients[iWgtMat][(iOutput* _nTrainDataRecords) +iDataInBatch]*(*forwardWeightsUpdate)  [(iNext*nOutputs) + iOutput];
                      }
                    }
                  }
                }
              }else{
                // We need the derivative of the weight multiplication on the input; by the activation dervi;
                // all chained with the derivatives of the weights on the output side of the units
                for(int iInput = 0; iInput < nInputs; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                    for(int iNext = 0; iNext < nOutputOutputs; iNext++){
                      //hiddenWeightsUpdate[iWgtMat][(iInput*nOutputs)+iOutput]  += (1.0/nInputs)*(*inData)[(iDataInBatch*nInputs)+iInput]* _hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)[(iOutput*nOutputOutputs)+iNext];
                      hiddenWeightsUpdate[iWgtMat][(iOutput*nInputs) + iInput]  += (*inData)[(iInput * _nTrainDataRecords)+iDataInBatch]* _hiddenGradients[iWgtMat][(iOutput*_nTrainDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)[iNext*nOutputs + iOutput];
                    }
                  }
                }
              
                // Biases have 1 for the weight multiplicaiton on input, otherwise the same
                for(int iInput = 0; iInput < 1; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                    for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                      //hiddenBiasesUpdate[iWgtMat][iOutput] += (1.0/nInputs)*_hiddenGradients[iWgtMat][iDataInBatch*nOutputs+iOutput]*(*forwardWeightsUpdate)  [(iOutput*nOutputOutputs)+iNext];
                      hiddenBiasesUpdate[iWgtMat][iOutput] += _hiddenGradients[iWgtMat][(iOutput * _nTrainDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)  [(iNext *nOutputs) + iOutput];
                    }
                  }
                }
              }
              break;
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
        if(_doMomentum){
          outputWeightsMomentum[iWgt] = momentum_mu * outputWeightsMomentum[iWgt] - wgtLearnRate * outputWeightsUpdate[iWgt];
          _outputWeights[iWgt] += outputWeightsMomentum[iWgt];
        }else{
          _outputWeights[iWgt] -= wgtLearnRate*outputWeightsUpdate[iWgt];
        }
      }
      for(int iBias = 0; iBias < _outputBiases.size(); iBias++){
        if(_doMomentum){
          outputBiasesMomentum[iBias] = momentum_mu * outputBiasesMomentum[iBias] - biasLearnRate * outputBiasesUpdate[iBias];
          _outputBiases[iBias] += outputWeightsMomentum[iBias];
        }else{
          _outputBiases[iBias] -= biasLearnRate*outputBiasesUpdate[iBias];
        }
      }
      // Back through the hidden layers updating the weights
      for(int iWgtMat = 0; iWgtMat < _hiddenWeights.size(); iWgtMat++){
        for(int iWgt = 0; iWgt < _hiddenWeights[iWgtMat].size(); iWgt++){
          if(_doMomentum){
            hiddenWeightsMomentum[iWgtMat][iWgt] = momentum_mu * hiddenWeightsMomentum[iWgtMat][iWgt] - wgtLearnRate * hiddenWeightsUpdate[iWgtMat][iWgt];
            _hiddenWeights[iWgtMat][iWgt] += hiddenWeightsMomentum[iWgtMat][iWgt] ;
          }else{
            _hiddenWeights[iWgtMat][iWgt] -= wgtLearnRate*hiddenWeightsUpdate[iWgtMat][iWgt];
            
          }
        }
        for(int iWgt = 0; iWgt < _hiddenBiases[iWgtMat].size(); iWgt++){
          if(_doMomentum){
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
    if(std::isnan(cost)){
      std::cout << "Nan cost so quitting!\n";
      allOk = false;
      break;
    }else{
      _epochTrainCostUpdates.push_back(cost);
      switch (_outputType){
        case  nnet::LIN_OUT_TYPE:
          std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << std::endl;
          break;
        case  nnet::SMAX_OUT_TYPE:
          std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << "-- Accuracy " << getTrainDataAccuracy() << "  ("  << getTrainDataAccuracy() * _nTrainDataRecords << ")" << std::endl;
          break;
      }
      if(_doTestCost){
        feedForwardNonTrainData();
        testCost = getNonTrainDataCost();
        _epochTestCostUpdates.push_back(testCost);
        std::cout << "Test Cost: " << testCost <<  std::endl;
      }
      if(_epochTrainCostUpdates.size()>2){
        if((_epochTrainCostUpdates[_epochTrainCostUpdates.size()-1] > _epochTrainCostUpdates[_epochTrainCostUpdates.size()-2]) &&
           (_epochTrainCostUpdates[_epochTrainCostUpdates.size()-2] > _epochTrainCostUpdates[_epochTrainCostUpdates.size()-3])){
          std::cout << "Training cost rose twice in a row\n";
          //allOk = false;
          //break;
          
        }
      }
    }
  }

  if(allOk){
    feedForwardTrainData(false,0.0);
    cost = getTrainDataCost();
    std::cout <<  " Cost went from " << initialCost << " to " <<  cost << std::endl;
  }
  return allOk;
}

bool nnet::loadTrainDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  
  std::vector<double> indata;
  
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
    _nInputUnits = nDelims + 1;
    _nTrainDataRecords = nRecords;
    
    _trainDataFeedForwardValues[0].resize(indata.size());
    
    for(size_t iCol = 0; iCol < _nInputUnits; iCol++){
      for(size_t iRow = 0; iRow < _nTrainDataRecords; iRow++){
        _trainDataFeedForwardValues[0][iCol*_nTrainDataRecords + iRow] = indata[iRow*_nInputUnits + iCol];
      }
    }
    _trainDataLoaded = true;
    _trainDataShuffled = false;
    
    
    std::cout << "Read " << _trainDataFeedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool nnet::loadTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  std::vector<double> indata;
  
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
    if(indata.size() != _nTrainDataRecords * (nDelims+1)){
      std::cout << "Expected " << _nTrainDataRecords * (nDelims+1) << " Labels, got " << indata.size() << "!" << std::endl;
      allOk = false;
    }
  }
  
  
  if(allOk){
    _nOutputUnits = nDelims + 1;
    
    _trainDataLabels.resize(indata.size());
    
    for(size_t iCol = 0; iCol < _nOutputUnits; iCol++){
      for(size_t iRow = 0; iRow < _nTrainDataRecords; iRow++){
        _trainDataLabels[iCol*_nTrainDataRecords + iRow] = indata[(iRow * _nOutputUnits)+iCol];
      }
    }
    
    _trainDataLabelsLoaded = true;
    
    std::cout << "Read " << _trainDataLabels.size() << " labels of " << nRecords << " by " << _nOutputUnits << std::endl;
  }
  return allOk;
};

bool nnet::loadNonTrainDataFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  std::vector<double> indata;
  
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
      std::cout << "Number of data is not an integer multiple of first line fields!\n";
      allOk = false;
    }

  }
  if(allOk){
    _nNonTrainDataRecords = nRecords;
    _nNonTrainDataInputUnits = nDelims + 1;
    
    _nonTrainDataFeedForwardValues.resize(indata.size());
    for(size_t iCol = 0; iCol < _nNonTrainDataInputUnits; iCol++){
      for(size_t iRow = 0; iRow < _nNonTrainDataRecords; iRow++){
        _nonTrainDataFeedForwardValues[0][iCol*_nNonTrainDataRecords + iRow] = indata[ (iRow*_nNonTrainDataInputUnits) + iCol];
      }
    }
    _nonTrainDataLoaded = true;
    
    std::cout << "Read " << _nonTrainDataFeedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
  }
  return allOk;
};

bool nnet::loadNonTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
  bool allOk = true;
  bool first = true;
  int nRecords = 0;
  size_t nDelims = 0;
  std::vector<double> indata;
  
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
    if(indata.size() != _nNonTrainDataRecords * (nDelims+1)){
      std::cout << "Expected " << _nNonTrainDataRecords * (nDelims+1) << "Labels, got " << indata.size() << "!" << std::endl;
      allOk = false;
    }
  }
  
  if(allOk){
    _nNonTrainDataOutputUnits = nDelims + 1;
    _nonTrainDataLabels.resize(indata.size());
    
    // Flip from row major to column major
    for(size_t iCol = 0; iCol < _nNonTrainDataOutputUnits; iCol++){
      for(size_t iRow = 0; iRow < _nNonTrainDataRecords; iRow++){
        _nonTrainDataLabels[iCol*_nNonTrainDataRecords + iRow] = indata[iRow*_nNonTrainDataOutputUnits + iCol];
      }
    }
    
    _nonTrainDataLabelsLoaded = true;
    
    std::cout << "Read " << _nonTrainDataLabels.size() << " labels of " << nRecords << " by " << _nOutputUnits << std::endl;
  }
  return allOk;
};

void nnet::writeWeightValues(){
  size_t nRows = 0, nCols = 0;
  nRows = _nInputUnits;
  for(int i = 0; i < _hiddenWeights.size(); i++){
        nCols = _hiddenLayerSizes[i];
    std::ostringstream oss;
    oss << _outputDir << "weights" << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _hiddenWeights[i],nRows,nCols);
    oss.str(std::string());
    oss << _outputDir << "bias" << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _hiddenBiases[i],1,nCols);
    nRows = _hiddenLayerSizes[i];
 }
  nCols = _nOutputUnits;
  std::ostringstream oss;
  oss << _outputDir << "weights" << _hiddenWeights.size() << ".csv";
  mat_ops::writeMatrix(oss.str(), _outputWeights,nRows,nCols);
  
  oss.str(std::string());
  oss << _outputDir << "bias" << _hiddenWeights.size() << ".csv";
  mat_ops::writeMatrix(oss.str(), _outputBiases,1,nCols);
  
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
  std::cout << "### Geometry ###\n";
  std::cout << "Input units: " << _nInputUnits << std::endl;
  if(_hiddenLayerSizes.size()>0){
    for(int iLayer =0; iLayer < _hiddenLayerSizes.size(); iLayer++) {
      std::cout << "Layer " << iLayer + 1 << " units: " << _hiddenLayerSizes[iLayer] << std::endl;
    }
  }else{
    std::cout << "No hidden Layers\n";
  }
  std::cout << "Output units: " << _nOutputUnits << std::endl;
}

void nnet::printTrainLabels(size_t nRecords){
  if(_trainDataLabelsLoaded){
    if(nRecords == 0){
      nRecords = _nTrainDataRecords;
    }else{
      nRecords = std::min(nRecords, _nTrainDataRecords);
    }
    std::cout << "Printing Labels" << std::endl;
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << "Record " << iRow + 1 << ": ";
      for(int iCol = 0; iCol < _nOutputUnits; iCol++){
        std::cout << _trainDataLabels[(iCol*_nTrainDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No Labels Loaded" << std::endl;
  }
}

void nnet::printNonTrainLabels(size_t nRecords){
  if(_nonTrainDataLabelsLoaded){
    if(nRecords == 0){
      nRecords = _nNonTrainDataRecords;
    }else{
      nRecords = std::min(nRecords, _nNonTrainDataRecords);
    }
    std::cout << "Printing Non Train Labels" << std::endl;
    for(int i = 0; i < nRecords; i++){
      std::cout << "Record " << i + 1 << ": ";
      for(int j = 0; j < _nNonTrainDataOutputUnits; j++){
        std::cout << _nonTrainDataLabels[(i*_nNonTrainDataOutputUnits)+j] << " | ";
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
          std::cout <<  "| " << *it;
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
        std::cout <<  "| " << *it;
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

void nnet::printTrainData(size_t nRecords){
  
  if(_trainDataLoaded){
    if(nRecords == 0){
      nRecords = _nTrainDataRecords;
    }else{
      nRecords = std::min(nRecords, _nTrainDataRecords);
    }

    size_t rowsToPrint = std::min(_nTrainDataRecords,nRecords);
    std::cout << "Printing data" << std::endl;
    for(int iRow = 0; iRow < rowsToPrint; iRow++){
      std::cout << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nInputUnits; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _trainDataFeedForwardValues[0][(iCol*_nTrainDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded" <<std::endl;
  }
}

void nnet::printNonTrainData(size_t nRecords){
  if(_nonTrainDataLoaded){
    if(nRecords == 0){
      nRecords = _nNonTrainDataRecords;
    }else{
      nRecords = std::min(nRecords, _nNonTrainDataRecords);
    }
    std::cout << "Printing data" << std::endl;
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nNonTrainDataInputUnits; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _nonTrainDataFeedForwardValues[0][(iCol*_nNonTrainDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No non-train data loaded" <<std::endl;
  }
}

void nnet::printOutputUnitValues(size_t nRecords){
  std::cout << "Printing the Output Unit Values" << std::endl;
  if(_trainLabelsGenerated){
    if(nRecords == 0){
      nRecords = _nTrainDataRecords;
    }else{
      nRecords = std::min(nRecords, _nTrainDataRecords);
    }
    
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nOutputUnits; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(2) << _trainGeneratedLabels[(iCol*_nTrainDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data created" <<std::endl;
  }
}

/* FIX THIS TO COLUMN MAJOR */
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


