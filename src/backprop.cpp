//
//  backprop.cpp
//  BasicNN
//
//  Created by Martin on 30/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#include "backprop.hpp"
#include "mat_ops.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>

#define BIAS_START 0.0

backpropper::backpropper(nnet *net){
  _net = net;
  _rng = new rng;
  _outputDir = "~/";
  _epochPrintSchedule = 1;
  
  _shuffleEachEpoch = true;
  _doDropout = false;
  _inputDropoutRate = 0.0;
  _hiddenDropoutRate = 0.0;
  _doMomentum =false;
  _momMu = 0.0;
  _momDecay = 0.0;
  _momDecaySchedule = 0;
  _momFinal = 0.0;
  
  _nBatchSizeIndicator = 1;
  _wgtLearnRate = 0.1;
  _biasLearnRate = 0.05;
  _nEpoch = 500;
  
  _trainDataShuffled = false;
  _hiddenGradients.resize(0);
  _hiddenGradients.resize(net->_hiddenWeights.size());
  for(int i = 0; i < net->_hiddenWeights.size(); i++){
    _hiddenGradients[i].resize(net->_hiddenWeights[i].size(),0.0);
  }
  
  _testdataLoaded = false;
  
}

backpropper::~backpropper(){
  delete _rng;
}

void backpropper::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}

void backpropper::setEpochPrintSchedule(size_t schedule){
  _epochPrintSchedule = schedule;
  return;
}

bool backpropper::setTestData(dataset *testdata){
  bool allOk = true;
  if(testdata->nRecords() == 0){
    allOk = false;
  }
  if(!testdata->labelsLoaded()){
    allOk =false;
  }
  if(allOk){
    _testdataLoaded = true;
    _testdata = testdata;
    _testdataFeedForwardValues.resize(1);
    _testdataFeedForwardValues[0] = _testdata->data();
  }else{
    allOk = false;
  }
  return allOk;
}

void backpropper::doTestCost(bool doTestCost){
  _doTestCost = doTestCost;
}

void backpropper::doDropout(bool doDropout){
  _doDropout = doDropout;
}

void backpropper::setShuffleData(bool doShuffle){
  _shuffleEachEpoch = doShuffle;
}

void backpropper::setDropoutRates(double inputDropout, double hiddenDropout){
  _inputDropoutRate = inputDropout;
  _hiddenDropoutRate = hiddenDropout;
  return;
}

void backpropper::setMomentum(bool doMomentum,
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

void backpropper::initialiseWeights(initialiseType initialtype, double param1){
  size_t nCurrentInputWidth = _net->_nInputUnits;
  
  // std::cout << "Initialising weights" << std::endl;
  _net->_hiddenWeights.resize(0);
  _net->_hiddenBiases.resize(0);
  
  switch (initialtype) {
    case INIT_CONST_TYPE:
      if(_net->_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _net->_hiddenLayerSizes.size(); i++) {
          _net->_hiddenWeights.resize(i+1);
          _net->_hiddenWeights[i].resize(nCurrentInputWidth*_net->_hiddenLayerSizes[i], param1);
          _net->_hiddenBiases.resize(i+1);
          _net->_hiddenBiases[i].resize(_net->_hiddenLayerSizes[i], BIAS_START);
          nCurrentInputWidth = _net->_hiddenLayerSizes[i];
        }
      }
      
      if(_net->_nOutputUnits > 0){
        _net->_outputWeights.resize(nCurrentInputWidth * _net->_nOutputUnits, 1);
        _net->_outputBiases.resize(_net->_nOutputUnits, BIAS_START);
      }
      break;
    case INIT_GAUSS_TYPE:
      if(_net->_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _net->_hiddenLayerSizes.size(); i++) {
          _net->_hiddenWeights.resize(i+1);
          _net->_hiddenWeights[i].resize(nCurrentInputWidth*_net->_hiddenLayerSizes[i]);
          _net->_hiddenBiases.resize(i+1);
          _net->_hiddenBiases[i].resize(_net->_hiddenLayerSizes[i], BIAS_START);
          _rng->getGaussianVector(_net->_hiddenWeights[i], param1);
          nCurrentInputWidth = _net->_hiddenLayerSizes[i];
        }
      }
      
      if(_net->_nOutputUnits > 0){
        _net->_outputWeights.resize(nCurrentInputWidth * _net->_nOutputUnits, 1);
        _net->_outputBiases.resize(_net->_nOutputUnits, BIAS_START);
        _rng->getGaussianVector(_net->_outputWeights, param1);
      }
      break;
    default:
      std::cout << "Don't know this type for weight initialisation!\n";
      break;
  }
}

// We are not physically shuffling the data, but creating an index of ordering
// for use in batch optimisation
void backpropper::shuffleTrainData(){
  std::vector<std::vector<int> > indataShuffle;
  std::vector<int> iClassShuffle;
  indataShuffle.resize(_net->_nOutputUnits);
  iClassShuffle.resize(_net->_nOutputUnits);
  if(_net->dataAndLabelsLoaded()){
    if(_net->_nOutputUnits > 1){
      for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
        indataShuffle[iLabelClass].resize(0);
        iClassShuffle[iLabelClass] = iLabelClass;
      }
      for(int iLabel = 0; iLabel < _net->_nDataRecords ; iLabel++ ){
        for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
          if(_net->_dataLabels[iLabelClass * _net->_nDataRecords +  iLabel ] > 0.8){
            indataShuffle[iLabelClass].push_back(iLabel);
            break;
          }
        }
      }
      for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
        if(indataShuffle[iLabelClass].size() > 1){
          _rng->getShuffled(indataShuffle[iLabelClass]);
        }
      }
      if(_net->_nOutputUnits > 1){
        _rng->getShuffled(iClassShuffle);
      }
      size_t maxClass = indataShuffle[0].size();
      for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
        maxClass = indataShuffle[0].size() > maxClass ? indataShuffle[0].size() : maxClass;
      }
      
      _dataShuffleIndex.resize(0);
      for(int iLabel = 0; iLabel < maxClass; iLabel++ ){
        for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
          if(iLabel < indataShuffle[iClassShuffle[iLabelClass]].size()){
            _dataShuffleIndex.push_back(indataShuffle[iClassShuffle[iLabelClass]][iLabel]);
          }
        }
      }
    }else{
      _dataShuffleIndex.resize(_net->_nDataRecords);
      for(int iRecord = 0; iRecord < _net->_nDataRecords; iRecord++){
        _dataShuffleIndex[iRecord] = iRecord;
      }
      _rng->getShuffled(_dataShuffleIndex);
    }
    _trainDataShuffled = true;
  }else{
    std::cout << "Shuffling but no train data loaded!\n";
  }
  return;
}

void backpropper::feedForwardTrainData(bool calcGradients,
                                double nestorovAdj){
  std::vector<double> tempMatrixForLabels;
  
  if(_net->_dataLoaded){
    _net->_labelsGenerated = false;
    
    flowDataThroughNetwork(_net->_feedForwardValues,
                           tempMatrixForLabels,
                           calcGradients,
                           nestorovAdj);
    
    _net->_generatedLabels = tempMatrixForLabels;
    
    _net->_labelsGenerated = true;
  }else{
    if (!_net->_dataLoaded){
      std::cout << "No data loaded\n";
    }
  }
  //printOutValues();
  return;
}


void backpropper::activateUnitsAndCalcGradients(std::vector<double>& values,
                                         std::vector<double>& gradients,
                                         double nestorovNudge){
  if(values.size() != gradients.size()){
    std::cout << "****  Problem with mismatch value <-> gradient sizes\n";
  }
  switch (_net->_activationType){
    case nnet::TANH_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        
        values[i] =  tanh(values[i]);
        gradients[i] =  (1-pow(values[i]+nestorovNudge,2)) ;
      }
      break;
    case nnet::LIN_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        gradients[i] =  values[i] + nestorovNudge;
      }
      break;
    case nnet::RELU_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        values[i] = fmax(0.0,values[i]);
        gradients[i] = (values[i]+nestorovNudge) > 0.0 ? 1 : 0.0;
      }
      break;
  }
  return;
};



void backpropper::feedForwardTrainData(bool calcGradients,
                                double nestorovAdj,
                                std::vector<std::vector<int> >& dropouts){
  std::vector<double> tempMatrixForLabels;
  
  if(_net->_dataLoaded){
    _net->_labelsGenerated = false;
    
    flowDataThroughNetwork(_net->_feedForwardValues,
                           tempMatrixForLabels,
                           calcGradients,
                           nestorovAdj);
    
    _net->_generatedLabels = tempMatrixForLabels;
    
    _net->_labelsGenerated = true;
  }else{
    if (!_net->_dataLoaded){
      std::cout << "No data loaded\n";
    }
  }
  return;
}

void backpropper::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                                  std::vector<double>& dataflowMatrix,
                                  bool calcTrainGradients,
                                  double nesterovAdj){
  size_t nInputRows, nInputCols, nWeightsCols;
  
  
  nInputCols = _net->_nInputUnits;
  nInputRows = dataflowStages[0].size()/nInputCols;
  
  if( dataflowStages.size() > 1){
    dataflowStages.resize(1);
  }
  
  for(int iLayer = 0; iLayer < _net->_hiddenLayerSizes.size(); iLayer++)
  {
    nWeightsCols =  _net->_hiddenLayerSizes[iLayer];
    
    dataflowMatrix.assign(nInputRows*_net->_hiddenBiases[iLayer].size(),0.0);
    for(int iRow = 0; iRow < nInputRows; ++iRow){
      for(int iCol = 0; iCol <  _net->_hiddenBiases[iLayer].size(); iCol++){
        dataflowMatrix[(iCol*nInputRows)+iRow] = _net->_hiddenBiases[iLayer][iCol];
      }
    }
    
    mat_ops::matMul(nInputRows, nInputCols, dataflowStages[iLayer], nWeightsCols, _net->_hiddenWeights[iLayer], dataflowMatrix);
    
    dataflowStages.resize(iLayer+2);
    dataflowStages[iLayer+1] = dataflowMatrix;
    
    if(calcTrainGradients){
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(dataflowMatrix.size(),0.0);
      activateUnitsAndCalcGradients(dataflowStages[iLayer+1],
                                    _hiddenGradients[iLayer],
                                    nesterovAdj);
    }else{
      _net->activateUnits(dataflowStages[iLayer+1]);
    }
    
    //Ninputrows doesn't change
    nInputCols = nWeightsCols;
  }
  
  //  std::cout << "Feed forward: calculating output layer " << std::endl;
  nWeightsCols =  _net->_nOutputUnits;
  dataflowMatrix.resize(0);
  dataflowMatrix = _net->_outputBiases;
  dataflowMatrix.resize(nInputRows*_net->_outputBiases.size());
  for(int iRow = 0; iRow < nInputRows; ++iRow){
    for(int iCol = 0; iCol <  _net->_outputBiases.size(); iCol++){
      dataflowMatrix[(iCol*nInputRows) + iRow] = _net->_outputBiases[iCol];
    }
  }
  
  mat_ops::matMul(nInputRows,
                  nInputCols ,
                  dataflowStages[dataflowStages.size()-1] ,
                  nWeightsCols,
                  _net->_outputWeights,
                  dataflowMatrix);
  
  _net->activateOutput(dataflowMatrix);
  
  return;
}

void backpropper::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                                  std::vector<double>& dataflowMatrix,
                                  bool calcTrainGradients,
                                  double nesterovAdj,
                                  std::vector<std::vector<int> >& dropouts){
  size_t nInputRows, nInputCols, nWeightsCols;
  
  
  nInputCols = _net->_nInputUnits;
  nInputRows = dataflowStages[0].size()/nInputCols;
  
  if( dataflowStages.size() > 1){
    dataflowStages.resize(1);
  }
  
  for(int iLayer = 0; iLayer < _net->_hiddenLayerSizes.size(); iLayer++){
    nWeightsCols =  _net->_hiddenLayerSizes[iLayer];
    
    dataflowMatrix.assign(nInputRows*_net->_hiddenBiases[iLayer].size(),0.0);
    for(int i = 0; i < nInputRows; ++i){
      for(int j = 0; j <  _net->_hiddenBiases[iLayer].size(); j++){
        dataflowMatrix[(i*_net->_hiddenBiases[iLayer].size())+j] = _net->_hiddenBiases[iLayer][j];
      }
    }
    
    mat_ops::matMul(nInputRows, nInputCols, dataflowStages[iLayer], nWeightsCols, _net->_hiddenWeights[iLayer], dataflowMatrix);
    
    dataflowStages.resize(iLayer+2);
    dataflowStages[iLayer+1] = dataflowMatrix;
    
    if(calcTrainGradients){
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(dataflowMatrix.size(),0.0);
      activateUnitsAndCalcGradients(dataflowStages[iLayer+1],
                                    _hiddenGradients[iLayer],
                                    nesterovAdj);
    }else{
      _net->activateUnits(dataflowStages[iLayer+1]);
    }
    
    
    //Ninputrows doesn't change
    nInputCols = nWeightsCols;
  }
  
  //  std::cout << "Feed forward: calculating output layer " << std::endl;
  nWeightsCols =  _net->_nOutputUnits;
  dataflowMatrix.resize(0);
  dataflowMatrix = _net->_outputBiases;
  dataflowMatrix.resize(nInputRows*_net->_outputBiases.size());
  for(int i = 0; i < nInputRows; ++i){
    for(int j = 0; j <  _net->_outputBiases.size(); j++){
      dataflowMatrix[(i*_net->_outputBiases.size()) + j] = _net->_outputBiases[j];
    }
  }
  
  mat_ops::matMul(nInputRows,
                  nInputCols ,
                  dataflowStages[dataflowStages.size()-1] ,
                  nWeightsCols,
                  _net->_outputWeights,
                  dataflowMatrix);
  
  _net->activateOutput(dataflowMatrix);
  
  return;
}


bool backpropper::doBackPropOptimise(size_t nBatchIndicator,
                    double wgtLearnRate,
                    double biasLearnRate,
                    size_t nEpoch){
  bool allOk = true;
  size_t nInputs, nOutputs, iDataStart, iDataStop;
  
  size_t nBatchSizeTarget;
  double initialCost = 0.0, cost = 0.0;
  double momentum_mu = _momMu;
  
  std::ostringstream oss;
  
  _epochTrainCostUpdates.resize(0);
  _epochTestCostUpdates.resize(0);
  
  if (nBatchIndicator < 1){
    nBatchSizeTarget = _net->_nDataRecords;
  }else{
    nBatchSizeTarget = nBatchIndicator;
  }
  
  if(_doTestCost){
    if (!_testdataLoaded){
      _doTestCost = false;
      std::cout << "There is no Non-Train data load, cannot calculate test error\n";
    }
  }
  //http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent
  //http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  //http://neuralnetworksanddeeplearning.com/chap3.html#softmax
  size_t nOutputOutputs;
  std::vector<std::vector<double> > hiddenWeightsUpdate;
  std::vector<std::vector<double> > hiddenBiasesUpdate;
  
  std::vector<std::vector<double> > hiddenWeightsMomentum;
  std::vector<std::vector<double> > hiddenBiasesMomentum;
  
  std::vector<double> outputWeightsUpdate(_net->_outputWeights.size(), 0.0);
  std::vector<double> outputBiasesUpdate(_net->_outputBiases.size(), 0.0);
  
  std::vector<double> outputWeightsMomentum;
  std::vector<double> outputBiasesMomentum;
  
  std::vector<std::vector<int> > dropoutMask;
  size_t iDropoutLayerIndex = 0;
  
  std::vector<double> *inData, *outData, *forwardWeightsUpdate; //, *biases, *gradients;
  
  hiddenWeightsUpdate.resize(_net->_hiddenWeights.size());
  hiddenBiasesUpdate.resize(_net->_hiddenBiases.size());
  
  for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
    hiddenWeightsUpdate[iWgtCount].resize(_net->_hiddenWeights[iWgtCount].size(),0.0);
    hiddenBiasesUpdate[iWgtCount].resize(_net->_hiddenBiases[iWgtCount].size(),0.0);
  }
  
  if(_doDropout){
    dropoutMask.resize(_net->_hiddenLayerSizes.size());
    for(int iLayer = 0; iLayer < _net->_hiddenLayerSizes.size();  ++iLayer) {
      dropoutMask[iLayer].resize(_net->_hiddenLayerSizes[iLayer]);
    }
  }
  
  if(_doMomentum){
    hiddenWeightsMomentum.resize(_net->_hiddenWeights.size());
    hiddenBiasesMomentum.resize(_net->_hiddenBiases.size());
    
    for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
      hiddenWeightsMomentum[iWgtCount].resize(_net->_hiddenWeights[iWgtCount].size(),0.0);
      hiddenBiasesMomentum[iWgtCount].resize(_net->_hiddenBiases[iWgtCount].size(),0.0);
    }
    
    outputWeightsMomentum.resize(_net->_outputWeights.size(), 0.0);
    outputBiasesMomentum.resize(_net->_outputBiases.size(), 0.0);
    
  }
  
  feedForwardTrainData(true,0.0);
  
  outData = &_net->_generatedLabels;
  initialCost = _net->getCost();
  _epochTrainCostUpdates.push_back(initialCost);
  std::cout << "Initial Cost: " << initialCost <<  std::endl;
  if(_doTestCost){
    _testdataFeedForwardValues.resize(1);
    _net->flowDataThroughNetwork(_testdataFeedForwardValues, _testdataGeneratedValues);
    double testCost = _net->calcCost(_testdata->labels(),_testdataGeneratedValues,_testdata->nRecords(), _testdata->nLabelFields());
    _epochTestCostUpdates.push_back(testCost);
    std::cout << "Initial Test Cost: " << testCost <<  std::endl;
  }
  
  size_t nIterations =  _net->_nDataRecords/nBatchSizeTarget;
  if (nIterations*nBatchSizeTarget < _net->_nDataRecords){
    nIterations++;
  }
  for(int iEpoch = 0; iEpoch < nEpoch; iEpoch++){
    if(_shuffleEachEpoch){
      shuffleTrainData();
    }
    if(_doDropout){
      for(int iLayer = 0; iLayer < _net->_hiddenLayerSizes.size();  ++iLayer) {
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
      
      iDataStart = iIteration * nBatchSizeTarget;
      iDataStop = (iIteration + 1) * nBatchSizeTarget;
      iDataStop = ((iDataStop <= _net->_nDataRecords) ? iDataStop : _net->_nDataRecords);
      // BatchSize is used for average gradient
      // see http://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
      size_t  nBatchSize = iDataStop-iDataStart;
      
      //std::cout << "+";
      inData = &_net->_feedForwardValues[_net->_feedForwardValues.size()-1];
      
      if(_doDropout){
        feedForwardTrainData(true, 0.0, dropoutMask);
        iDropoutLayerIndex = _net->_hiddenLayerSizes.size()-1;
      }else{
        feedForwardTrainData(true, 0.0);
      }
      if(iEpoch == 0 & iDataStart == 0){
        initialCost = _net->getCost();
        std::cout << "Initial Cost Check: " << initialCost <<  std::endl;
      }
      
      // Zeroing the update matricies
      std::fill(outputWeightsUpdate.begin(),outputWeightsUpdate.end(),0.0);
      std::fill(outputBiasesUpdate.begin(),outputBiasesUpdate.end(),0.0);
      
      for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
        std::fill(hiddenWeightsUpdate[iWgtCount].begin(), hiddenWeightsUpdate[iWgtCount].end(), 0.0);
        std::fill(hiddenBiasesUpdate[iWgtCount].begin(), hiddenBiasesUpdate[iWgtCount].end(), 0.0);
      }
      // Starting with the output layer what is the output size and input size
      if(_net->_hiddenLayerSizes.size() > 0){
        nInputs = _net->_hiddenLayerSizes[_net->_hiddenLayerSizes.size()-1];
      }else{
        nInputs = _net->_nInputUnits;
      }
      nOutputs = _net->_nOutputUnits;
      
      for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
        size_t iDataInBatch = 0;
        if(_trainDataShuffled){
          iDataInBatch = _dataShuffleIndex[iDataIndex];
        }else{
          iDataInBatch = iDataIndex;
        }
        switch (_net->_outputType){
          case  nnet::LIN_OUT_TYPE:
          case  nnet::SMAX_OUT_TYPE:
            if(_doDropout){
              // Softmax with cross entropy has a simple derviative form for the weights on the input to the output units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                if(dropoutMask[iDropoutLayerIndex][iInput] == 0){
                  for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                    outputWeightsUpdate[(iOutput *nInputs) +  iInput] += (1.0/nBatchSize)*(*inData)[(iInput*_net->_nDataRecords) + iDataInBatch] * (_net->_generatedLabels[(iOutput*_net->_nDataRecords) + iDataInBatch]-_net->_dataLabels[(iOutput*_net->_nDataRecords) + iDataInBatch]);
                  }
                }
              }
            }else{
              // Softmax with cross entropy has a simple derviative form for the weights on the input to the output units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                  outputWeightsUpdate[(iOutput *nInputs) +  iInput] += (1.0/nBatchSize)*(*inData)[(iInput*_net->_nDataRecords) + iDataInBatch] * (_net->_generatedLabels[(iOutput*_net->_nDataRecords) + iDataInBatch]-_net->_dataLabels[(iOutput*_net->_nDataRecords) + iDataInBatch]);
                  
                }
              }
            }
            // Bias units have a 1 for the input weights of the unit, otherwise same as above
            for(int iBias = 0; iBias < nOutputs; ++iBias){
              outputBiasesUpdate[iBias] += (1.0/nBatchSize)*(_net->_generatedLabels[(iBias *_net->_nDataRecords) + iDataInBatch] - _net->_dataLabels[(iBias *_net->_nDataRecords)  + iDataInBatch]);
            }
            break;
        }
      }
      forwardWeightsUpdate = &outputWeightsUpdate;
      nOutputOutputs = nOutputs;
      nOutputs = nInputs;
      
      // We are travelling backwards through the Weight matricies
      for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
        // As we go backwards find the correct weights to update
        size_t iWgtMat = _net->_hiddenWeights.size() - 1 - iWgtCount;
        if(iWgtMat == 0){
          nInputs = _net->_nInputUnits;
        }else{
          nInputs = _net->_hiddenLayerSizes[iWgtMat-1];
        }
        
        inData = &_net->_feedForwardValues[iWgtMat];
        
        for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
          size_t iDataInBatch;
          if(_trainDataShuffled){
            iDataInBatch = _dataShuffleIndex[iDataIndex];
          }else{
            iDataInBatch = iDataIndex;
          }
          switch (_net->_outputType){
            case  nnet::LIN_OUT_TYPE:
            case  nnet::SMAX_OUT_TYPE:
              if(_doDropout){
                iDropoutLayerIndex--;
                // We need the derivative of the weight multiplication on the input; by the activation dervi;
                // all chained with the derivatives of the weights on the output side of the units
                for(int iInput = 0; iInput < nInputs; ++iInput){
                  if(dropoutMask[iDropoutLayerIndex][iInput] == 0){
                    for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                      if(dropoutMask[iDropoutLayerIndex+1][iOutput] == 0){
                        for(int iNext = 0; iNext < nOutputOutputs; iNext++){
                          hiddenWeightsUpdate[iWgtMat][(iOutput*nInputs) + iInput]  += (1.0/nBatchSize)*(*inData)[(iInput * _net->_nDataRecords) +iDataInBatch]* _hiddenGradients[iWgtMat][(iOutput*_net->_nDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)[(iNext*nOutputs) + iOutput];
                        }
                      }
                    }
                  }
                }
                // Biases have 1 for the weight multiplicaiton on input, otherwise the same
                for(int iInput = 0; iInput < 1; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                    if(dropoutMask[iDropoutLayerIndex+1][iOutput] == 0){
                      for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                        hiddenBiasesUpdate[iWgtMat][iOutput] += (1.0/nBatchSize)*_hiddenGradients[iWgtMat][(iOutput* _net->_nDataRecords) +iDataInBatch]*(*forwardWeightsUpdate)  [(iNext*nOutputs) + iOutput];
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
                      hiddenWeightsUpdate[iWgtMat][(iOutput*nInputs) + iInput]  += (1.0/nBatchSize)*(*inData)[(iInput * _net->_nDataRecords)+iDataInBatch]* _hiddenGradients[iWgtMat][(iOutput*_net->_nDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)[iNext*nOutputs + iOutput];
                    }
                  }
                }
                
                // Biases have 1 for the weight multiplicaiton on input, otherwise the same
                for(int iInput = 0; iInput < 1; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                    for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                      hiddenBiasesUpdate[iWgtMat][iOutput] += (1.0/nBatchSize)*_hiddenGradients[iWgtMat][(iOutput * _net->_nDataRecords) + iDataInBatch]*(*forwardWeightsUpdate)  [(iNext *nOutputs) + iOutput];
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
      for(int iWgt = 0; iWgt < _net->_outputWeights.size(); iWgt++){
        if(_doMomentum){
          outputWeightsMomentum[iWgt] = momentum_mu * outputWeightsMomentum[iWgt] - wgtLearnRate * outputWeightsUpdate[iWgt];
          _net->_outputWeights[iWgt] += outputWeightsMomentum[iWgt];
        }else{
          _net->_outputWeights[iWgt] -= wgtLearnRate*outputWeightsUpdate[iWgt];
        }
      }
      for(int iBias = 0; iBias < _net->_outputBiases.size(); iBias++){
        if(_doMomentum){
          outputBiasesMomentum[iBias] = momentum_mu * outputBiasesMomentum[iBias] - biasLearnRate * outputBiasesUpdate[iBias];
          _net->_outputBiases[iBias] += outputWeightsMomentum[iBias];
        }else{
          _net->_outputBiases[iBias] -= biasLearnRate*outputBiasesUpdate[iBias];
        }
      }
      
      // Back through the hidden layers updating the weights
      for(int iWgtMat = 0; iWgtMat < _net->_hiddenWeights.size(); iWgtMat++){
        for(int iWgt = 0; iWgt < _net->_hiddenWeights[iWgtMat].size(); iWgt++){
          if(_doMomentum){
            hiddenWeightsMomentum[iWgtMat][iWgt] = momentum_mu * hiddenWeightsMomentum[iWgtMat][iWgt] - wgtLearnRate * hiddenWeightsUpdate[iWgtMat][iWgt];
            _net->_hiddenWeights[iWgtMat][iWgt] += hiddenWeightsMomentum[iWgtMat][iWgt] ;
          }else{
            _net->_hiddenWeights[iWgtMat][iWgt] -= wgtLearnRate*hiddenWeightsUpdate[iWgtMat][iWgt];
            
          }
        }
        for(int iWgt = 0; iWgt < _net->_hiddenBiases[iWgtMat].size(); iWgt++){
          if(_doMomentum){
            hiddenBiasesMomentum[iWgtMat][iWgt] = momentum_mu * hiddenBiasesMomentum[iWgtMat][iWgt] - biasLearnRate * hiddenBiasesUpdate[iWgtMat][iWgt];
            _net->_hiddenBiases[iWgtMat][iWgt] += hiddenBiasesMomentum[iWgtMat][iWgt] ;
          }else{
            _net->_hiddenBiases[iWgtMat][iWgt] -= biasLearnRate*hiddenBiasesUpdate[iWgtMat][iWgt];
          }
        }
      }
    }
    
    //feedForwardTrainData();
    cost = _net->getCost();
    if(std::isnan(cost)){
      std::cout << "Nan cost so quitting!\n";
      allOk = false;
      break;
    }else{
      _epochTrainCostUpdates.push_back(cost);
      if(iEpoch % _epochPrintSchedule == 0){
        switch (_net->_outputType){
          case  nnet::LIN_OUT_TYPE:
            std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << std::endl;
            break;
          case  nnet::SMAX_OUT_TYPE:
            std::cout << std::endl << "Epoch " << iEpoch << "-- Cost " << cost << "-- Accuracy " << _net->getAccuracy() << "  ("  << _net->getAccuracy() * _net->_nDataRecords << ")" << std::endl;
            break;
        }
        
        if(_doTestCost){
          _testdataFeedForwardValues.resize(1);
          _net->flowDataThroughNetwork(_testdataFeedForwardValues, _testdataGeneratedValues);
          double testCost = _net->calcCost(_testdata->labels(),_testdataGeneratedValues,_testdata->nRecords(), _testdata->nLabelFields());
          _epochTestCostUpdates.push_back(testCost);
          std::cout << "Test Cost: " << testCost <<  std::endl;
        }
      }
    }
  }
  
  if(allOk){
    feedForwardTrainData(false,0.0);
    cost = _net->getCost();
    std::cout << std::fixed;
    
    std::cout << std::setprecision(2) <<  "Cost went from " << initialCost << " to " <<  cost << std::endl;
    if(_net->_outputType == nnet::SMAX_OUT_TYPE){
      std::cout << std::setprecision(2) <<  "Final accuracy is " << 100* _net->getAccuracy() << "%"<< std::endl;
    }
    if(_doTestCost){
      _testdataFeedForwardValues.resize(1);
      _net->flowDataThroughNetwork(_testdataFeedForwardValues, _testdataGeneratedValues);
      double testCost = _net->calcCost(_testdata->labels(),_testdataGeneratedValues,_testdata->nRecords(), _testdata->nLabelFields());
      _epochTestCostUpdates.push_back(testCost);
      std::cout << "Test Cost: " << testCost <<  std::endl;
      if(_net->_outputType == nnet::SMAX_OUT_TYPE){
          std::cout << "Test Accuracy: " << 100*_net->calcAccuracy(_testdata->labels(),_testdataGeneratedValues,_testdata->nRecords(), _testdata->nLabelFields()) << "%" << std::endl;
      }
    }
  }
  return allOk;
}


void backpropper::writeEpochTrainCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTrainCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTrainCostUpdates, _epochTrainCostUpdates.size(),1);
  return;
}

void backpropper::writeEpochTestCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTestCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTestCostUpdates, _epochTestCostUpdates.size(),1);
  return;
}

void backpropper::printGradients(int iWeightsIndex){
  size_t nRows = 0;
  size_t nCols = 0;
  if(iWeightsIndex <= _hiddenGradients.size()){
    std::cout << "Gradients " << iWeightsIndex << " ";
    if(iWeightsIndex == 0){
      nRows = _net->_nInputUnits;
      nCols = _net->_hiddenLayerSizes[0];
    }else{
      if(iWeightsIndex == _hiddenGradients.size()-1){
        nRows = _net->_nOutputUnits;
        nCols = _net->_hiddenLayerSizes[_net->_hiddenLayerSizes.size()-1];
      }else{
        nRows = _net->_hiddenLayerSizes[iWeightsIndex-1];
        nCols = _net->_hiddenLayerSizes[iWeightsIndex];
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
      std::cout << "Error printing weights " << nRows << " by " << nCols << " as it is actually " << _net->_hiddenWeights[iWeightsIndex].size() << std::endl;
    }
    if(iWeightsIndex < _net->_hiddenBiases.size()){
      std::cout << "Bias Vector: " << std::endl;
      for(std::vector<double>::const_iterator it = _net->_hiddenBiases[iWeightsIndex].begin(); it !=  _net->_hiddenBiases[iWeightsIndex].end(); ++it){
        if(it == _net->_hiddenBiases[iWeightsIndex].begin()){
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

void backpropper::printTrainData(size_t nRecords){
  
  if(_net->_dataLoaded){
    if(nRecords == 0){
      nRecords = _net->_nDataRecords;
    }else{
      nRecords = std::min(nRecords, _net->_nDataRecords);
    }
    
    size_t rowsToPrint = std::min(_net->_nDataRecords,nRecords);
    std::cout << "Printing data" << std::endl;
    for(int iRow = 0; iRow < rowsToPrint; iRow++){
      std::cout << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _net->_nInputUnits; iCol++){
        std::cout << std::fixed;
        std::cout << std::setprecision(3) << _net->_feedForwardValues[0][(iCol*_net->_nDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No data loaded" <<std::endl;
  }
}

void backpropper::printTrainLabels(size_t nRecords){
  if(_net->_dataLabelsLoaded){
    if(nRecords == 0){
      nRecords = _net->_nDataRecords;
    }else{
      nRecords = std::min(nRecords, _net->_nDataRecords);
    }
    std::cout << "Printing Labels" << std::endl;
    for(int iRow = 0; iRow < nRecords; iRow++){
      std::cout << "Record " << iRow + 1 << ": ";
      for(int iCol = 0; iCol < _net->_nOutputUnits; iCol++){
        std::cout << _net->_dataLabels[(iCol*_net->_nDataRecords)+iRow] << " | ";
      }
      std::cout << std::endl;
    }
  }else{
    std::cout << "No Labels Loaded" << std::endl;
  }
}

