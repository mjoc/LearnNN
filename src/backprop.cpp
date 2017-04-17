//
//  backprop.cpp
//  BasicNN
//
//  Created by Martin on 30/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#include "backprop.hpp"
#include "mat_ops.hpp"
#include "message.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>

#define BIAS_START 0.0

Backpropper::Backpropper(Nnet &net){
  std::ostringstream message;

  _net = &net;
  _rng = new rng;
  _outputDir = "~/";
  _epochPrintSchedule = 1;
  _lossType = MSE_LOSS_TYPE;

  _shuffleEachEpoch = true;
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

  // msg::info(std::string("Backpropper\n"));
  // message << _hiddenGradients.size() << " " << std::endl;
  // msg::info(message);
  // message << "net._hiddenWeights.size() " << net._hiddenWeights.size() << std::endl;
  // msg::info(message);

  _hiddenGradients.resize(0);
  _hiddenGradients.resize(net._hiddenWeights.size());
  for(int i = 0; i < net._hiddenWeights.size(); i++){
    _hiddenGradients[i].assign(net._hiddenWeights[i].size(),0.0);
  }

  _testdataLoaded = false;

}

Backpropper::Backpropper(Nnet &net, std::string lossType){
  _net = &net;
  _rng = new rng;
  _outputDir = "~/";
  _epochPrintSchedule = 1;
  _lossType = MSE_LOSS_TYPE;

  _shuffleEachEpoch = true;
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
  _hiddenGradients.resize(net._hiddenWeights.size());
  for(int i = 0; i < net._hiddenWeights.size(); i++){
    _hiddenGradients[i].resize(net._hiddenWeights[i].size());
    std::fill(_hiddenGradients[i].begin(),_hiddenGradients[i].end(),0.0);
  }
  setLossType(lossType);

  _testdataLoaded = false;

}

Backpropper::~Backpropper(){
  delete _rng;
}

void Backpropper::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}

void Backpropper::setEpochPrintSchedule(size_t schedule){
  _epochPrintSchedule = schedule;
  return;
}

void Backpropper::setShuffleData(bool doShuffle){
  _shuffleEachEpoch = doShuffle;
}

void Backpropper::setLossType(std::string lossType){
  std::ostringstream message;
  bool valid = false;
  if(lossType == std::string("mse")){
    valid = true;
    _lossType = MSE_LOSS_TYPE;
  }
  if(lossType == "xent"){
    valid = true;
    _lossType = CROSS_ENT_TYPE;
  }
  if(!valid){
    msg::info(std::string("Invalid Loss Type, only 'mse' or 'xent', setting to 'mse'!\n"));
  }
  return;
}

std::string Backpropper::getLossType(){
  std::string returnValue;
  switch (_lossType){
  case MSE_LOSS_TYPE:
    returnValue = std::string("mse");
    break;
  case CROSS_ENT_TYPE:
    returnValue = std::string("xent");
    break;
  }
  return returnValue;
}



bool Backpropper::setTestData(Dataset& testdata){
  bool allOk = true;
  if(testdata.nRecords() == 0){
    allOk = false;
    msg::error("Test data has zero records, cannot set test data!");
  }
  if(!testdata.labelsLoaded()){
    allOk =false;
    msg::error("Test data has no labels, cannot set test data!");
  }
  if(allOk){
    _testdataLoaded = true;
    _testdata = &testdata;
    _testInputData = _testdata->data();
    _testdataFeedForwardValues.resize(0);
  }else{
    allOk = false;
  }
  return allOk;
}

void Backpropper::doTestCost(bool doTestCost){
  _doTestCost = doTestCost;
}

void Backpropper::setMomentum(bool doMomentum,
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

void Backpropper::initialiseWeights(std::string initialtypeString, double param1){
  std::ostringstream message;

  size_t nCurrentInputWidth = _net->_nInputUnits;


  // std::cout << "Initialising weights" << std::endl;
  _net->_hiddenWeights.resize(0);
  _net->_hiddenBiases.resize(0);
  initialiseType initialtype = INIT_CONST_TYPE;
  bool initialTypeValid = false;
  if(initialtypeString == "const"){
    initialTypeValid = true;
    initialtype = INIT_CONST_TYPE;
  }
  if(initialtypeString == "unif"){
    initialTypeValid = true;
    initialtype = INIT_UNIFORM_TYPE;
  }
  if(initialtypeString == "gauss"){
    initialTypeValid = true;
    initialtype = INIT_GAUSS_TYPE;
  }

  if(!initialTypeValid){
    msg::error(std::string("Invalid weight initialisation specification, only 'const' or 'unif' or 'gauss'!\n"));
  }

  switch (initialtype) {
    case INIT_CONST_TYPE:
      if(_net->_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _net->_hiddenLayerSizes.size(); i++) {
          _net->_hiddenWeights.resize(i+1);
          _net->_hiddenWeights[i].resize(nCurrentInputWidth*_net->_hiddenLayerSizes[i]);
          _net->_hiddenBiases.resize(i+1);
          _net->_hiddenBiases[i].resize(_net->_hiddenLayerSizes[i]);
          std::fill(_net->_hiddenWeights[i].begin(),_net->_hiddenWeights[i].end(),param1);
          std::fill(_net->_hiddenBiases[i].begin(),_net->_hiddenBiases[i].end(),BIAS_START);
         nCurrentInputWidth = _net->_hiddenLayerSizes[i];
        }
      }

      if(_net->_nOutputUnits > 0){
        _net->_outputWeights.resize(nCurrentInputWidth * _net->_nOutputUnits);
        _net->_outputBiases.resize(_net->_nOutputUnits);
        std::fill(_net->_outputWeights.begin(),_net->_outputWeights.end(),param1);
        std::fill(_net->_outputBiases.begin(),_net->_outputBiases.end(),BIAS_START);
      }
      break;
    case INIT_GAUSS_TYPE:
      if(_net->_hiddenLayerSizes.size() > 0){
        for(int i = 0 ; i < _net->_hiddenLayerSizes.size(); i++) {
          _net->_hiddenWeights.resize(i+1);
          _net->_hiddenWeights[i].resize(nCurrentInputWidth*_net->_hiddenLayerSizes[i]);
          _rng->getGaussianVector(_net->_hiddenWeights[i], param1);
          _net->_hiddenBiases.resize(i+1);
          _net->_hiddenBiases[i].resize(_net->_hiddenLayerSizes[i]);
          std::fill(_net->_hiddenBiases[i].begin(),_net->_hiddenBiases[i].end(),BIAS_START);
          nCurrentInputWidth = _net->_hiddenLayerSizes[i];
        }
      }

      if(_net->_nOutputUnits > 0){
        _net->_outputWeights.resize(nCurrentInputWidth * _net->_nOutputUnits);
        _net->_outputBiases.resize(_net->_nOutputUnits);
        std::fill(_net->_outputBiases.begin(),_net->_outputBiases.end(),BIAS_START);
        _rng->getGaussianVector(_net->_outputWeights, param1);
      }
      break;
    default:
      message << "Don't know this type for weight initialisation!\n";
      msg::error(message);
      break;
  }
}

// We are not physically shuffling the data, but creating an index of ordering
// for use in batch optimisation
void Backpropper::shuffleTrainData(){
  std::ostringstream message;
  std::vector<std::vector<int> > indataShuffle;
  std::vector<int> iClassShuffle;
  std::vector<double>* dataLabels = _net->_dataLabels;
  size_t nDataRecords = _net->_nDataRecords;

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
          if((*dataLabels)[iLabelClass * nDataRecords +  iLabel ] > 0.8){
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

      size_t maxNumberInSingleClass = indataShuffle[0].size();
      for(int iLabelClass = 0; iLabelClass < _net->_nOutputUnits; iLabelClass++){
        maxNumberInSingleClass = indataShuffle[iLabelClass].size() > maxNumberInSingleClass ? indataShuffle[iLabelClass].size() : maxNumberInSingleClass;
      }

      _dataShuffleIndex.resize(0);
      for(int iLabel = 0; iLabel < maxNumberInSingleClass; iLabel++ ){
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
    message << "Shuffling but no train data loaded!\n";
    msg::error(message);
  }
  return;
}

void Backpropper::feedForwardTrainData(bool calcGradients,
                                double nestorovAdj){
  std::ostringstream message;
  std::vector<double> tempMatrixForLabels;

  if(_net->_dataLoaded){
    _net->_labelsGenerated = false;

    flowDataThroughNetwork((*_net->_inputData),
                           _net->_feedForwardValues,
                           tempMatrixForLabels,
                           calcGradients,
                           nestorovAdj);

    _net->_generatedLabels = tempMatrixForLabels;

    _net->_labelsGenerated = true;
  }else{
    if (!_net->_dataLoaded){
      message << "No data loaded\n";
      msg::error(message);
    }
  }
  //printOutValues();
  return;
}


void Backpropper::activateUnitsAndCalcGradients(std::vector<double>& values,
                                         std::vector<double>& gradients,
                                         double nestorovNudge){
  std::ostringstream message;
  if(values.size() != gradients.size()){
    message << "Problem with mismatch value <-> gradient sizes\n";
    msg::error(message);
  }
  switch (_net->_activationType){
    case Nnet::TANH_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {

        values[i] =  tanh(values[i]);
        gradients[i] =  (1-pow(values[i]+nestorovNudge,2)) ;
      }
      break;
    case Nnet::LIN_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        gradients[i] =  values[i] + nestorovNudge;
      }
      break;
    case Nnet::RELU_ACT_TYPE:
      for (int i= 0; i < values.size(); i++) {
        values[i] = fmax(0.0,values[i]);
        gradients[i] = (values[i]+nestorovNudge) > 0.0 ? 1 : 0.0;
      }
      break;
  }
  return;
};

void Backpropper::flowDataThroughNetwork(std::vector<double>& inputData,
                                         std::vector<std::vector<double> >& dataflowStages,
                                         std::vector<double>& dataflowMatrix,
                                         bool calcTrainGradients,
                                         double nesterovAdj){
  size_t nInputRows, nInputCols, nWeightsCols;
  bool allOk = true, mutliplyOK = true;

  nInputCols = _net->_nInputUnits;
  nInputRows = inputData.size()/nInputCols;

  if( dataflowStages.size() > 0){
    dataflowStages.resize(0);
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
    if(iLayer > 0){
      mutliplyOK = mat_ops::matMul(nInputRows,
                                   nInputCols,
                                   dataflowStages[iLayer],
                                   nWeightsCols,
                                   _net->_hiddenWeights[iLayer],
                                   dataflowMatrix);
    }else{
      mutliplyOK = mat_ops::matMul(nInputRows,
                                   nInputCols,
                                   inputData,
                                   nWeightsCols,
                                   _net->_hiddenWeights[iLayer],
                                   dataflowMatrix);
    }
    allOk = allOk and mutliplyOK;
    dataflowStages.resize(iLayer+1);
    dataflowStages[iLayer] = dataflowMatrix;

    if(calcTrainGradients){
      _hiddenGradients.resize(iLayer+1);
      _hiddenGradients[iLayer].assign(dataflowMatrix.size(),0.0);
      activateUnitsAndCalcGradients(dataflowStages[iLayer],
                                    _hiddenGradients[iLayer],
                                    nesterovAdj);
    }else{
      _net->activateUnits(dataflowStages[iLayer]);
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
  if(_net->_hiddenLayerSizes.size()>0){
    mutliplyOK = mat_ops::matMul(nInputRows,
                                 nInputCols ,
                                 dataflowStages[dataflowStages.size()-1] ,
                                 nWeightsCols,
                                 _net->_outputWeights,
                                 dataflowMatrix);
  }else{
    mutliplyOK = mat_ops::matMul(nInputRows,
                                 nInputCols,
                                 inputData,
                                 nWeightsCols,
                                 _net->_outputWeights,
                                 dataflowMatrix);


  }

  allOk = allOk and mutliplyOK;

  _net->activateOutput(dataflowMatrix);

  return;
}

double Backpropper::calcCost(bool doTestdata){
  std::vector<double> *actual = NULL;
  std::vector<double> *fitted = NULL;
  size_t nRecords = 0;
  bool allOk = true;

  if(doTestdata){
    if(_testdataLoaded){
      actual = _testdata->labels();
      fitted = &_testdataGeneratedValues;
      nRecords = _testdata->nRecords();
    }else{
      msg::error(std::string("There is no testdata loaded, cannot calculate test cost\n"));
      allOk = false;
    }
  }else{
    if(_net->dataAndLabelsLoaded()){
      actual = _net->_dataLabels;
      fitted = &_net->_generatedLabels;
      nRecords = _net->_nDataRecords;
    }else{
      msg::error(std::string("There is no data loaded, cannot calculate train cost\n"));
      allOk = false;
    }
  }
  double cost = 0.0;
  if(allOk){
    switch (_lossType){
      case Backpropper::MSE_LOSS_TYPE:
        cost = 0.0;
        for(int iRecord = 0; iRecord < nRecords; ++iRecord){
          for(int iUnit = 0; iUnit < _net->_nOutputUnits; ++iUnit){
            cost += pow((*fitted)[(iUnit*nRecords)+iRecord] - (*actual)[(iUnit*nRecords)+iRecord],2.0)/nRecords;
          }
        }

        break;
      case  Backpropper::CROSS_ENT_TYPE:
        cost = 0.0;
        for(int iRecord = 0; iRecord < nRecords; ++iRecord){
          for(int iUnit = 0; iUnit < _net->_nOutputUnits; ++iUnit){
            if((*actual)[(iUnit*nRecords)+iRecord] > 0.5){
              cost -= log( (*fitted)[(iUnit*nRecords)+iRecord]);
            }
          }
        }
    }
  }else{
    cost = -1.0;
  }
  return cost;
}

double Backpropper::calcAccuracy(bool doTestdata){
  size_t nRecords = 0;
  std::vector<double>* actual = NULL;
  std::vector<double>* fitted = NULL;
  double accuracy = 0.0;
  double maxProb = 0.0;
  bool correct = false;
  double allOk = true;


  if(doTestdata){
    if(_testdataLoaded){
      actual = _testdata->labels();
      fitted = &_testdataGeneratedValues;
      nRecords = _testdata->nRecords();
    }else{
      msg::error(std::string("There is no testdata load, cannot calculate test cost\n"));
      allOk = false;
    }

  }else{
    if(_net->dataAndLabelsLoaded()){
      actual = _net->_dataLabels;
      fitted = &_net->_generatedLabels;
      nRecords = _net->_nDataRecords;
    }else{
      msg::error(std::string("There is no testdata load, cannot calculate train cost\n"));
      allOk = false;
    }
  }

  if(allOk){
    for(int iRecord = 0; iRecord < nRecords; ++iRecord){
      for(int iUnit = 0; iUnit < _net->_nOutputUnits; ++iUnit){
        if(iUnit == 0){
          maxProb = (*fitted)[(iUnit*nRecords)+iRecord];
          if((*actual)[(iUnit*nRecords)+iRecord] > 0.5){
            correct = true;
          }else{
            correct = false;
          }
        }else{
          if((*fitted)[(iUnit*nRecords)+iRecord] > maxProb){
            maxProb = (*fitted)[(iUnit*nRecords)+iRecord];
            if((*actual)[(iUnit*nRecords)+iRecord] > 0.5){
              correct = true;
            }else{
              correct = false;
            }
          }
        }
      }
      if(correct){accuracy += 1.0;}
    }
    accuracy /= nRecords;
  }else{
    accuracy = -1.0;
  }
  return accuracy;
}

void Backpropper::doBackPropOptimise(size_t nBatchSizeRequested,
                    double wgtLearnRate,
                    double biasLearnRate,
                    size_t nEpoch){
  bool allOk = true;
  size_t nInputs, nOutputs, iDataStart, iDataStop;
  std::ostringstream message;
  size_t nBatchSizeTarget;
  double initialCost = 0.0, cost = 0.0;
  double momentum_mu = _momMu;

  std::ostringstream oss;

  _epochTrainCostUpdates.resize(0);
  _epochTestCostUpdates.resize(0);

  if (nBatchSizeRequested < 1){
    nBatchSizeTarget = _net->_nDataRecords;
  }else{
    nBatchSizeTarget = nBatchSizeRequested;
  }

  if(_doTestCost){
    if (!_testdataLoaded){
      _doTestCost = false;
      msg::error(std::string("There is no Non-Train data load, cannot calculate test error\n"));
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

  //const std::vector<double>* inputData = _net->_inputData;
  const std::vector<double>* inputDataLabels = _net->_dataLabels;
  const std::vector<double>* generatedDataLabels = &_net->_generatedLabels;
  const size_t nInputData = _net->_nDataRecords;

  std::vector<double> *currentInputData, *forwardWeightsUpdate; //, *biases, *gradients;

  if((!_net->dataAndLabelsLoaded()) || (nInputData==0) ){
    msg::error("No labelled data, giving up!\n");
    allOk = false;
  }

  if(allOk){
    hiddenWeightsUpdate.resize(_net->_hiddenWeights.size());
    hiddenBiasesUpdate.resize(_net->_hiddenBiases.size());

    for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
      hiddenWeightsUpdate[iWgtCount].resize(_net->_hiddenWeights[iWgtCount].size(),0.0);
      hiddenBiasesUpdate[iWgtCount].resize(_net->_hiddenBiases[iWgtCount].size(),0.0);

      std::fill(hiddenWeightsUpdate[iWgtCount].begin(),hiddenWeightsUpdate[iWgtCount].end(),0.0);
      std::fill(hiddenBiasesUpdate[iWgtCount].begin(),hiddenBiasesUpdate[iWgtCount].end(),0.0);
    }

    if(_doMomentum){
      hiddenWeightsMomentum.resize(_net->_hiddenWeights.size());
      hiddenBiasesMomentum.resize(_net->_hiddenBiases.size());

      for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.  size() ; iWgtCount++){
        hiddenWeightsMomentum[iWgtCount].resize(_net->_hiddenWeights[iWgtCount].size());
        hiddenBiasesMomentum[iWgtCount].resize(_net->_hiddenBiases[iWgtCount].size());
        std::fill(hiddenWeightsMomentum[iWgtCount].begin(),hiddenWeightsMomentum[iWgtCount].end(),0.0);
        std::fill(hiddenBiasesMomentum[iWgtCount].begin(),hiddenBiasesMomentum[iWgtCount].end(),0.0);
      }

      outputWeightsMomentum.resize(_net->_outputWeights.size());
      outputBiasesMomentum.resize(_net->_outputBiases.size());
      std::fill(outputWeightsMomentum.begin(),outputWeightsMomentum.end(),0.0);
      std::fill(outputBiasesMomentum.begin(),outputBiasesMomentum.end(),0.0);

    }

    feedForwardTrainData(true,0.0);


    initialCost = calcCost();
    _epochTrainCostUpdates.push_back(initialCost);
    message << "Initial Cost: " << initialCost <<  std::endl;
    msg::info(message);
    if(_doTestCost){
      _testdataFeedForwardValues.resize(0);
      _net->flowDataThroughNetwork(_testInputData, _testdataFeedForwardValues, _testdataGeneratedValues);
      double testCost = calcCost();
      _epochTestCostUpdates.push_back(testCost);
      message << "Initial Test Cost: " << testCost <<  std::endl;
      msg::info(message);
    }

    size_t nIterations =  nInputData/nBatchSizeTarget;
    if (nIterations*nBatchSizeTarget < nInputData){
      nIterations++;
    }
    for(int iEpoch = 0; iEpoch < nEpoch; iEpoch++){
      if(_shuffleEachEpoch){
        shuffleTrainData();
      }
      if(_doMomentum && iEpoch > 0){
        if(_momDecaySchedule > 0){
          if(iEpoch% _momDecaySchedule == 0){
            momentum_mu += _momDecay * (_momFinal - momentum_mu);
            message << "Momentum decay parameter" << momentum_mu << std::endl;
            msg::info(message);
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

        // Starting with the output layer what is the output size and input size
        if(_net->_hiddenLayerSizes.size() > 0){
          currentInputData = &_net->_feedForwardValues[_net->_feedForwardValues.size()-1];
          nInputs = _net->_hiddenLayerSizes[_net->_hiddenLayerSizes.size()-1];
        }else{
          nInputs = _net->_nInputUnits;
          currentInputData = _net->_inputData;
        }
        nOutputs = _net->_nOutputUnits;

        feedForwardTrainData(true, 0.0);
        if(iEpoch == 0 & iDataStart == 0){
          initialCost = calcCost();
          message << "Initial Cost Check: " << initialCost <<  std::endl;
          msg::info(message);
        }

        // Zeroing the update matricies
        std::fill(outputWeightsUpdate.begin(),outputWeightsUpdate.end(),0.0);
        std::fill(outputBiasesUpdate.begin(),outputBiasesUpdate.end(),0.0);

        for(size_t iWgtCount= 0; iWgtCount < _net->_hiddenWeights.size() ; iWgtCount++){
          std::fill(hiddenWeightsUpdate[iWgtCount].begin(), hiddenWeightsUpdate[iWgtCount].end(), 0.0);
          std::fill(hiddenBiasesUpdate[iWgtCount].begin(), hiddenBiasesUpdate[iWgtCount].end(), 0.0);
        }




        for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
          size_t iDataInBatch = 0;
          if(_trainDataShuffled){
            iDataInBatch = _dataShuffleIndex[iDataIndex];
          }else{
            iDataInBatch = iDataIndex;
          }
          switch (_net->_outputType){
            case  Nnet::LIN_OUT_TYPE:
            case  Nnet::SMAX_OUT_TYPE:
              // Softmax with cross entropy has a simple derviative form for the weights on the input to the output units
              for(int iInput = 0; iInput < nInputs; ++iInput){
                for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                  outputWeightsUpdate[(iOutput *nInputs) +  iInput] += (1.0/nBatchSize)*(*currentInputData)[(iInput*nInputData) + iDataInBatch] * ((*generatedDataLabels)[(iOutput*nInputData) + iDataInBatch]-(*inputDataLabels)[(iOutput*nInputData) + iDataInBatch]);

                }
              }

              // Bias units have a 1 for the input weights of the unit, otherwise same as above
              for(int iBias = 0; iBias < nOutputs; ++iBias){
                outputBiasesUpdate[iBias] += (1.0/nBatchSize)*((*generatedDataLabels)[(iBias *nInputData) + iDataInBatch] - (*inputDataLabels)[(iBias *nInputData)  + iDataInBatch]);
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
            currentInputData = _net->_inputData;
          }else{
            nInputs = _net->_hiddenLayerSizes[iWgtMat-1];
            currentInputData = &_net->_feedForwardValues[iWgtMat-1];
          }

          for(size_t iDataIndex = iDataStart; iDataIndex < iDataStop; iDataIndex++){
            size_t iDataInBatch;
            if(_trainDataShuffled){
              iDataInBatch = _dataShuffleIndex[iDataIndex];
            }else{
              iDataInBatch = iDataIndex;
            }
            switch (_net->_outputType){
              case  Nnet::LIN_OUT_TYPE:
              case  Nnet::SMAX_OUT_TYPE:
                // We need the derivative of the weight multiplication on the input; by the activation dervi;
                // all chained with the derivatives of the weights on the output side of the units
                for(int iInput = 0; iInput < nInputs; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; ++iOutput){
                    for(int iNext = 0; iNext < nOutputOutputs; iNext++){
                      hiddenWeightsUpdate[iWgtMat][(iOutput*nInputs) + iInput]  += (1.0/nBatchSize)*(*currentInputData)[(iInput * nInputData)+iDataInBatch]* _hiddenGradients[iWgtMat][(iOutput*nInputData) + iDataInBatch]*(*forwardWeightsUpdate)[iNext*nOutputs + iOutput];
                    }
                  }
                }

                // Biases have 1 for the weight multiplicaiton on input, otherwise the same
                for(int iInput = 0; iInput < 1; ++iInput){
                  for(int iOutput = 0; iOutput < nOutputs; iOutput++){
                    for(int iNext = 0; iNext< nOutputOutputs; iNext++){
                      hiddenBiasesUpdate[iWgtMat][iOutput] += (1.0/nBatchSize)*_hiddenGradients[iWgtMat][(iOutput * nInputData) + iDataInBatch]*(*forwardWeightsUpdate)  [(iNext *nOutputs) + iOutput];
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
      cost = calcCost();
      if(std::isnan(cost)){
        message << "Nan cost so quitting!\n";
        msg::warn(message);
        allOk = false;
        break;
      }else{
        _epochTrainCostUpdates.push_back(cost);
        if(iEpoch % _epochPrintSchedule == 0){
          switch (_net->_outputType){
            case  Nnet::LIN_OUT_TYPE:
              message << std::endl << "Epoch " << iEpoch << "--Cost " << cost;
              msg::info(message);
              break;
            case  Nnet::SMAX_OUT_TYPE:
              double accuracy = calcAccuracy();
              message << std::endl << "Epoch " << iEpoch << "--Cost " << cost << "-- Accuracy " << accuracy << "  (" << accuracy * _net->_nDataRecords << ")" << std::endl;
              msg::info(message);
              break;
          }

          if(_doTestCost){
            _testdataFeedForwardValues.resize(0);
            _net->flowDataThroughNetwork(_testInputData,
                                         _testdataFeedForwardValues,
                                         _testdataGeneratedValues);
            double testCost = calcCost(true);
            _epochTestCostUpdates.push_back(testCost);
            message << "Test Cost: " << testCost <<  std::endl;
            msg::info(message);
          }
        }
      }
    }
  }
  if(allOk){
    feedForwardTrainData(false,0.0);
    cost = calcCost();

    message << std::fixed;
    message << std::setprecision(2) << std::endl <<  "Cost went from " << initialCost << " to " <<  cost << std::endl;
    msg::info(message);
    if(_net->_outputType == Nnet::SMAX_OUT_TYPE){
      message << std::setprecision(2) <<  "Final accuracy is " << 100* calcAccuracy() << "%"<< std::endl;
      msg::info(message);
    }
    if(_doTestCost){
      _testdataFeedForwardValues.resize(0);
      _net->flowDataThroughNetwork(_testInputData,
                                   _testdataFeedForwardValues,
                                   _testdataGeneratedValues);
      double testCost = calcCost(true);
      _epochTestCostUpdates.push_back(testCost);
      message << "Test Cost: " << testCost <<  std::endl;
      msg::info(message);
      if(_net->_outputType == Nnet::SMAX_OUT_TYPE){
        message << "Test Accuracy: " << 100*calcAccuracy() << "%" << std::endl;
        msg::info(message);
      }
    }
  }
  if(!allOk){
    msg::error("OPtimisation with Back Propagation ended in error state");
  }
  return ;
}


void Backpropper::writeEpochTrainCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTrainCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTrainCostUpdates, _epochTrainCostUpdates.size(),1);
  return;
}

void Backpropper::writeEpochTestCostUpdates(){
  std::ostringstream oss;
  oss << _outputDir << "epochTestCostUpdates.csv";
  mat_ops::writeMatrix(oss.str(), _epochTestCostUpdates, _epochTestCostUpdates.size(),1);
  return;
}

void Backpropper::printGradients(int iWeightsIndex){
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if(iWeightsIndex <= _hiddenGradients.size()){
    message << "Gradients " << iWeightsIndex << " ";
    msg::info(message);
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
      message << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        message << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          message << std::fixed;
          message << std::setprecision(2) << _hiddenGradients[iWeightsIndex][iRow*nCols+iCol] << " | ";
        }
        message << std::endl;
        msg::info(message);
      }

    }else{
      message << "Error printing weights " << nRows << " by " << nCols << " as it is actually " << _net->_hiddenWeights[iWeightsIndex].size() << std::endl;
      msg::error(message);
    }
    if(iWeightsIndex < _net->_hiddenBiases.size()){
      message << "Bias Vector: " << std::endl;

      for(std::vector<double>::const_iterator it = _net->_hiddenBiases[iWeightsIndex].begin(); it !=  _net->_hiddenBiases[iWeightsIndex].end(); ++it){
        if(it == _net->_hiddenBiases[iWeightsIndex].begin()){
          message << *it;
        }else{
          message <<  " : " << *it;
        }
      }
      message << std::endl;
      msg::info(message);
    }else{
      msg::warn(std::string("Cannot print Gradient, invalid Bias index!\n"));
    }
  }else{
    msg::warn(std::string("Cannot print Gradient, invalid Bias index!\n"));
  }
}

void Backpropper::printTrainData(size_t nRecords){
  std::ostringstream message;
  const std::vector<double>* inputData = _net->_inputData;
  if(_net->_dataLoaded){
    if(nRecords == 0){
      nRecords = _net->_nDataRecords;
    }else{
      nRecords = std::min(nRecords, _net->_nDataRecords);
    }

    size_t rowsToPrint = std::min(_net->_nDataRecords,nRecords);
    message << "Printing data" << std::endl;
    msg::info(message);
    for(int iRow = 0; iRow < rowsToPrint; iRow++){
      message << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _net->_nInputUnits; iCol++){
        message << std::fixed;
        message << std::setprecision(3) << (*inputData)[(iCol*_net->_nDataRecords)+iRow] << " | ";
      }
      message << std::endl;
      msg::info(message);
    }
  }else{
    msg::warn(std::string("No data loaded\n"));
  }
}

void Backpropper::printTrainLabels(size_t nRecords){
  std::ostringstream message;
  const std::vector<double>* dataLabels = _net->_dataLabels;
  if(_net->_dataLabelsLoaded){
    if(nRecords == 0){
      nRecords = _net->_nDataRecords;
    }else{
      nRecords = std::min(nRecords, _net->_nDataRecords);
    }
    msg::info(std::string("Printing Labels\n"));
    msg::info(message);
    for(int iRow = 0; iRow < nRecords; iRow++){
      message << "Record " << iRow + 1 << ": ";
      for(int iCol = 0; iCol < _net->_nOutputUnits; iCol++){
        message << (*dataLabels)[(iCol*_net->_nDataRecords)+iRow] << " | ";
      }
      message << std::endl;
      msg::info(message);
    }
  }else{
    msg::warn(std::string("No Labels Loaded\n"));
  }
}

#ifndef IGNORE_THIS_RCPP_CODE

SEXP Backpropper::calcCostR(){
  double cost = calcCost(false);

  if(cost < 0 ){
    return R_NilValue;
  }else{
    return Rcpp::wrap(cost);
  }
}
#endif




