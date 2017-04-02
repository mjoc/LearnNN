#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>
#include <gsl/gsl_cblas.h>
#include <vector>
#include <sstream>
#include "mat_ops.hpp"
#include "nnet.hpp"
#include "message.hpp"
#include "backprop.hpp"

#define MIN_DATA_RANGE 1e-4


// http://cs231n.github.io/neural-networks-3/

Nnet::Nnet(){
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = Nnet::LIN_OUT_TYPE;

  _activationType = Nnet::LIN_ACT_TYPE;

  _dataLabelsLoaded = false;
  _dataLoaded = false;
  _nDataRecords = 0;

  _outputDir = "~/";
};

Nnet::Nnet(std::vector<int> networkGeometry,
           std::string hiddenUnitActivation,
           std::string outputUnitActivation){
  std::ostringstream message;
  bool geometryValid = true;
  size_t nInputSize = 0, nOutputSize = 0;
  std::vector<int> hiddenLayerSizes;

  //// DEFAULTS ////
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = Nnet::LIN_OUT_TYPE;
  _activationType = Nnet::LIN_ACT_TYPE;


  _dataLoaded = false;
  _dataLabelsLoaded = false;
  _nDataRecords = 0;
  _labelsGenerated = false;

  _outputDir = "~/";
  //// END DEFAULTS ////

  bool activationTypeValid = false;
  if(hiddenUnitActivation == "linear"){
    activationTypeValid = true;
    _activationType = Nnet::LIN_ACT_TYPE;
  }
  if(hiddenUnitActivation == "tanh"){
    activationTypeValid = true;
    _activationType = Nnet::TANH_ACT_TYPE;
  }
  if(hiddenUnitActivation == "relu"){
    activationTypeValid = true;
    _activationType = Nnet::RELU_ACT_TYPE;
  }
  if(!activationTypeValid){
    msg::error(std::string("Invalid Hidden Unit Activation, only 'linear' or 'tanh' or 'relu'!\n"));
  }

  bool outputTypeValid = false;
  if(outputUnitActivation == "linear"){
    outputTypeValid = true;
    _outputType = Nnet::LIN_OUT_TYPE;
  }
  if(outputUnitActivation == "softmax"){
    outputTypeValid = true;
    _outputType = Nnet::SMAX_OUT_TYPE;
  }
  if(!outputTypeValid){
    msg::error(std::string("Invalid Output Unit Type, only 'linear' or 'softmax'!\n"));
  }

  size_t nLayers = networkGeometry.size();
  if (nLayers < 2){
    msg::error(std::string("Network Geometry Size Vector is of size 1, needs to include at least input and output sizes!\n"));
    geometryValid = false;
  }else{
    nInputSize = networkGeometry[0];
    nOutputSize = networkGeometry[networkGeometry.size()-1];

    if(nInputSize < 1){
      msg::error(std::string("Number of Input Units needs to be >0\n"));
      geometryValid = false;
    }
    if(nInputSize < 1){
      msg::error(std::string("Number of Output Units needs to be >0\n"));
      geometryValid = false;
    }
    hiddenLayerSizes.resize(0);
    for(size_t iHidden = 1; iHidden < networkGeometry.size()-1; iHidden++){
      if(networkGeometry[iHidden] < 1){
        geometryValid = false;
        msg::error(std::string("Network Geometry Size Vector contains 0!\n"));
        break;
      }else{
        hiddenLayerSizes.push_back(networkGeometry[iHidden]);
      }
    }
  }
  if(geometryValid){
    _nInputUnits = nInputSize;
    _nOutputUnits = nOutputSize;
    _hiddenLayerSizes.resize(0);
    for(std::vector<int>::const_iterator it = hiddenLayerSizes.begin(); it != hiddenLayerSizes.end(); ++it) {
      _hiddenLayerSizes.push_back(*it);
    }
    size_t nCurrentInputWidth = _nInputUnits;

    // std::cout << "Initialising weights" << std::endl;
    _hiddenWeights.resize(0);
    _hiddenBiases.resize(0);

    if(_hiddenLayerSizes.size() > 0){
      for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
        _hiddenWeights.resize(i+1);
        _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i]);
        _hiddenBiases.resize(i+1);
        _hiddenBiases[i].resize(_hiddenLayerSizes[i]);
        std::fill(_hiddenWeights[i].begin(),_hiddenWeights[i].end(),0.0);
        std::fill(_hiddenBiases[i].begin(),_hiddenBiases[i].end(),0.0);
        nCurrentInputWidth = _hiddenLayerSizes[i];
      }
    }

    if(_nOutputUnits > 0){
      _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 0);
      _outputBiases.resize(_nOutputUnits, 0);
    }

  }
  if((!geometryValid) | (!activationTypeValid) | (!outputTypeValid)){
    msg::error(std::string("There were problems with Network Specifications\n"));
    msg::error(std::string("Check the Activation Type, Output Type and Geometry\n"));
  }
}

Nnet::~Nnet(){
};

void Nnet::setInputSize(size_t nInputUnits){
  _nInputUnits = nInputUnits;
}

size_t Nnet::getInputSize(){
  return _nInputUnits;
}

void Nnet::setHiddenLayerSizes(std::vector<int> layerSizes){
  _hiddenLayerSizes.resize(0);
  for(std::vector<int>::const_iterator it = layerSizes.begin(); it != layerSizes.end(); ++it) {
    _hiddenLayerSizes.push_back(*it);
  }
  initialiseWeights();
  return;
}

std::vector<int> Nnet::getHiddenLayerSizes(){
  return _hiddenLayerSizes;
}

bool Nnet::labelsGenerated(){
  return _labelsGenerated;
}

void Nnet::setActivationType(std::string activationType){
  bool valid = false;
  if(activationType == "linear"){
    valid = true;
    _activationType = Nnet::LIN_ACT_TYPE;
  }
  if(activationType == "tanh"){
    valid = true;
    _activationType = Nnet::TANH_ACT_TYPE;
  }
  if(activationType == "relu"){
    valid = true;
    _activationType = Nnet::RELU_ACT_TYPE;
  }
  if(!valid){
    msg::error(std::string("Invalid Hidden Unit Activation, only 'linear' or 'tanh' or 'relu'!\n"));
  }
  return;
}

std::string Nnet::getActivationType(){
  std::string returnValue;
  switch (_activationType){
  case LIN_ACT_TYPE:
    returnValue = "linear";
    break;
  case TANH_ACT_TYPE:
    returnValue = "tanh";
    break;
  case RELU_ACT_TYPE:
    returnValue = "relu";
    break;
  }
  return returnValue;
}

void Nnet::setOutputType(std::string outputType){
  bool valid = false;
  if(outputType == "linear"){
    valid = true;
    _outputType = Nnet::LIN_OUT_TYPE;
  }
  if(outputType == "softmax"){
    valid = true;
    _outputType = Nnet::SMAX_OUT_TYPE;
  }
  if(!valid){
    msg::error(std::string("Invalid Output Unit Type, only 'linear' or 'softmax'!\n"));
  }
  return;
}

std::string Nnet::getOutputType(){
  std::string returnValue;
  switch (_outputType){
  case Nnet::LIN_OUT_TYPE:
    returnValue = "linear";
    break;
  case SMAX_OUT_TYPE:
    returnValue = "softmax";
    break;
  }
  return returnValue;
}

void Nnet::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}


bool Nnet::dataLoaded(){
  return _dataLoaded;
}

bool Nnet::clampData(Dataset& dataset){
  std::ostringstream message;
  bool allOk = true;
  _dataLabelsLoaded = false;
  _dataLoaded = false;
  _nDataRecords = 0;

  if(dataset.dataLoaded() && dataset.nRecords()>0){
    if(dataset.nFields() == _nInputUnits){
      if(dataset.labelsLoaded()){
        if(dataset.nLabelFields() != _nOutputUnits){
          msg::error("Number of output fields in the clamped data not consistent with the network geometry\n");
          allOk = false;
        }
      }
    }else{
      msg::error("Number of fields in the clamped data not consistent with the network geometry\n");
      allOk = false;
    }
  }else{
    msg::error("No data in clamped dataset");
    allOk = false;
  }

  if(allOk){
    _feedForwardValues.resize(0);
    _inputData = dataset.data();

    _nDataRecords = dataset.nRecords();
    _dataLoaded = true;

    if(dataset.labelsLoaded()){
      _dataLabels = dataset.labels();
      _nOutputUnits = dataset.nLabelFields();
      _dataLabelsLoaded = true;

      message << "Loaded " << _nDataRecords << " data records with labels\n";
      msg::info(message);
    }else{
      message << "Loaded " << _nDataRecords << " unlabeled data records\n";
      msg::info(message);
    }
  }
  return allOk;
}


bool Nnet::dataAndLabelsLoaded(){
  return _dataLoaded && _dataLabelsLoaded;
}

size_t Nnet::nDataRecords(){
  return _nDataRecords;
}

bool Nnet::setDataAndLabels(Dataset dataToClamp){
  bool allOk = true;

  if(!dataToClamp.dataLoaded()){
    allOk = false;
  }
  if(!dataToClamp.labelsLoaded()){
    allOk = false;
  }
  if(dataToClamp.nFields() != _nInputUnits){
    allOk = false;
  }
  if(dataToClamp.nLabelFields() != _nOutputUnits){
    allOk = false;
  }
  if(allOk){
    _feedForwardValues.resize(0);
    _inputData = dataToClamp.data();
    _nInputUnits = dataToClamp.nFields();
    _nDataRecords = dataToClamp.nRecords();
    _dataLoaded = true;

    _dataLabels = dataToClamp.labels();
    _nOutputUnits = dataToClamp.nLabelFields();
    _dataLabelsLoaded = true;

  }
  return allOk;
}

void Nnet::activateUnits(std::vector<double>& values){
  switch (_activationType){
    case Nnet::TANH_ACT_TYPE:
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


void Nnet::activateOutput(std::vector<double>& values){
  switch (_outputType){
    case Nnet::LIN_OUT_TYPE:
      // do nothing
      break;
    case SMAX_OUT_TYPE:
      // Big help from: http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
      // std::cout << "Activate Softmax final layer" << std::endl;
      double denominator = 0.0;
      double max = 0.0;

      std::vector<double> numerator(_nOutputUnits);
      for(int iRow = 0; iRow < _nDataRecords; iRow++){
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          if(iCol == 0){
            max = values[(iCol*_nDataRecords)+iRow];
          }else{
            if(values[(iCol*_nDataRecords)+iRow]>max){
              max = values[(iCol*_nDataRecords)+iRow];
            }
          }
        }
        // An adjustment to avoid overflow
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          values[(iCol*_nDataRecords)+iRow] -= max;
        }

        denominator = 0.0;
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          numerator[iCol] = exp(values[(iCol*_nDataRecords)+iRow]);
          denominator  += numerator[iCol];
        }
        for(int iCol = 0; iCol < _nOutputUnits; iCol++){
          values[(iCol*_nDataRecords)+iRow] = numerator[iCol]/denominator;
        }
      }
      break;
  }

}



void Nnet::initialiseWeights(){
  size_t nCurrentInputWidth = _nInputUnits;

  // std::cout << "Initialising weights" << std::endl;
  _hiddenWeights.resize(0);
  _hiddenBiases.resize(0);

  if(_hiddenLayerSizes.size() > 0){
    for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
      _hiddenWeights.resize(i+1);
      _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i], 0);
      _hiddenBiases.resize(i+1);
      _hiddenBiases[i].resize(_hiddenLayerSizes[i]);
      nCurrentInputWidth = _hiddenLayerSizes[i];
      std::fill(_hiddenWeights[i].begin(),_hiddenWeights[i].end(),0.0);
      std::fill(_hiddenBiases[i].begin(),_hiddenBiases[i].end(),0.0);
    }
  }

  if(_nOutputUnits > 0){
    _outputWeights.resize(nCurrentInputWidth * _nOutputUnits);
    _outputBiases.resize(_nOutputUnits);
    std::fill(_outputWeights.begin(),_outputWeights.end(),0.0);
    std::fill(_outputBiases.begin(),_outputBiases.end(),0.0);
  }
  return;
}

bool Nnet::setWgtAndBias(int iIndex, std::vector<double> weights, std::vector<double> bias){
  bool   allOk = true;
  bool doOutputWeights = false;
  std::vector<double> *wgts_ptr = NULL;
  std::vector<double> *biases_ptr = NULL;
  std::ostringstream message;

  if(iIndex < 0){
    msg::error(std::string("Invalid index, less than 0\n"));
    allOk = false;
  }
  if(allOk){
    if (iIndex > _hiddenWeights.size()){
      message << "Invalid index, greater than " << _hiddenWeights.size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }

  if(allOk){
    // Convert from R 1 index to C++ 0 index

    if(iIndex == _hiddenWeights.size()){
      doOutputWeights = true;
      wgts_ptr = &_outputWeights;
      biases_ptr = &_outputBiases;
    }else{
      wgts_ptr = &_hiddenWeights[iIndex];
      biases_ptr = &_hiddenBiases[iIndex];
    }
  }

  size_t nRows = 0;
  size_t nCols = 0;

  if(allOk){
    if(iIndex == 0){
      nRows = _nInputUnits;
    }else{
      nRows = _hiddenLayerSizes[iIndex-1];
    }
    if(doOutputWeights){
      nCols = _nOutputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex];
    }
  }
  if(allOk){
    if(nRows * nCols !=  weights.size()) {
      message << "Weights matrix expected to be " << nRows << " by " << nCols << " but found to be " << weights.size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    if(nCols != bias.size()){
      message << "Bias vector expected to be " << nCols << " but found to be " << bias.size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    for(int iRow = 0; iRow < nRows; iRow++){
      for(int iCol = 0; iCol < nCols; iCol++){
        (*wgts_ptr)[iCol*nRows+iRow] = weights[(iCol*nRows)+iRow];
      }
    }

    for(int iBias = 0; iBias < nCols; iBias++){
      (*biases_ptr)[iBias] = bias[iBias];
    }
  }
  return allOk;
}





void Nnet::flowDataThroughNetwork(std::vector<double>* inputDataMatrix,
                                  std::vector<std::vector<double> >& dataflowStages,
                                         std::vector<double>& dataflowMatrix){
  size_t nInputRows, nInputCols, nWeightsCols;


  nInputCols = _nInputUnits;
  nInputRows = inputDataMatrix->size()/nInputCols;

  if( dataflowStages.size() > 0){
    dataflowStages.resize(0);
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
    if(iLayer == 0){
      mat_ops::matMul(nInputRows, nInputCols, (*inputDataMatrix), nWeightsCols, _hiddenWeights[iLayer], dataflowMatrix);
    }else{
      mat_ops::matMul(nInputRows, nInputCols, dataflowStages[iLayer-1], nWeightsCols, _hiddenWeights[iLayer], dataflowMatrix);
    }

    dataflowStages.resize(iLayer+1);
    dataflowStages[iLayer] = dataflowMatrix;
    activateUnits(dataflowStages[iLayer]);

    //Ninputrows doesn't change
    nInputCols = nWeightsCols;
  }

  //  std::cout << "Feed forward: calculating output layer " << std::endl;
  nWeightsCols =  _nOutputUnits;
  dataflowMatrix.assign(nInputRows*_outputBiases.size(),0.0);
  for(int iRow = 0; iRow < nInputRows; ++iRow){
    for(int iCol = 0; iCol <  _outputBiases.size(); iCol++){
      dataflowMatrix[(iCol*nInputRows) + iRow] = _outputBiases[iCol];
    }
  }
  if(_hiddenLayerSizes.size()>0){
    mat_ops::matMul(nInputRows,
                    nInputCols ,
                    dataflowStages[dataflowStages.size()-1] ,
                    nWeightsCols,
                    _outputWeights,
                    dataflowMatrix);
  }else{
    mat_ops::matMul(nInputRows,
                    nInputCols ,
                    (*inputDataMatrix) ,
                    nWeightsCols,
                    _outputWeights,
                    dataflowMatrix);

  }

  activateOutput(dataflowMatrix);

  return;
}

void Nnet::feedForward(){
  std::vector<double> tempMatrixForLabels;
  bool allOk = true;

  if(!_dataLoaded){
    msg::error(std::string("No data loaded!\n"));
    allOk = false;
  }
  if(_nDataRecords == 0){
    msg::error(std::string("No data loaded!\n"));
    allOk = false;
  }
  if(allOk){
    _labelsGenerated = false;

    flowDataThroughNetwork(_inputData, _feedForwardValues, tempMatrixForLabels);

    _generatedLabels = tempMatrixForLabels;
    _labelsGenerated = true;
  }else{
    if (!_dataLoaded){
      msg::error(std::string("No data loaded\n"));
    }
  }
  //printOutValues();
  return;
}

void Nnet::writeWeights(){
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

void Nnet::writeFeedForwardValues(){
  size_t nCols = 0;

  for(int i = 0; i < _feedForwardValues.size(); i++){
    nCols = _hiddenLayerSizes[i];
    std::ostringstream oss;
    oss << _outputDir << "feedforward" << i << ".csv";
    mat_ops::writeMatrix(oss.str(), _feedForwardValues[i],_nDataRecords,nCols);

  }
  return;
}

void Nnet::writeOutputUnitValues(){
  std::ostringstream oss;
  if(_labelsGenerated){
    oss << _outputDir << "outvalues.csv";
    mat_ops::writeMatrix(oss.str(), _generatedLabels,_nDataRecords,_nOutputUnits);
  }else{
    msg::warn(std::string("No output data created\n"));
  }
  return;
}


void Nnet::printUnitType(){
  switch (_activationType){
    case Nnet::TANH_ACT_TYPE:
      msg::info(std::string("Unit: TANH\n"));
      break;
    case LIN_ACT_TYPE:
      msg::info(std::string("Unit: LINEAR\n"));
      break;
    case RELU_ACT_TYPE:
      msg::info(std::string("Unit: RELU\n"));
      break;
 	}

}

void Nnet::printOutputType(){
  switch (_outputType){
    case Nnet::LIN_OUT_TYPE:
      msg::info(std::string("Output: LINEAR\n"));
      break;
    case SMAX_OUT_TYPE:
      msg::info("Output: SOFTMAX\n");
      break;
  }

}

void Nnet::printGeometry(){
  std::ostringstream message;
  msg::info(std::string("### Geometry ###\n"));
  message << "Input units: " << _nInputUnits << std::endl;
  msg::info(message);
  if(_hiddenLayerSizes.size()>0){
    for(int iLayer =0; iLayer < _hiddenLayerSizes.size(); iLayer++) {
      message << "Hidden Layer " << iLayer + 1 << " units: " << _hiddenLayerSizes[iLayer] << std::endl;
      msg::info(message);
    }
  }else{
    msg::info(std::string("No hidden Layers\n"));
  }
  message << "Output units: " << _nOutputUnits << "\n################\n";
  msg::info(message);

}



void Nnet::printWeights(int iWeightsIndex){
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if(iWeightsIndex < _hiddenWeights.size()){
    message << "Weights " << iWeightsIndex << " ";
    if(iWeightsIndex == 0){
      nRows = _nInputUnits;
      nCols = _hiddenLayerSizes[0];
    }else{
      nRows = _hiddenLayerSizes[iWeightsIndex-1];
      nCols = _hiddenLayerSizes[iWeightsIndex];
    }
    if(nRows * nCols ==  _hiddenWeights[iWeightsIndex].size()){
      message << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        message << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          message << std::fixed;
          message << std::setprecision(2) << _hiddenWeights[iWeightsIndex][iCol*nRows+iRow] << " | ";
        }
        message << std::endl;
        msg::info(message);
      }
    }else{
      message << "Error printing weights " << nRows << " by " << nCols << " as it is actually " << _hiddenWeights[iWeightsIndex].size() << std::endl;
      msg::error(message);
    }
    if(iWeightsIndex < _hiddenBiases.size()){
      message << "Bias Vector: " << std::endl;
      for(std::vector<double>::const_iterator it = _hiddenBiases[iWeightsIndex].begin(); it !=  _hiddenBiases[iWeightsIndex].end(); ++it){
        if(it == _hiddenBiases[iWeightsIndex].begin()){
          message << *it;
        }else{
          message <<  "| " << *it;
        }
      }
      message << std::endl;
      msg::info(message);
    }else{
      msg::error(std::string("Invalid Bias index!\n"));
    }
  }else{
    if(iWeightsIndex == _hiddenWeights.size()){
      printOutputWeights();
    }else{
      msg::error(std::string("Invalid layer index!\n"));
    }
  }
}

void Nnet::printOutputWeights(){
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if( _nOutputUnits > 0){
    message << "Output Weights ";
    if(_hiddenLayerSizes.size() == 0){
      nRows = _nInputUnits;
      nCols = _nOutputUnits;
    }else{
      nRows = _hiddenLayerSizes[_hiddenLayerSizes.size()-1];
      nCols = _nOutputUnits;
    }

    message << "( " << nRows << " x " << nCols << " )" << std::endl;
    for(int iRow = 0; iRow < nRows; iRow++){
      message << "| " ;
      for(int iCol = 0; iCol < nCols; iCol++){
        message << std::fixed;
        message << std::setprecision(2) << _outputWeights[iCol*nRows+iRow] << " | ";
      }
      message << std::endl;
      msg::info(message);
    }
    message << "Bias Vector: " << std::endl;
    for(std::vector<double>::const_iterator it = _outputBiases.begin(); it !=  _outputBiases.end(); ++it){
      if(it == _outputBiases.begin()){
        message << *it;
      }else{
        message <<  "| " << *it;
      }
    }
    message << std::endl;
    msg::info(message);
  }
  return;
}



void Nnet::printOutputUnitValues(size_t nRecords){
  std::ostringstream message;
  msg::info(std::string("Printing the Output Unit Values\n"));
  if(_labelsGenerated){
    if(nRecords == 0){
      nRecords = _nDataRecords;
    }else{
      nRecords = std::min(nRecords, _nDataRecords);
    }

    for(int iRow = 0; iRow < nRecords; iRow++){
      message << " Record " << iRow + 1 << ": \t";
      for(int iCol = 0; iCol < _nOutputUnits; iCol++){
        message << std::fixed;
        message << std::setprecision(2) << _generatedLabels[(iCol*_nDataRecords)+iRow] << " | ";
      }
      message << std::endl;
      msg::info(message);
    }
  }else{
    msg::error(std::string("No data created\n"));
  }
}


void Nnet::printFeedForwardValues(int iIndex){
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if(iIndex >= 0 && iIndex < _feedForwardValues.size()){
    message << "FF " << iIndex << " ";;
    nRows = _nDataRecords;
    nCols = _hiddenLayerSizes[iIndex];
    if(nRows * nCols ==  _feedForwardValues[iIndex].size()){
      message << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        message << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          message << std::fixed;
          message << std::setprecision(2) << _feedForwardValues[iIndex][iCol*nRows +iRow] << " | ";
        }
        message << std::endl;
        msg::info(message);
      }
    }else{
      message << "Error printing FF values " << nRows << " by " << nCols << " as it is actually " << _feedForwardValues[iIndex].size() << std::endl;
      msg::error(message);
    }
  }else{
    msg::error(std::string("Invalid FF index!\n"));
  }
}

//// RCPP stuff

#ifndef IGNORE_THIS_RCPP_CODE

Nnet::Nnet(Rcpp::IntegerVector networkGeometry,
     Rcpp::String hiddenUnitActivation,
     Rcpp::String outputUnitActivation){
  std::ostringstream message;
  bool geometryValid = true;
  size_t nInputSize = 0, nOutputSize = 0;
  std::vector<int> hiddenLayerSizes;

  //// DEFAULTS ////
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = Nnet::LIN_OUT_TYPE;
  _activationType = Nnet::LIN_ACT_TYPE;


  _dataLoaded = false;
  _dataLabelsLoaded = false;
  _nDataRecords = 0;
  _labelsGenerated = false;

  _outputDir = "~/";
  //// END DEFAULTS ////

  bool activationTypeValid = false;
  if(hiddenUnitActivation == "linear"){
    activationTypeValid = true;
    _activationType = Nnet::LIN_ACT_TYPE;
  }
  if(hiddenUnitActivation == "tanh"){
    activationTypeValid = true;
    _activationType = Nnet::TANH_ACT_TYPE;
  }
  if(hiddenUnitActivation == "relu"){
    activationTypeValid = true;
    _activationType = Nnet::RELU_ACT_TYPE;
  }
  if(!activationTypeValid){
    msg::error(std::string("Invalid Hidden Unit Activation, only 'linear' or 'tanh' or 'relu'!\n"));
  }

  bool outputTypeValid = false;
  if(outputUnitActivation == "linear"){
    outputTypeValid = true;
    _outputType = Nnet::LIN_OUT_TYPE;
  }
  if(outputUnitActivation == "softmax"){
    outputTypeValid = true;
    _outputType = Nnet::SMAX_OUT_TYPE;
  }
  if(!outputTypeValid){
    msg::error(std::string("Invalid Output Unit Type, only 'linear' or 'softmax'!\n"));
  }

  size_t nLayers = networkGeometry.length();
  if (nLayers < 2){
    msg::error(std::string("Network Geometry Size Vector is of size 1, needs to include at least input and output sizes!\n"));
    geometryValid = false;
  }else{
    nInputSize = networkGeometry[0];
    nOutputSize = networkGeometry[networkGeometry.length()-1];

    if(nInputSize < 1){
      msg::error(std::string("Number of Input Units needs to be >0\n"));
      geometryValid = false;
    }
    if(nInputSize < 1){
      msg::error(std::string("Number of Output Units needs to be >0\n"));
      geometryValid = false;
    }
    hiddenLayerSizes.resize(0);
    for(size_t iHidden = 1; iHidden < networkGeometry.length()-1; iHidden++){
      hiddenLayerSizes.push_back(networkGeometry[iHidden]);
      if(networkGeometry[iHidden] < 1){
        geometryValid = false;
        msg::error(std::string("Network Geometry Size Vector contains 0!\n"));
        break;
      }
    }
  }
  if(geometryValid){
    _nInputUnits = nInputSize;
    _nOutputUnits = nOutputSize;
    _hiddenLayerSizes.resize(0);
    for(std::vector<int>::const_iterator it = hiddenLayerSizes.begin(); it != hiddenLayerSizes.end(); ++it) {
      _hiddenLayerSizes.push_back(*it);
    }
    size_t nCurrentInputWidth = _nInputUnits;

    // std::cout << "Initialising weights" << std::endl;
    _hiddenWeights.resize(0);
    _hiddenBiases.resize(0);

    if(_hiddenLayerSizes.size() > 0){
      for(int i = 0 ; i < _hiddenLayerSizes.size(); i++) {
        _hiddenWeights.resize(i+1);
        _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i]);
        _hiddenBiases.resize(i+1);
        _hiddenBiases[i].resize(_hiddenLayerSizes[i], 0);
        nCurrentInputWidth = _hiddenLayerSizes[i];
        std::fill(_hiddenWeights[i].begin(),_hiddenWeights[i].end(),0.0);
        std::fill(_hiddenBiases[i].begin(),_hiddenBiases[i].end(),0.0);
      }
    }

    if(_nOutputUnits > 0){
      _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 0);
      _outputBiases.resize(_nOutputUnits, 0);
      std::fill(_outputWeights.begin(),_outputWeights.end(),0.0);
      std::fill(_outputBiases.begin(),_outputBiases.end(),0.0);
    }

  }
  if((!geometryValid) | (!activationTypeValid) | (!outputTypeValid)){
    msg::error(std::string("There were problems with Network Specifications\n"));
    msg::error(std::string("Check the Activation Type, Output Type and Geometry\n"));
  }
}

SEXP Nnet::generatedLabelsR() const {
  bool allOk = true;
  std::ostringstream message;
  if(!_labelsGenerated){
    allOk = false;
  }else{
    if(_generatedLabels.size() != _nOutputUnits*_nDataRecords){
      allOk = false;
      message << "Problem with generated labels, expected size: " << _nOutputUnits*_nDataRecords << " but found size: " << _generatedLabels.size() << std::endl;
      msg::warn(message);
    }
  }
  if(allOk){
    Rcpp::NumericMatrix generatedLabels( _nDataRecords , (int)_nOutputUnits);

    for(int iDatum = 0; iDatum < _nDataRecords; iDatum++){
      for(int iUnit = 0; iUnit < _nOutputUnits; iUnit++){
        generatedLabels(iDatum,iUnit) = _generatedLabels[iUnit*_nDataRecords + iDatum];
      }
    }
    return generatedLabels;
  }else{
    return R_NilValue;
  }
}



SEXP Nnet::getWgtAndBiasR(int iIndex) const{
  bool allOk = true;
  bool doOutputWeights = false;
  const std::vector<double> *wgts_ptr;
  const std::vector<double> *biases_ptr;
  std::ostringstream message;

  // Convert from R 1 index to C++ 0 index
  if(iIndex < 1){
    msg::error(std::string("Invalid index, less than 1\n"));
    allOk= false;
  }
  if(allOk){
    if (iIndex > _hiddenWeights.size()+1){
      message << "Invalid index, greater than " << _hiddenWeights.size() + 2 << std::endl;
      msg::error(message);
      allOk= false;
    }
  }

  if(allOk){
    iIndex = iIndex - 1;
    if(iIndex == _hiddenWeights.size()){
      doOutputWeights = true;
      wgts_ptr = &_outputWeights;
      biases_ptr = &_outputBiases;
    }else{
      wgts_ptr = &_hiddenWeights[iIndex];
      biases_ptr = &_hiddenBiases[iIndex];
    }
  }

  size_t nRows = 0;
  size_t nCols = 0;

  if(allOk){
    if(iIndex == 0){
      nRows = _nInputUnits;
    }else{
      nRows = _hiddenLayerSizes[iIndex-1];
    }
    if(doOutputWeights){
      nCols = _nOutputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex];
    }
  }
  if(allOk){
    if(nRows * nCols !=  wgts_ptr->size()) {
      message << "Weights matrix expected to be " << nRows << " by " << nCols << " but found to be " << wgts_ptr->size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    if(nCols != biases_ptr->size()){
      message << "Bias vector expected to be " << nCols << " but found to be " << biases_ptr->size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    Rcpp::NumericMatrix weightsMatrix((int)nRows , (int)nCols);
    Rcpp::NumericVector biasVector((int)nCols);
    for(int iRow = 0; iRow < nRows; iRow++){
      for(int iCol = 0; iCol < nCols; iCol++){
        weightsMatrix(iRow,iCol) =  (*wgts_ptr)[iCol*nRows+iRow];
      }
    }

    for(int iBias = 0; iBias < nCols; iBias++){
      biasVector(iBias) =  (*biases_ptr)[iBias];
    }

    Rcpp::List wgtAndBias = Rcpp::List::create(Rcpp::Named("weight") = weightsMatrix , Rcpp::Named("bias") = biasVector);
    return wgtAndBias;
  }else{
    return R_NilValue;
  }
  return R_NilValue;
}

Rcpp::LogicalVector Nnet::setWgtAndBiasR(int iIndex, Rcpp::NumericMatrix weights, Rcpp::NumericVector bias){
  bool allOk = true;
  bool doOutputWeights = false;
  std::vector<double> *wgts_ptr;
  std::vector<double> *biases_ptr;
  std::ostringstream message;

  if(iIndex < 1){
    msg::error(std::string("Invalid index, less than 1\n"));
    allOk = false;
  }
  if(allOk){
    if (iIndex > _hiddenWeights.size()+1){
      message << "Invalid index, greater than " << _hiddenWeights.size() + 1 << std::endl;
      msg::error(message);
      allOk = false;
    }
  }

  if(allOk){
    // Convert from R 1 index to C++ 0 index
    iIndex = iIndex - 1;
    if(iIndex == _hiddenWeights.size()){
      doOutputWeights = true;
      wgts_ptr = &_outputWeights;
      biases_ptr = &_outputBiases;
    }else{
      wgts_ptr = &_hiddenWeights[iIndex];
      biases_ptr = &_hiddenBiases[iIndex];
    }
  }

  size_t nRows = 0;
  size_t nCols = 0;

  if(allOk){
    if(iIndex == 0){
      nRows = _nInputUnits;
    }else{
      nRows = _hiddenLayerSizes[iIndex-1];
    }
    if(doOutputWeights){
      nCols = _nOutputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex];
    }
  }
  if(allOk){
    if(nRows * nCols !=  weights.length()) {
      message << "Weights matrix expected to be " << nRows << " by " << nCols << " but found to be " << weights.length() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    if(nCols != bias.length()){
      message << "Bias vector expected to be " << nCols << " but found to be " << bias.length() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }
  if(allOk){
    for(int iRow = 0; iRow < nRows; iRow++){
      for(int iCol = 0; iCol < nCols; iCol++){
        (*wgts_ptr)[iCol*nRows+iRow] = weights(iRow,iCol);
      }
    }

    for(int iBias = 0; iBias < nCols; iBias++){
      (*biases_ptr)[iBias] = bias(iBias);
    }
  }
  return allOk;
}



SEXP Nnet::getFFValues(int iIndex) const{
  bool allOk = true;
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if(iIndex > _feedForwardValues.size()){
    message << "Invalid FF index, should be at most " << _feedForwardValues.size() << std::endl;
    msg::error(message);
    allOk = false;
  }
  if(allOk && iIndex < 1){
    msg::error(std::string("Invalid FF indexm should be greater than zero!\n"));
    allOk = false;
  }
  if(allOk){
    iIndex = iIndex - 1;
    nRows = _nDataRecords;
    nCols = _hiddenLayerSizes[iIndex];
    if(nRows * nCols !=  _feedForwardValues[iIndex].size()){
      message << "Error printing FF values " << nRows << " by " << nCols << " as it is actually " << _feedForwardValues[iIndex].size() << std::endl;
      msg::error(message);
      allOk = false;
    }
  }

  if(allOk){
    Rcpp::NumericMatrix feedforwardMatrix((int)nRows , (int)nCols);
    for(int iRow = 0; iRow < nRows; iRow++){
      message << "| " ;
      for(int iCol = 0; iCol < nCols; iCol++){
        feedforwardMatrix(iRow,iCol) =  _feedForwardValues[iIndex][iCol*nRows +iRow];
      }
    }
    return feedforwardMatrix;
  }
  return R_NilValue;
}

RCPP_MODULE(af_nnet) {

   Rcpp::class_<Nnet>("Nnet")

   .constructor<
      Rcpp::IntegerVector ,
      Rcpp::String,
      Rcpp::String
    >("net geometry; hidden unit activation; output unit type")

   .property("HiddenLayerSizes",&Nnet::getHiddenLayerSizes, &Nnet::setHiddenLayerSizes, "Vector of Hidden layer sizes")
   .property("actType", &Nnet::getActivationType, &Nnet::setActivationType,"Hidden layer activation type")
   .property("outType",&Nnet::getOutputType , &Nnet::setOutputType, "Output Unit Type")

   .property("dataClamped",&Nnet::dataLoaded, "is data loaded (clamped)")
   .property("dataClampedWithLabels",&Nnet::dataAndLabelsLoaded, "is data loaded (clamped) together with labels")
   .property("nDataRecords",&Nnet::nDataRecords, "is data loaded (clamped)")
   .property("labelsGenerated",&Nnet::labelsGenerated, "have labels been generated")
   .property("generatedLabels",&Nnet::generatedLabelsR, "Generated labels")

    .method("printGeometry", &Nnet::printGeometry, "Print the data")
    .method("clampData", &Nnet::clampData, "Clamp data")
    .method("feedForward", & Nnet::feedForward, "feedforward")
    .method("getWeights", &Nnet::getWgtAndBiasR,"Using Index get weights and bias as a list")
    .method("setWeights", &Nnet::setWgtAndBiasR, "Using Index set weights and bias as a list")
    .method("getFedForward", &Nnet::getFFValues,"Using Index get feedforward values of the network")
   ;


  //http://stackoverflow.com/questions/33549712/using-a-class-as-a-parameter-in-a-constructor-of-another-class

  Rcpp::class_<Dataset>("Dataset")
    .constructor<
    Rcpp::String,
    Rcpp::String,
    Rcpp::LogicalVector,
    Rcpp::String
  >("data and labels")

  .property("hasLabels", &Dataset::labelsLoaded, "Does the data have labels?")
  .property("isDataLoaded", &Dataset::dataLoaded, "Is there data loaded?")
  .property("nRecords", &Dataset::nRecords,"How many records are loaded?")
  .property("nFields", &Dataset::nFields,"How many fields in the data?")
  .property("nLabelFields", &Dataset::nLabelFields,"How many label fields in the data?")
  .property("isPCA", &Dataset::isPcaDone,"Has PCA been performed on this dataset?")
  .property("pcaMat", &Dataset::getPcaMatrixR,"get the PCA matrix if available")
  .property("transformType", &Dataset::getNormType,"get the data transformation type, if there was one")
  .property("paramsForTransform", &Dataset::getNormParamMat,"get the data transformation params, in a matrix")
  .property("data",&Dataset::getDataR, "Get the data into R")
  .property("labels",&Dataset::getLabelsR, "Get the labels into R, if available")

  .method("printData", &Dataset::printData, "Print the data")
  .method("printLabels", &Dataset::printLabels, "Print the data labels, if available")

  .method("nrow",&Dataset::nRecords, "Number of records in the dataset")
  .method("ncol",&Dataset::nFields, "Number of fields in the dataset")

  .method("doPCA",&Dataset::doPca, "Map data to its principle comps, with reduction if requested")
  .method("isPCA",&Dataset::isPcaDone, "Has PCA been performed?")

  .method("internalNorm",&Dataset::analyseAndNorm, "Normalise the data, with type snorm or range")
  .method("paramNorm",&Dataset::normWithParams, "Normalise the data, with type snorm or range")
  .method("copyTransform",&Dataset::normWithPrototype, "Transform dataset like given dataset")
  ;


  Rcpp::class_<Backpropper>("Backpropper")
    .constructor<
    Nnet&
    >("Algorithm to find weights for a Neural Network")

  .property("lossType", &Backpropper::getLossType, &Backpropper::setLossType, "Loss type for labels when training")
  .property("cost",&Backpropper::calcCostR, "Cost")

  .method("initWeights", &Backpropper::initialiseWeights, "Initialise the network weights")
  .method("doBPOptim", &Backpropper::doBackPropOptimise, "Do Backprop Optimisation")

    ;
}

#endif

