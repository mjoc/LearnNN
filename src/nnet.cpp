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

Nnet::Nnet(Dataset initialData){
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = Nnet::LIN_OUT_TYPE;
  _activationType = Nnet::LIN_ACT_TYPE;

  _dataLabelsLoaded = false;
  _dataLoaded = false;
  _nDataRecords = 0;

  if(initialData.dataLoaded()){
    _feedForwardValues.resize(1);
    _feedForwardValues[0] = *initialData.data();
    _nInputUnits = initialData.nFields();
    _nDataRecords = initialData.nRecords();
    _dataLoaded = true;


    if(initialData.labelsLoaded()){
      _dataLabels = *initialData.labels();
      _nOutputUnits = initialData.nLabelFields();
      _dataLabelsLoaded = true;

      _outputWeights.resize(_nInputUnits * _nOutputUnits, 0);
      _outputBiases.resize(_nOutputUnits, 0);

    }

  }
  _outputDir = "~/";
};


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
  msg::error(std::string("here\n"));
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


bool Nnet::dataAndLabelsLoaded(){
  return _dataLoaded && _dataLabelsLoaded;
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
    _feedForwardValues.resize(1);
    _feedForwardValues[0] = *dataToClamp.data();
    _nInputUnits = dataToClamp.nFields();
    _nDataRecords = dataToClamp.nRecords();
    _dataLoaded = true;

    _dataLabels = *dataToClamp.labels();
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
      _hiddenBiases[i].resize(_hiddenLayerSizes[i], 0);
      nCurrentInputWidth = _hiddenLayerSizes[i];
    }
  }

  if(_nOutputUnits > 0){
    _outputWeights.resize(nCurrentInputWidth * _nOutputUnits, 0);
    _outputBiases.resize(_nOutputUnits, 0);
  }
  return;
}

void Nnet::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                                         std::vector<double>& dataflowMatrix){
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
    activateUnits(dataflowStages[iLayer+1]);

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

    flowDataThroughNetwork(_feedForwardValues, tempMatrixForLabels);

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

//bool Nnet::loadDataFromFile(char *filename, bool hasHeader, char delim){
//  bool allOk = true;
//  bool first = true;
//  int nRecords = 0;
//  size_t nDelims = 0;
//  std::vector<double> indata;
//
//  _dataLoaded = false;
//  _dataLoaded = false;
//
//  _nDataRecords = 0;
//  _trainDataNormType = Nnet::DATA_NORM_NONE;
//  _dataPCA = false;
//
//  std::ifstream infile(filename, std::ios_base::in);
//
//  if (!infile.is_open()){
//    allOk = false;
//  }else{
//    std::cout << "Reading in data from file " << filename << std::endl;
//    _feedForwardValues.resize(1);
//    _feedForwardValues[0].resize(0);
//    if(hasHeader){
//      std::string headerline;
//      std::getline(infile, headerline);
//      nDelims = std::count(headerline.begin(), headerline.end(), delim);
//      first = false;
//    }
//    // http://stackoverflow.com/questions/18818777/c-program-for-reading-an-unknown-size-csv-file-filled-only-with-floats-with
//
//    for (std::string line; std::getline(infile, line); )
//    {
//      if(line.find_first_not_of(' ') == std::string::npos){
//        break;
//      }else{
//        nRecords++;
//        if(first){
//          nDelims = std::count(line.begin(), line.end(), delim);
//          first = false;
//        }else{
//          if(nDelims != std::count(line.begin(), line.end(), delim)){
//            std::cout << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
//            allOk = false;
//          }
//        }
//
//        std::replace(line.begin(), line.end(), delim, ' ');
//        std::istringstream in(line);
//
//
//        if(allOk){
//          std::vector<double> rowValues = std::vector<double>(std::istream_iterator<double>(in), std::istream_iterator<double>());
//          for (std::vector<double>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
//            indata.push_back(*it);
//          }
//        }else{
//          break;
//        }
//      }
//    }
//  }
//  if(allOk){
//    if((indata.size()% (nDelims+1)) != 0 ){
//      std::cout << "Number of data is not an integer multiple of first line fields!\n";
//      allOk = false;
//    }
//
//  }
//  if(allOk){
//    _nDataRecords = nRecords;
//    _nNonTrainDataInputUnits = nDelims + 1;
//
//    _feedForwardValues.resize(indata.size());
//    for(size_t iCol = 0; iCol < _nNonTrainDataInputUnits; iCol++){
//      for(size_t iRow = 0; iRow < _nDataRecords; iRow++){
//        _feedForwardValues[0][iCol*_nDataRecords + iRow] = indata[ (iRow*_nNonTrainDataInputUnits) + iCol];
//      }
//    }
//    _dataLoaded = true;
//
//    std::cout << "Read " << _feedForwardValues[0].size() << " data of " << nRecords << " by " << nDelims + 1 << std::endl;
//  }
//  return allOk;
//};
//
//bool Nnet::loadNonTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
//  bool allOk = true;
//  bool first = true;
//  int nRecords = 0;
//  size_t nDelims = 0;
//  std::vector<double> indata;
//
//  _dataLoaded = false;
//  _dataLabels.resize(0);
//
//  std::ifstream infile(filename, std::ios_base::in);
//
//  if (!infile.is_open()){
//    allOk = false;
//  }else{
//    std::cout << "Reading in data from file " << filename << std::endl;
//    if(hasHeader){
//      std::string headerline;
//      std::getline(infile, headerline);
//      nDelims = std::count(headerline.begin(), headerline.end(), delim);
//      first = false;
//    }
//    // http://stackoverflow.com/questions/18818777/c-program-for-reading-an-unknown-size-csv-file-filled-only-with-floats-with
//
//    for (std::string line; std::getline(infile, line); )
//    {
//      if(line.find_first_not_of(' ') == std::string::npos){
//        break;
//      }else{
//        nRecords++;
//        if(first){
//          nDelims = std::count(line.begin(), line.end(), delim);
//          first = false;
//        }else{
//          if(nDelims != std::count(line.begin(), line.end(), delim)){
//            std::cout << "Problem with line " << nRecords << "; it has " << std::count(line.begin(), line.end(), delim) << " delimiters, expected " << nDelims << std::endl;
//            allOk = false;
//          }
//        }
//
//        std::replace(line.begin(), line.end(), delim, ' ');
//        std::istringstream in(line);
//
//        if(allOk){
//          std::vector<int> rowValues = std::vector<int>(std::istream_iterator<int>(in), std::istream_iterator<int>());
//          for (std::vector<int>::const_iterator it(rowValues.begin()), end(rowValues.end()); it != end; ++it) {
//            indata.push_back(*it);
//          }
//        }else{
//          break;
//        }
//      }
//    }
//  }
//
//  if(allOk){
//    if((indata.size()% (nDelims+1)) != 0 ){
//      std::cout << "Number of data is not an integer multiple of first line fields!" << std::endl;
//      allOk = false;
//    }
//    if(indata.size() != _nDataRecords * (nDelims+1)){
//      std::cout << "Expected " << _nDataRecords * (nDelims+1) << "Labels, got " << indata.size() << "!" << std::endl;
//      allOk = false;
//    }
//  }
//
//  if(allOk){
//    _nNonTrainDataOutputUnits = nDelims + 1;
//    _dataLabels.resize(indata.size());
//
//    // Flip from row major to column major
//    for(size_t iCol = 0; iCol < _nNonTrainDataOutputUnits; iCol++){
//      for(size_t iRow = 0; iRow < _nDataRecords; iRow++){
//        _dataLabels[iCol*_nDataRecords + iRow] = indata[iRow*_nNonTrainDataOutputUnits + iCol];
//      }
//    }
//
//    _dataLoaded = true;
//
//    std::cout << "Read " << _dataLabels.size() << " labels of " << nRecords << " by " << _nOutputUnits << std::endl;
//  }
//  return allOk;
//};

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
    if(i == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[i -1];
    }

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
      message << "Layer " << iLayer + 1 << " units: " << _hiddenLayerSizes[iLayer] << std::endl;
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
          message << std::setprecision(2) << _hiddenWeights[iWeightsIndex][iRow*nCols+iCol] << " | ";
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
    msg::error(std::string("Invalid layer index!\n"));
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
        message << std::setprecision(2) << _outputWeights[iRow*nCols+iCol] << " | ";
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

/* FIX THIS TO COLUMN MAJOR */
void Nnet::printFeedForwardValues(int iIndex){
  std::ostringstream message;
  size_t nRows = 0;
  size_t nCols = 0;
  if(iIndex < _feedForwardValues.size()){
    message << "FF " << iIndex << " ";;
    nRows = _nDataRecords;
    if(iIndex == 0){
      nCols = _nInputUnits;
    }else{
      nCols = _hiddenLayerSizes[iIndex -1];
    }
    if(nRows * nCols ==  _feedForwardValues[iIndex].size()){
      message << "( " << nRows << " x " << nCols << " )" << std::endl;
      for(int iRow = 0; iRow < nRows; iRow++){
        message << "| " ;
        for(int iCol = 0; iCol < nCols; iCol++){
          message << std::fixed;
          message << std::setprecision(2) << _feedForwardValues[iIndex][iRow*nCols+iCol] << " | ";
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

Nnet::Nnet(Rcpp::IntegerVector networkgeometry,
     Rcpp::String hiddenUnitActivation,
     Rcpp::String outputUnitActivation){
  std::ostringstream message;
  bool geometryValid = true;
  size_t nInputSize, nOutputSize;
  std::vector<int> hiddenLayerSizes;

  //// DEFAULTS ////
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = Nnet::LIN_OUT_TYPE;
  _activationType = Nnet::LIN_ACT_TYPE;


  _dataLoaded = false;
  _dataLabelsLoaded = false;
  _nDataRecords = 0;

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

  size_t nLayers = networkgeometry.length();
  if (nLayers < 3){
    msg::error(std::string("Network Geometry Size Vector is of size 1, needs to include at least input and output sizes!\n"));
    geometryValid = false;
  }else{
    nInputSize = networkgeometry[0];
    nOutputSize = networkgeometry[networkgeometry.length()-1];

    message << "Input size " << nInputSize << " output size " << nOutputSize << std::endl;
    msg::info(message);

    if(nInputSize < 1){
      msg::error(std::string("Number of Input Units needs to be >0\n"));
      geometryValid = false;
    }
    if(nInputSize < 1){
      msg::error(std::string("Number of Output Units needs to be >0\n"));
      geometryValid = false;
    }
    hiddenLayerSizes.resize(0);
    for(size_t iHidden = 1; iHidden < networkgeometry.length()-2; iHidden++){
      hiddenLayerSizes.push_back(networkgeometry[iHidden]);
      if(networkgeometry[iHidden] < 1){
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
        _hiddenWeights[i].resize(nCurrentInputWidth*_hiddenLayerSizes[i], 0);
        _hiddenBiases.resize(i+1);
        _hiddenBiases[i].resize(_hiddenLayerSizes[i], 0);
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



RCPP_MODULE(af_nnet) {

  Rcpp::class_<Nnet>("Nnet")

  .constructor<Rcpp::IntegerVector , Rcpp::String, Rcpp::String >("net details")

  .method("printGeom", &Nnet::printGeometry, "Print the data")
  .property("HiddenLayers",&Nnet::getHiddenLayerSizes, &Nnet::setHiddenLayerSizes, "Vector of Hidden layer sizes")
  .property("actType", &Nnet::getActivationType, &Nnet::setActivationType,"Hidden layer activation type")
  .property("outType",&Nnet::getOutputType , &Nnet::setOutputType, "Output Unit Type")



  ;
}



