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

nnet::nnet(){
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = nnet::LIN_OUT_TYPE;
  _lossType = nnet::MSE_LOSS_TYPE;
  _activationType = nnet::LIN_ACT_TYPE;

  _dataLabelsLoaded = false;
  _dataLoaded = false;
  _nDataRecords = 0;

  _outputDir = "~/";
};

nnet::nnet(Dataset initialData){
  _nOutputUnits = 0;
  _nInputUnits = 0;
  _outputType = nnet::LIN_OUT_TYPE;
  _lossType = nnet::MSE_LOSS_TYPE;
  _activationType = nnet::LIN_ACT_TYPE;

  _dataLabelsLoaded = false;
  _dataLoaded = false;
  _nDataRecords = 0;

  if(initialData.dataLoaded()){
    _feedForwardValues.resize(1);
    _feedForwardValues[0] = initialData.data();
    _nInputUnits = initialData.nFields();
    _nDataRecords = initialData.nRecords();
    _dataLoaded = true;


    if(initialData.labelsLoaded()){
      _dataLabels = initialData.labels();
      _nOutputUnits = initialData.nLabelFields();
      _dataLabelsLoaded = true;

      _outputWeights.resize(_nInputUnits * _nOutputUnits, 0);
      _outputBiases.resize(_nOutputUnits, 0);

    }

  }
  _outputDir = "~/";
};


nnet::~nnet(){
};

void nnet::setHiddenLayerSizes(const std::vector<int>& layerSizes){
  _hiddenLayerSizes.resize(0);
  for(std::vector<int>::const_iterator it = layerSizes.begin(); it != layerSizes.end(); ++it) {
    _hiddenLayerSizes.push_back(*it);
  }
  initialiseWeights();
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

void nnet::setOutputFolder(char *filename){
  _outputDir = filename;
  return;
}


bool nnet::dataLoaded(){
  return _dataLoaded;
}


bool nnet::dataAndLabelsLoaded(){
  return _dataLoaded && _dataLabelsLoaded;
}

bool nnet::setDataAndLabels(Dataset dataToClamp){
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
    _feedForwardValues[0] = dataToClamp.data();
    _nInputUnits = dataToClamp.nFields();
    _nDataRecords = dataToClamp.nRecords();
    _dataLoaded = true;

    _dataLabels = dataToClamp.labels();
    _nOutputUnits = dataToClamp.nLabelFields();
    _dataLabelsLoaded = true;

  }
  return allOk;
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

double nnet::getCost(){
  return calcCost(_dataLabels,_generatedLabels,_nDataRecords,_nOutputUnits);
}

double nnet::getAccuracy(){
  return calcAccuracy(_dataLabels,_generatedLabels,_nDataRecords,_nOutputUnits);
}



double nnet::calcCost(std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits){

  double cost = 0.0;
  switch (_lossType){
    case nnet::MSE_LOSS_TYPE:
      cost = 0.0;
      for(int iRecord = 0; iRecord < nRecords; ++iRecord){
        for(int iUnit = 0; iUnit < nOutputUnits; ++iUnit){
          cost += pow(fitted[(iUnit*nRecords)+iRecord] - actual[(iUnit*nRecords)+iRecord],2.0)/nRecords;
        }
      }

      break;
    case  nnet::CROSS_ENT_TYPE:
      cost = 0.0;
      for(int iRecord = 0; iRecord < nRecords; ++iRecord){
        for(int iUnit = 0; iUnit < nOutputUnits; ++iUnit){
          if(actual[(iUnit*nRecords)+iRecord] > 0.5){
            cost -= log( fitted[(iUnit*nRecords)+iRecord]);
          }
        }
      }
  }
  return cost;
}

double nnet::calcAccuracy(std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits){
  double accuracy = 0.0;
  double maxProb = 0.0;
  bool correct = false;
  for(int iRecord = 0; iRecord < nRecords; ++iRecord){
    for(int iUnit = 0; iUnit < nOutputUnits; ++iUnit){
      if(iUnit == 0){
        maxProb = fitted[(iUnit*nRecords)+iRecord];
        if(actual[(iUnit*nRecords)+iRecord] > 0.5){
          correct = true;
        }else{
          correct = false;
        }
      }else{
        if(fitted[(iUnit*nRecords)+iRecord] > maxProb){
          maxProb = fitted[(iUnit*nRecords)+iRecord];
          if(actual[(iUnit*nRecords)+iRecord] > 0.5){
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
  return accuracy;
}

void nnet::initialiseWeights(){
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

void nnet::flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
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




void nnet::feedForward(){
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

//bool nnet::loadDataFromFile(char *filename, bool hasHeader, char delim){
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
//  _trainDataNormType = nnet::DATA_NORM_NONE;
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
//bool nnet::loadNonTrainDataLabelsFromFile(char *filename, bool hasHeader, char delim){
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

void nnet::writeWeights(){
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

void nnet::writeOutputUnitValues(){
  std::ostringstream oss;
  if(_labelsGenerated){
    oss << _outputDir << "outvalues.csv";
    mat_ops::writeMatrix(oss.str(), _generatedLabels,_nDataRecords,_nOutputUnits);
  }else{
    msg::warn(std::string("No output data created\n"));
  }
  return;
}


void nnet::printUnitType(){
  switch (_activationType){
    case nnet::TANH_ACT_TYPE:
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

void nnet::printOutputType(){
  switch (_outputType){
    case nnet::LIN_OUT_TYPE:
      msg::info(std::string("Output: LINEAR\n"));
      break;
    case SMAX_OUT_TYPE:
      msg::info("Output: SOFTMAX\n");
      break;
  }

}

void nnet::printGeometry(){
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



void nnet::printWeights(int iWeightsIndex){
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

void nnet::printOutputWeights(){
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



void nnet::printOutputUnitValues(size_t nRecords){
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
void nnet::printFeedForwardValues(int iIndex){
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


