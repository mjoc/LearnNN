#ifndef __MOC_Nnet_
#define __MOC_Nnet_

#include "dataset.hpp"
#include <vector>

class nnet{
  friend class backpropper;
public:
  
  
  enum activationType
  {
    LIN_ACT_TYPE,
    TANH_ACT_TYPE,
    RELU_ACT_TYPE
  };
  
  
  enum outputType
  {
    LIN_OUT_TYPE,
    SMAX_OUT_TYPE
  };
  
  enum lossType
  {
    MSE_LOSS_TYPE,
    CROSS_ENT_TYPE
  };
  
  
private:
  size_t _nInputUnits;
  // Things to do with geometry and weights
  std::vector<int> _hiddenLayerSizes;
  activationType _activationType;
  std::vector<std::vector<double> > _hiddenWeights;
  std::vector<std::vector<double> > _hiddenBiases;
  
  std::vector<double> _outputWeights;
  std::vector<double> _outputBiases;
  size_t _nOutputUnits;
  outputType _outputType;
  lossType _lossType;
  
  // Things to do with non- Train Data
  bool _dataLabelsLoaded;
  bool _dataLoaded;
  size_t _nDataRecords;
  bool _doPca;
  size_t _nPcaDimensions;
  //size_t _nNonTrainDataInputUnits;
  //size_t _nNonTrainDataOutputUnits;
  std::vector<std::vector<double> > _feedForwardValues;
  std::vector<double> _dataLabels;
  bool _labelsGenerated;
  std::vector<double> _generatedLabels;
  
  // Private functions
  void activateUnits(std::vector<double>& values);
  void activateOutput(std::vector<double>& values);

  // Functions
  void flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataflowMatrix);
  double calcCost(std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits);
  double calcAccuracy(std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits);
  // Wrting out to file
  std::string _outputDir;
  
public:
  nnet();
  nnet(dataset initialData);
  ~nnet();
  
  
  
// Import Data Methods
  bool loadDataFromFile(char *filename,
                        bool hasHeader,
                        char delim);
  bool loadDataLabelsFromFile(char *filename,
                              bool hasHeader,
                              char delim);
  bool dataLoaded();
  bool dataAndLabelsLoaded();
  
  // Set Network Details
  void setInputSize(size_t nInputUnits);
  void setOutputSize(size_t nOutputUnits);
  void setHiddenLayerSizes(const std::vector<int>& layerSizes);
  void setActivationType(activationType activationType);
  void setOutputType(outputType outputType);
  void setLossType(lossType lossType);
 
  bool setDataAndLabels(dataset dataToClamp);
  
  void initialiseWeights();
  void feedForward();
  
  //Informational queries
  bool numericLabels();
  bool classLabels();
  
  double getCost();
  double getAccuracy();
  
  // Write to file
  void setOutputFolder(char *filename);
  void writeWeights();
  void writeFeedForwardValues();
  void writeOutputUnitValues();
  
// Print to screen methods
  void printOutputUnitValues(size_t nRecords = 0);
  void printWeights(int iLayer);
  void printOutputWeights();
  void printGeometry();
  void printUnitType();
  void printOutputType();
  void printFeedForwardValues(int iIndex);
  
  
};

#endif
