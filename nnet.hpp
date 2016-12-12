#ifndef __MOC_Nnet_
#define __MOC_Nnet_

#include "rng.hpp"

class nnet{
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
  
  enum normDataType
  {
    DATA_NORM_NONE,
    DATA_STAN_NORM,
    DATA_RANGE_BOUND
  };
  
private:
  rng *_rng;
  std::string _outputDir;
  
  
  // Things to do with the input data
  size_t _nInputUnits;
  bool _trainDataLoaded;
  bool _trainDataLabelsLoaded;
  bool _trainDataShuffled;
  bool _trainDataPCA;
  size_t _nTrainDataRecords;
  std::vector<double> _trainDataLabels;
  // The input is considered the first feedforward
  std::vector<std::vector<double> > _trainDataFeedForwardValues;
  
  
  // preprocessing/changes to the input data
  std::vector<double> _trainDataNormParam1;
  std::vector<double> _trainDataNormParam2;
  std::vector<int> _trainDataShuffleIndex;
  std::vector<double> _pcaEigenMat;
  normDataType _trainDataNormType;
  normDataType _nonTrainDataNormType;
  
  // Things to do with testdata
  bool _nonTrainDataLabelsLoaded = false;
  bool _nonTrainDataLoaded = false;
  size_t _nNonTrainDataRecords = 0;
  bool _nonTrainDataPCA = false;
  size_t _nNonTrainDataInputUnits;
  size_t _nNonTrainDataOutputUnits;
  std::vector<std::vector<double> > _nonTrainDataFeedForwardValues;
  std::vector<double> _nonTrainDataLabels;
  
  
  
  bool _nonTrainLabelsGenerated;
  std::vector<double> _nonTrainGeneratedLabels;
  
  // Things to do with geometry and weights
  std::vector<int> _hiddenLayerSizes;
  activationType _activationType;
  std::vector<std::vector<double> > _hiddenWeights;
  std::vector<std::vector<double> > _hiddenBiases;
  std::vector<std::vector<double> > _hiddenGradients;
  std::vector<double> _outputWeights;
  std::vector<double> _outputBiases;
  bool _weightsInitialised;
  
  // Things to do with the output
  size_t _nOutputUnits;
  std::vector<double> _trainGeneratedLabels;
  bool _trainLabelsGenerated;
  outputType _outputType;
  lossType _lossType;
 
  // Private functions
  void activateUnits(std::vector<double>& values);
  void activateUnitsAndCalcGradients(std::vector<double>& values, std::vector<double>& gradients);
  void activateOutput(std::vector<double>& values);
  void flowDataThroughNetwork(std::vector<std::vector<double> > dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients);
  // Functions
public:
  nnet();
  ~nnet();
  
// Import Data Methods
  bool loadTrainDataFromFile(char *filename,
                        bool hasHeader,
                        char delim);
  bool loadTrainDataLabelsFromFile(char *filename,
                              bool hasHeader,
                              char delim);
  bool loadNonTrainDataFromFile(char *filename,
                                bool hasHeader,
                                char delim);
  bool loadNonTrainDataLabelsFromFile(char *filename,
                                      bool hasHeader,
                                      char delim);
  
 
  // Set Network Details
  void setHiddenLayerSizes(const std::vector<int>& layerSizes);
  void setActivationType(activationType activationType);
  void setOutputType(outputType outputType);
  void setLossType(lossType lossType);

  // Preprocessing methods on Train Data
  void normTrainData(normDataType normType);
  void shuffleTrainData();
  void pcaTrainData(size_t dimensions);

  // Preprocessing methods on Non Train Data

  void normNonTrainData(normDataType normType);
  void pcaNonTrainData();
  
  
  // Weight fitting stuff
  void initialiseWeights(double stdev);
  void feedForwardTrainData();
  void backProp(size_t nBatchIndicator,
                double wgtLearnRate,
                double biasLearnRate,
                size_t nEpoch,
                bool doMomentum,
                double mom_mu,
                double mom_decay,
                size_t mom_decay_schedule,
                double mom_final,
                bool doTestCost);
  void feedForwardNonTrainData();
  
  std::vector<double> _epochTrainCostUpdates;
  std::vector<double> _epochTestCostUpdates;
  
  //Informational queries
  bool dataLoaded();
  bool numericLabels();
  bool classLabels();
  double getTrainDataCost();
  double getTrainDataAccuracy();
  double getNonTrainDataCost();
  double getNonTrainDataAccuracy();
  
  
// Print to screen methods
  void printOutValues();
  void printTrainData();
  void printTrainData(size_t rows);
  void printNonTrainData();
  void printNonTrainData(size_t rows);
  void printLabels();
  void printWeights(int iLayer);
  void printOutputWeights();
  void printGradients(int iLayer);
  void printGeometry();
  void printUnitType();
  void printOutputType();
  void printFeedForwardValues(int iIndex);
  
  
// Write to file methods
  void writeFeedForwardValues();
  void writeDataToFile(char *filename);
  void writeOutValues();
  void writeWeightValues();
  void writeEpochCostUpdates();
  void writeEpochTrainCostUpdates();
  void writeEpochTestCostUpdates();
};

#endif
