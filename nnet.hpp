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
    DATA_STAN_NORM,
    DATA_RANGE_BOUND
  };
  
private:
  rng *_rng;
  std::string _outputDir;
  activationType _activationType;
  outputType _outputType;
  lossType _lossType;
  size_t _nInputUnits;
  size_t _nOutputUnits;

  bool _indataNormed;
  bool _indataShuffled;
  bool _weightsInitialised;
  bool _indataLoaded;
  bool _outdataCreated;
  bool _indataLabelsLoaded;

  normDataType _indataNormType;
  std::vector<double> _indataNormParam1;
  std::vector<double> _indataNormParam2;
  std::vector<int> _indataShuffleIndex;
  std::vector<int> _hiddenLayerSizes;
  std::vector<std::vector<double> > _hiddenWeights;
  std::vector<std::vector<double> > _hiddenBiases;
  std::vector<std::vector<double> > _hiddenGradients;
  std::vector<double> _outputWeights;
  std::vector<double> _outputBiases;
  std::vector<double> _outputGradients;
  std::vector<std::vector<double> > _feedForwardValues;
  std::vector<double> _outData;
  size_t _nIndataRecords;
  
  std::vector<double> _indataLabels;
  
  void activateUnits(std::vector<double>& values, std::vector<double>& gradients);
  void activateOutput(std::vector<double>& values, std::vector<double>& gradients);
  // Functions
public:
  nnet();
  ~nnet();
  
// Import Date Methods
  bool loadDataFromFile(char *filename, bool hasHeader, char delim);
  bool loadLabelsFromFile(char *filename, bool hasHeader, char delim);
  
// Set Network Details
  void setHiddenLayerSizes(const std::vector<int>& layerSizes);
  void setActivationType(activationType activationType);
  void setOutputType(outputType outputType);
  void setLossType(lossType lossType);

// Preprocessing methods
  void normIndata(normDataType normType);
  void shuffleIndata();
  
  // Weight fitting stuff
  void initialiseWeights(double stdev);
  void feedForward();
  void backProp(size_t nBatchIndicator,
                double wgtLearnRate,
                double biasLearnRate,
                size_t nEpoch,
                bool doMomentum,
                double mom_mu,
                double mom_decay,
                size_t mom_decay_schedule,
                double mom_final);
  
  
  //Informational queries
  bool dataLoaded();
  bool numericLabels();
  bool classLabels();
  double getCost();
  double getAccuracy();
  
// Print to screen methods
  void printOutValues();
  void printIndata();
  void printLabels();
  void printWeights(int iLayer);
  void printOutputWeights();
  void printGradients(int iLayer);
  void printGeometry();
  void printUnitType();
  void printOutputType();
  void printFeedForwardValues(int iIndex);
  void writeFeedForwardValues();
  
// Write to file methods
  void writeDataToFile(char *filename);
  void writeOutValues();
  void writeWeightValues();
  void writeEpochCostUpdates(std::vector<double> epochCostUpdates);

};

#endif
