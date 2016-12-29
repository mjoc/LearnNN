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
  
  enum initialiseType
  {
    INIT_CONST_TYPE,
    INIT_GAUSS_TYPE,
    INIT_UNIFORM_TYPE
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
  
  // Things to do with non- Train Data
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
  
  // Dropout during optimisation
  bool _doTestCost;
  bool _doDropout;
  double _inputDropoutRate;
  double _hiddenDropoutRate;
  bool _doMomentum;
  double _momMu;
  double _momDecay;
  size_t _momDecaySchedule;
  double _momFinal;
  std::vector<std::vector<int> > _hiddenWeightsDropout;
  std::vector<std::vector<int> > _hiddenBiasesDropout;
  std::vector<double> _outputWeightsDropout;
  std::vector<double> _outputBiasesDropout;
  
  std::vector<double> _epochTrainCostUpdates;
  std::vector<double> _epochTestCostUpdates;

  // Things to do with the output
  size_t _nOutputUnits;
  std::vector<double> _trainGeneratedLabels;
  bool _trainLabelsGenerated;
  outputType _outputType;
  lossType _lossType;
 
  
  // Private functions
  void activateUnits(std::vector<double>& values);
  void activateUnitsAndCalcGradients(std::vector<double>& values,
                                     std::vector<double>& gradients,
                                     double nesterovAdj = 0.0);
  void activateOutput(std::vector<double>& values);
  void flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients,
                              double nesterovAdj);
  void flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients,
                              double nesterovAdj,
                              std::vector<std::vector<int> >& dropouts);
  // Functions
public:
  nnet();
  ~nnet();
  
  void setOutputFolder(char *filename);
  
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
  void pcaTrainData(size_t dimensions = 0);
  
 
  // Preprocessing methods on Non Train Data

  void normNonTrainData(normDataType normType);
  void pcaNonTrainData();

  // Feed forward
  void feedForwardTrainData(bool calcGradients = false,
                            double nestorovAdj = 0.0);
  void feedForwardTrainData(bool calcGradients,
                            double nestorovAdj,
                            std::vector<std::vector<int> >& dropouts);

  void feedForwardNonTrainData();

  // Backpropagation stuff
  void doDropout(bool doDropout);
  void setDropoutRates(double inputDropout, double hiddenDropout);
  void doTestCost(bool doTestCost);
  void setMomentum(bool doMomentum,
                   double momMu,
                   double momDecay,
                   size_t momDecaySchedule,
                   double momFinal);

  void initialiseWeights(initialiseType initialtype = INIT_CONST_TYPE, double param1 = 0.0);
  void setDropout(bool doDropout);
  bool backProp(size_t nBatchIndicator,
                double wgtLearnRate,
                double biasLearnRate,
                size_t nEpoch);
  
  
  
  //Informational queries
  bool dataLoaded();
  bool numericLabels();
  bool classLabels();
  double getTrainDataCost();
  double getTrainDataAccuracy();
  double getNonTrainDataCost();
  double getNonTrainDataAccuracy();
  size_t getNTrainData();
  
  
// Print to screen methods
  void printOutputUnitValues(size_t nRecords = 0);
  void printTrainData(size_t nRecords = 0);
  void printNonTrainData(size_t nRecords = 0);
  void printTrainLabels(size_t nRecords = 0);
  void printNonTrainLabels(size_t nRecords = 0);
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
