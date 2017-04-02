#ifndef __MOC_Nnet_
#define __MOC_Nnet_

#include "dataset.hpp"
#include <vector>

class Nnet{
  friend class Backpropper;
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


  // Things to do with loaded Data
  bool _dataLabelsLoaded;
  bool _dataLoaded;
  size_t _nDataRecords;
  bool _doPca;
  size_t _nPcaDimensions;
  //size_t _nNonTrainDataInputUnits;
  //size_t _nNonTrainDataOutputUnits;
  std::vector<double>* _inputData;
  std::vector<std::vector<double> > _feedForwardValues;
  std::vector<double>* _dataLabels;
  bool _labelsGenerated;
  std::vector<double> _generatedLabels;

  // Private functions
  void activateUnits(std::vector<double>& values);
  void activateOutput(std::vector<double>& values);

  // Functions
  void flowDataThroughNetwork(std::vector<double>* inputDataMatrix,
                              std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataflowMatrix);
  // Wrting out to file
  std::string _outputDir;

public:
  Nnet();
  //Nnet(Dataset initialData);
  Nnet(std::vector<int> networkgeometry,
       std::string hiddenUnitActivation = std::string("linear"),
       std::string outputUnitActivation = std::string("linear"));

  ~Nnet();



// Import Data Methods
  bool loadDataFromFile(char *filename,
                        bool hasHeader,
                        char delim);
  bool loadDataLabelsFromFile(char *filename,
                              bool hasHeader,
                              char delim);
  bool loadDataLabelsFromFile(char *dataFileName,
                              char *labelsFileName,
                              bool hasHeader,
                              char delim);

  bool clampData(Dataset& dataset);

  bool dataLoaded();
  bool dataAndLabelsLoaded();
  size_t nDataRecords();

  // Set Network Details
  void setInputSize(size_t nInputUnits);
  size_t getInputSize();
  void setOutputSize(size_t nOutputUnits);
  size_t getOutputSize();
  void setHiddenLayerSizes(std::vector<int> layerSizes);
  std::vector<int>  getHiddenLayerSizes();
  void setActivationType(std::string activationType);
  std::string getActivationType();
  void setOutputType(std::string outputType);
  std::string getOutputType();

  bool setDataAndLabels(Dataset dataToClamp);

  void initialiseWeights();
  bool setWgtAndBias(int iIndex, std::vector<double> wgts, std::vector<double> bias);

  void feedForward();


  //Informational queries
  bool numericLabels();
  bool classLabels();
  bool labelsGenerated();

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

  // R package stuff
  #ifndef IGNORE_THIS_RCPP_CODE
    Nnet(Rcpp::IntegerVector networkgeometry,
       Rcpp::String hiddenUnitActivation,
       Rcpp::String outputUnitActivation);

  SEXP generatedLabelsR() const;
  SEXP getWgtAndBiasR(int Index) const;
  Rcpp::LogicalVector setWgtAndBiasR(int iIndex, Rcpp::NumericMatrix weights, Rcpp::NumericVector bias);
  SEXP getFFValues(int iIndex) const;
  #endif
};

#endif
