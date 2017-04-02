//
//  backprop.hpp
//  BasicNN
//
//  Created by Martin on 30/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef __MOC_BACKPROP_HPP_
#define __MOC_BACKPROP_HPP_

#include <string>
#include "dataset.hpp"
#include "nnet.hpp"
#include "rng.hpp"

#ifndef IGNORE_THIS_RCPP_CODE

class Nnet; // fwd declarations
RCPP_EXPOSED_CLASS(Nnet);

#endif


class Backpropper{
public:
  enum initialiseType
  {
    INIT_CONST_TYPE,
    INIT_GAUSS_TYPE,
    INIT_UNIFORM_TYPE
  };
  enum lossType
  {
    MSE_LOSS_TYPE,
    CROSS_ENT_TYPE
  };

private:
  Nnet *_net;
  rng *_rng;
  std::string _outputDir;
  size_t _epochPrintSchedule;

  bool _dataShuffled;
  std::vector<int> _dataShuffleIndex;

  std::vector<std::vector<double> > _hiddenGradients;

  // Batch Size Indicator
  int _nBatchSizeIndicator;
  double _wgtLearnRate;
  double _biasLearnRate;
  // Number of epochs
  size_t _nEpoch;

  lossType _lossType;


  bool _doTestCost;
  Dataset *_testdata;
  std::vector<double>* _testInputData;
  std::vector<std::vector<double> > _testdataFeedForwardValues;
  std::vector<double> _testdataGeneratedValues;

  // During optimisation
  bool _trainDataShuffled;
  bool _shuffleEachEpoch;
  bool _testdataLoaded;

  bool _doMomentum;
  double _momMu;
  double _momDecay;
  size_t _momDecaySchedule;
  double _momFinal;

  std::vector<double> _epochTrainCostUpdates;
  std::vector<double> _epochTestCostUpdates;

  //std::vector<double> _trainGeneratedLabels;

  void shuffleTrainData();

  void activateUnitsAndCalcGradients(std::vector<double>& values,
                                     std::vector<double>& gradients,
                                     double nesterovAdj = 0.0);
  void flowDataThroughNetwork(std::vector<double>& inputData,
                              std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients,
                              double nesterovAdj);

public:
  Backpropper(Nnet &net);
  Backpropper(Nnet &net, std::string lossType);
  ~Backpropper();

  void setOutputFolder(char *filename);
  void setEpochPrintSchedule(size_t schedule);

  void setLossType(std::string lossType);
  std::string getLossType();

  bool setTestData(Dataset& testdata);
  void doTestCost(bool doTestCost = true);
  // Preprocessing methods on Train Data

  //double getTrainDataCost();
  //double getTrainDataAccuracy();



  void setShuffleData(bool doShuffle = true);

  // Backpropagation stuff
  void setMomentum(bool doMomentum,
                   double momMu,
                   double momDecay,
                   size_t momDecaySchedule,
                   double momFinal);


  void initialiseWeights(std::string initialtype = "const", double param1 = 0.0);

  // Feed forward
  void feedForwardTrainData(bool calcGradients = false,
                            double nestorovAdj = 0.0);

  void doBackPropOptimise(size_t nBatchSize,
                          double wgtLearnRate,
                          double biasLearnRate,
                          size_t nEpoch);
  bool fitWeights(Nnet netToFit);

  double calcAccuracy(bool testdata = false);
  // std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits
  double calcCost(bool testdata = false);
  // std::vector<double> actual, std::vector<double> fitted ,size_t nRecords, size_t nOutputUnits

  void printGradients(int iLayer);
  void printTrainData(size_t nRecords = 0);
  void printTrainLabels(size_t nRecords = 0);
  // Write to file methods
  void writeEpochCostUpdates();
  void writeEpochTrainCostUpdates();
  void writeEpochTestCostUpdates();

  // Rcpp stuff
#ifndef IGNORE_THIS_RCPP_CODE
  SEXP calcCostR();
#endif

};

#endif /* __MOC_BACKPROP_HPP_ */
