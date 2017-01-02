//
//  backprop.hpp
//  BasicNN
//
//  Created by Martin on 30/12/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef backprop_hpp
#define backprop_hpp

#include <string>
#include "weightfitter.hpp"
#include "dataset.hpp"
#include "nnet.hpp"
#include "rng.hpp"

class backpropper{
public:
  enum initialiseType
  {
    INIT_CONST_TYPE,
    INIT_GAUSS_TYPE,
    INIT_UNIFORM_TYPE
  };

private:
  nnet *_net;
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
  
  bool _doTestCost;
  dataset *_testdata;
  std::vector<std::vector<double> > _testdataFeedForwardValues;
  std::vector<double> _testdataGeneratedValues;
  
  // During optimisation
  bool _trainDataShuffled;
  bool _shuffleEachEpoch;
  bool _testdataLoaded;
  
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

  //std::vector<double> _trainGeneratedLabels;
  
  void shuffleTrainData();
  
  void activateUnitsAndCalcGradients(std::vector<double>& values,
                                     std::vector<double>& gradients,
                                     double nesterovAdj = 0.0);
  void flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients,
                              double nesterovAdj);

  void flowDataThroughNetwork(std::vector<std::vector<double> >& dataflowStages,
                              std::vector<double>& dataFlowMatrix,
                              bool calcTrainGradients,
                              double nesterovAdj,
                              std::vector<std::vector<int> >& dropouts);
  
public:
  backpropper(nnet *net);
  ~backpropper();

  void setOutputFolder(char *filename);
  void setEpochPrintSchedule(size_t schedule);
  
  bool setTestData(dataset *testdata);
  void doTestCost(bool doTestCost = true);
  // Preprocessing methods on Train Data
  
  //double getTrainDataCost();
  //double getTrainDataAccuracy();

  
  
  void setShuffleData(bool doShuffle = true);
  
  // Backpropagation stuff
  void doDropout(bool doDropout);
  void setDropoutRates(double inputDropout, double hiddenDropout);
  void setMomentum(bool doMomentum,
                   double momMu,
                   double momDecay,
                   size_t momDecaySchedule,
                   double momFinal);
  
  
  void initialiseWeights(initialiseType initialtype = INIT_CONST_TYPE, double param1 = 0.0);
  void setDropout(bool doDropout);
  
  // Feed forward
  void feedForwardTrainData(bool calcGradients = false,
                            double nestorovAdj = 0.0);
  void feedForwardTrainData(bool calcGradients,
                            double nestorovAdj,
                            std::vector<std::vector<int> >& dropouts);

  
  
  bool doBackPropOptimise(size_t nBatchIndicator,
                          double wgtLearnRate,
                          double biasLearnRate,
                          size_t nEpoch);
  bool fitWeights(nnet netToFit);
  
  void printGradients(int iLayer);
  void printTrainData(size_t nRecords = 0);
  void printTrainLabels(size_t nRecords = 0);
  // Write to file methods
  void writeEpochCostUpdates();
  void writeEpochTrainCostUpdates();
  void writeEpochTestCostUpdates();

};




#endif /* backprop_hpp */
