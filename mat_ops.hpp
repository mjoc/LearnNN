//
//  mat_ops.hpp
//  BasicNN
//
//  Created by Martin on 20/11/2016.
//  Copyright © 2016 Martin. All rights reserved.
//

#ifndef mat_ops_hpp
#define mat_ops_hpp

class mat_ops{
public:
  static void writeMatrix(std::string filePathAndName, std::vector<double> outMat, size_t nRows, size_t nCols);
  static void matMul(size_t Arows, size_t Acols, std::vector<double>& A, size_t Bcols, std::vector<double>& B, std::vector<double>& C);
};

#endif /* mat_ops_hpp */
