
#include <iostream>
#include <sstream>
//#include <fstream>

#include "message.hpp"

namespace msg {
void info(std::ostringstream& message){ std::cout << message.str(); message.str(std::string());}
void warn(std::ostringstream& message){ std::cout << "Warning: " << message.str(); message.str(std::string());}
void error(std::ostringstream& message){ std::cout << "Error: " << message.str(); message.str(std::string());}
void info(const std::string& message){ std::cout << message;}
void warn(const std::string& message){ std::cout << "Warning: " << message;}
void error(const std::string& message){ std::cout << "Error: " << message;}
};
