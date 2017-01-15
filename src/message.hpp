#ifndef __MOC_message_hpp_
#define __MOC_message_hpp_


namespace msg {
  void info(std::ostringstream& message);
  void warn(std::ostringstream& message);
  void error(std::ostringstream& message);
  void info(const std::string& message);
  void warn(const std::string& message);
  void error(const std::string& message);
//   void info(char *message){ std::cout << message;}
//   void warn(char *message){ std::cout << "! " << message;}
//   void error(char *message){ std::cout << "### " << message;}
};


#endif /* moc_message_hpp */
