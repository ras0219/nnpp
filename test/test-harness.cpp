#include "test-harness.hpp"
#include <iostream>
#include <sstream>

using namespace std;

stringstream testresults;
size_t counter = 0;
bool failed_test = false;

void ok(const string& msg) {
  testresults << "ok " << ++counter << " - " << msg << endl;
}
void notok(const string& msg) {
  testresults << "not ok " << ++counter << " - " << msg << endl;
  failed_test = true;
}
int flushtests(ostream& os = cout) {
  os << "1.." << counter << endl;
  os << testresults.str();

  return failed_test ? -1 : 0;
}


int main(int argc, char** argv)
{
  try {
    int rcode = safe_main(argc, argv);
    if (rcode)
      notok("safe_main Return Code Nonzero");
  } catch (std::exception& e) {
    notok(e.what());
  } catch (...) {
    notok("Oh gosh... something *really bad* went wrong.");
  }

  return flushtests();
}
