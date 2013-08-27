#ifndef _NEURAL_TEST_HPP_
#define _NEURAL_TEST_HPP_

#include <string>

// Create a test with message <msg> and mark it ok
void ok(const std::string& msg);

// Create a test with message <msg> and mark it notok
void notok(const std::string& msg);

// YOU need to define this, somewhere.
int safe_main(int argc, char** argv);

#endif
