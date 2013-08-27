#include "test-harness.hpp"

#include <vector>
#include <armadillo>

#include <multilayer>
#include <sigmoid>

using namespace std;
using namespace arma;
using namespace neural;

int safe_main(int argc, char** argv)
{
  std::vector< std::pair<rowvec, rowvec> > training_data =
    {
      // {{0, 1, 1}, {1, 1, 0, 0}},
      // {{0, 1, 0}, {0, 1, 1, 1}},
      // {{0, 0, 1}, {0, 1, 1, 1}},
      // {{0, 0, 0}, {0, 0, 0, 1}},
      {{1, 1, 1}, {1, 1, 0, 0}},
      {{1, 1, 0}, {0, 1, 1, 1}},
      {{1, 0, 1}, {0, 1, 1, 1}},
      {{1, 0, 0}, {0, 0, 0, 1}}
    };

  multilayer<sigmoid, sigmoid> L(3, 4, 4);

  for (int x = 0; x < 3000; ++x) {
    double passing = 0.0;

    for (auto p : training_data) {
      auto err = L.train(p.first, p.second);
      passing += accu(abs(err)) / err.size();
    }

    passing /= training_data.size();
  }

  mat m;
  
  for (auto p : training_data) {
    typename multilayer<sigmoid, sigmoid>::output_type out = L.eval(p.first);

    m = join_cols(m, abs(round(out) - p.second));
  }

  if (accu(m.col(0)) > 0)
    notok("AND");
  else
    ok("AND");

  if (accu(m.col(1)) > 0)
    notok("OR");
  else
    ok("OR");

  if (accu(m.col(2)) > 0)
    notok("XOR");
  else
    ok("XOR");

  if (accu(m.col(3)) > 0)
    notok("NAND");
  else
    ok("NAND");

  return 0;
}
