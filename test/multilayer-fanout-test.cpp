#include "test-harness.hpp"

#include <vector>
#include <armadillo>

#include <fanout>
#include <multilayer>
#include <sigmoid>

using namespace std;
using namespace arma;
using namespace nnpp;

string charmsg(const vector<size_t>& c) { return string{(char)c[0], (char)c[1]}; }

int safe_main(int argc, char** argv)
{
  vector< pair< vector<size_t>, rowvec > > training_data;

  for (char x = 32; x < 127; ++x) {
    for (char y = 32; y < 127; ++y) {
      vector<size_t> first{(size_t)x, (size_t)y};
      rowvec second{(double)(isalpha(x) & isalpha(y)),(double)(isdigit(x) & isalpha(y))};
      training_data.push_back({first, second});
    }
  }

  multilayer<fanout, sigmoid> L(fanout(128, 2, 2), sigmoid(4, 2));

  for (int x = 0; x < 41; ++x)
    for (auto p : training_data)
      L.train(p.first, p.second);

  mat m;
  for (auto p : training_data) {
    typename multilayer<table, sigmoid>::output_type out = L.eval(p.first);
    typename multilayer<table, sigmoid>::output_type err = abs(round(out) - p.second);

    if (accu(err) > 0)
      notok(charmsg(p.first));
    else
      ok(charmsg(p.first));

    m = join_cols(m, err);
  }

  if (accu(m.col(0)) > 0)
    notok("X and Y are alphas");
  else
    ok("X and Y are alphas");

  if (accu(m.col(1)) > 0)
    notok("X is digit and y is alpha");
  else
    ok("X is digit and y is alpha");

  return 0;
}
