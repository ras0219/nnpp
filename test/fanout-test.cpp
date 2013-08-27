#include "test-harness.hpp"

#include <vector>
#include <armadillo>

#include <table>
#include <fanout>

using namespace std;
using namespace arma;
using namespace nnpp;

string makemsg(const vector<size_t>& c) {
  return string{(char)c[0], (char)c[1]};
}

int safe_main(int argc, char** argv)
{
  std::vector< std::pair<vector<size_t>, rowvec> > training_data;

  for (char x = 32; x < 127; ++x) {
    for (char y = 32; y < 127; ++y) {
      vector<size_t> first{(size_t)x, (size_t)y};
      rowvec second{(double)(isalpha(x)), (double)(isalpha(y))};
      training_data.push_back({first, second});
    }
  }

  fanout L(128, 2, 1);

  for (int x = 0; x < 41; ++x)
    for (auto p : training_data)
      L.train(p.first, p.second);

  mat m;
  for (auto p : training_data) {
    fanout::output_type out = L.eval(p.first);
    fanout::output_type err = abs(round(out) - p.second);

    if (accu(err) > 0)
      notok(makemsg(p.first));
    else
      ok(makemsg(p.first));

    m = join_cols(m, err);
  }

  if (accu(m.col(0)) > 0)
    notok("first isalpha");
  else
    ok("first isalpha");

  if (accu(m.col(1)) > 0)
    notok("second isalpha");
  else
    ok("second isalpha");
    
  return 0;
}
