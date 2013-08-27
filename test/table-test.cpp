#include "test-harness.hpp"

#include <vector>
#include <armadillo>

#include <table>

using namespace std;
using namespace arma;
using namespace nnpp;

string charmsg(char c) { return string(1, c); }

int safe_main(int argc, char** argv)
{
  std::vector< std::pair<size_t, rowvec> > training_data;

  const size_t features = 2;

  for (char x = 32; x < 127; ++x) {
    training_data.push_back({x, {(double)isdigit(x), (double)isalnum(x)}});
  }

  table L(128, features);

  for (int x = 0; x < 2001; ++x) {
    double passing = 0.0;

    for (auto p : training_data) {
      auto err = L.train(p.first, p.second);
      passing += accu(abs(err)) / err.size();
    }

    passing /= training_data.size();
  }

  cerr << "FINISHED TRAINING" << endl;

  mat m;
  
  for (auto p : training_data) {
    table::output_type out = L.eval(p.first);
    table::output_type err = abs(round(out) - p.second);

    if (accu(err) > 0)
      notok(charmsg(p.first));
    else
      ok(charmsg(p.first));

    cerr << "tests '" << (char)p.first << "'" << endl;

    m = join_cols(m, err);
  }

  if (accu(m.col(0)) > 0)
    notok("isdigit");
  else
    ok("isdigit");

  if (accu(m.col(1)) > 0)
    notok("isalnum");
  else
    ok("isalnum");
    
  return 0;
}
