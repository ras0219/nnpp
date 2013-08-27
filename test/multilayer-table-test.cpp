#include "test-harness.hpp"

#include <vector>
#include <armadillo>

#include <table>
#include <multilayer>
#include <sigmoid>

using namespace std;
using namespace arma;
using namespace nnpp;

string charmsg(char c) { return string(1, c); }

int safe_main(int argc, char** argv)
{
  std::vector< std::pair<size_t, rowvec> > training_data;

  for (char x = 32; x < 127; ++x) {
    training_data.push_back({x,
        {   (double)isdigit(x),
            (double)isalnum(x),
            (double)isalpha(x) }});
  }

  multilayer<table, sigmoid> L(255, 2, 3);

  for (int x = 0; x < 2001; ++x)
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
    notok("isdigit");
  else
    ok("isdigit");

  if (accu(m.col(1)) > 0)
    notok("isalnum");
  else
    ok("isalnum");

  if (accu(m.col(2)) > 0)
    notok("isalpha");
  else
    ok("isalpha");

  return 0;
}
