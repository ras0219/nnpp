#include <table>
#include <armadillo>
#include <cassert>

using namespace nnpp;
using namespace arma;
using namespace std;

table::table(size_t categories, size_t outputs)
  : m(randu<mat>(categories, outputs)),
    learning_rate(0.1)
{}

table::out_error_type
table::train(table::cref_input_type in,
             table::cref_output_type desired)
{
  return train(in, eval(in), desired);
}

table::out_error_type
table::train(table::cref_input_type in,
             table::cref_output_type out,
             table::cref_output_type desired)
{
  return backpropagate(in, out, (desired - out).t());
}

table::output_type
table::eval(table::cref_input_type in)
{
  output_type out = m.row(in);
  for (auto& e : out)
    e = function(e);

  return out;
}

table::full_output_type
table::eval_full(table::cref_input_type in)
{
  return eval(in);
}

size_t table::categories()
{
  return m.n_rows;
}

size_t table::outputs()
{
  return m.n_cols;
}

double table::function(double x)
{
  // If this was the table functon,
  return 1 / (1 + exp(-x));
  //return x > 0 ? 1 : 0;
}

double table::derivative(double y)
{
  // If this was the table functon,
  return y * (1 - y);
  // return 1;
}

table::out_error_type
table::backpropagate(table::cref_input_type in,
                     table::cref_output_type out,
                     table::in_error_type err)
{
  for (size_t i = 0; i < err.size(); ++i)
    err[i] *= derivative(out[i]);

  assert(in < m.n_rows);
  m.row(in) += err.t() * learning_rate;

  return mat();
}
