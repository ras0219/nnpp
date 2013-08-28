#include <fanout>
#include <armadillo>
#include <cassert>

using namespace nnpp;
using namespace arma;
using namespace std;

fanout::fanout(size_t categories, size_t inputs, size_t table_outputs)
  : t(categories, table_outputs), insize(inputs), outsize(inputs * table_outputs)
{}

fanout::out_error_type
fanout::train(fanout::cref_input_type in,
              fanout::cref_output_type desired)
{
  return train(in, eval_full(in), desired);
}

fanout::out_error_type
fanout::train(fanout::cref_input_type in,
              fanout::full_output_type out,
              fanout::cref_output_type desired)
{
  return backpropagate(in, out, (desired - out).t());
}

fanout::output_type
fanout::eval(fanout::cref_input_type in)
{
  rowvec out;
  for (auto n : in)
    out = join_rows(out, t.eval(n));

  return out;
}

fanout::full_output_type
fanout::eval_full(fanout::cref_input_type in)
{
  return eval(in);
}

size_t fanout::inputs()
{
  return insize;
}

size_t fanout::categories()
{
  return t.categories();
}

size_t fanout::outputs()
{
  return outsize;
}

fanout::out_error_type
fanout::backpropagate(fanout::cref_input_type in,
                      fanout::full_output_type out,
                      fanout::in_error_type err)
{
  out.reshape(insize, t.outputs());
  err.reshape(t.outputs(), insize);

  assert(in.size() == out.n_rows);
  for (size_t i = 0; i < in.size(); ++i)
    t.backpropagate(in[i], out.row(i), err.col(i));

  return mat();
}
