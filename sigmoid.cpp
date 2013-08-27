#include <sigmoid>
#include <armadillo>

using namespace arma;
using namespace std;
using namespace neural;

sigmoid::cref_output_type
sigmoid::final_output(sigmoid::cref_full_output_type o)
{
  return o;
}


sigmoid::sigmoid(size_t inputs, size_t outputs)
  : m(randu<mat>(inputs + 1, outputs)),
    learning_rate(0.2)
{}

sigmoid::out_error_type
sigmoid::train(sigmoid::input_type in,
               const sigmoid::output_type& desired)
{
  in.insert_cols(in.size(), vec{1.0});
  return train_ex(in, eval_ex(in), desired);
}

sigmoid::out_error_type
sigmoid::train(sigmoid::input_type in,
               const sigmoid::output_type& out,
               const sigmoid::output_type& desired)
{
  in.insert_cols(in.size(), vec{1.0});
  return train_ex(in, out, desired);
}

sigmoid::out_error_type
sigmoid::train_ex(const sigmoid::input_type& in,
                  const sigmoid::output_type& out,
                  const sigmoid::output_type& desired)
{
  return backpropagate_ex(in, out, (desired - out).t());
}

sigmoid::output_type
sigmoid::eval(sigmoid::input_type in)
{
  in.insert_cols(in.size(), vec{1.0});
  return eval_ex(in);
}

sigmoid::full_output_type
sigmoid::eval_full(sigmoid::input_type in)
{
  return eval(in);
}

sigmoid::output_type
sigmoid::eval_ex(const sigmoid::input_type& in)
{
  output_type out = in * m;
  for (auto& e : out)
    e = function(e);

  return out;
}

// sigmoid::output_type
// sigmoid::eval_full_ex(const sigmoid::input_type& in)
// {
//   auto out = eval_ex(in);
//   out.insert_cols(out.size(), vec{1.0});
//   return out;
// }

size_t sigmoid::inputs()
{
  return m.n_rows - 1;
}

size_t sigmoid::outputs()
{
  return m.n_cols;
}

double sigmoid::function(double x)
{
  // If this was the sigmoid functon,
  return 1 / (1 + exp(-x));
  //return x > 0 ? 1 : 0;
}

double sigmoid::derivative(double y)
{
  // If this was the sigmoid functon,
  return y * (1 - y);
  // return 1;
}

sigmoid::out_error_type
sigmoid::backpropagate(sigmoid::input_type in,
                       const sigmoid::output_type& out,
                       sigmoid::in_error_type err)
{
  in.insert_cols(in.size(), vec{1.0});
  return backpropagate_ex(in, out, err);
}

sigmoid::out_error_type
sigmoid::backpropagate_ex(const sigmoid::input_type& in,
                          const sigmoid::output_type& out,
                          sigmoid::in_error_type err)
{
  for (size_t i = 0; i < err.size(); ++i)
    err[i] *= derivative(out[i]);
    
  vec delta_wh = m * err;
  delta_wh.shed_row(delta_wh.size() - 1);

  m += in.t() * err.t() * learning_rate;

  return delta_wh;
}
