#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

#include <armadillo>

struct perceptron_layer {
  perceptron_layer(std::size_t inputs, std::size_t outputs)
    : m(arma::randu<arma::mat>(inputs + 1, outputs)), learning_rate(0.1) {}

  // Each column is a perceptron

  arma::vec train(arma::rowvec in,
                  const arma::rowvec& desired) {

    in.insert_cols(in.size(), arma::vec{1.0});
    return train_ex(in, eval_ex(in), desired);
  }

  arma::vec train(arma::rowvec in,
                  const arma::rowvec& out,
                  const arma::rowvec& desired) {
    in.insert_cols(in.size(), arma::vec{1.0});
    return train_ex(in, out, desired);
  }

  arma::vec train_ex(const arma::rowvec& in,
                     const arma::rowvec& out,
                     const arma::rowvec& desired) {
    return backpropagate(in, out, (desired - out).t());
  }

  arma::rowvec eval(arma::rowvec in) {
    in.insert_cols(in.size(), arma::vec{1.0});
    return eval_ex(in);
  }

  arma::rowvec eval_ex(const arma::rowvec& in) {
    arma::rowvec out = in * m;
    for (auto& e : out)
      e = function(e);

    return out;
  }

  arma::rowvec eval_full_ex(const arma::rowvec& in) {
    auto out = eval_ex(in);
    out.insert_cols(out.size(), arma::vec{1.0});
    return out;
  }

  static double function(double x) {
    // If this was the sigmoid functon,
    return 1 / (1 + exp(-x));
    //return x > 0 ? 1 : 0;
  }

  static double derivative(double y) {
    // If this was the sigmoid functon,
    return y * (1 - y);
    // return 1;
  }

  arma::vec backpropagate(const arma::rowvec& in,
                          const arma::rowvec& out,
                          arma::vec err) {
    for (std::size_t i = 0; i < err.size(); ++i)
      err[i] *= derivative(out[i]);
    
    arma::vec delta_wh = m * err;
    delta_wh.shed_row(delta_wh.size() - 1);

    m += in.t() * err.t() * learning_rate;

    return delta_wh;
  }

  arma::mat m;
  double learning_rate;
};

struct multilayer {
  multilayer(std::size_t inputs, std::size_t hidden, std::size_t outputs)
    : outer(hidden, outputs), inner(inputs, hidden)
    { }

  arma::vec train(arma::rowvec in,
                  const arma::rowvec& desired) {

    in.insert_cols(in.size(), arma::vec{1.0});
    return train_ex(in, desired);
  }

  arma::vec train_ex(const arma::rowvec& in,
                     const arma::rowvec& desired) {
    // Eval, but save the info
    auto hidden = inner.eval_ex(in);
    hidden.insert_cols(hidden.size(), arma::vec{1.0});
    auto out = outer.eval_ex(hidden);
    
    // Now do backpropagation
    auto outer_err = (desired - out).t();
    auto inner_err = outer.backpropagate(hidden, out, outer_err);
    auto err = inner.backpropagate(in, hidden, inner_err);
    return err;
  }

  arma::rowvec eval(arma::rowvec in) {
    in.insert_cols(in.size(), arma::vec{1.0});
    return eval_ex(in);
  }

  arma::rowvec eval_ex(const arma::rowvec& in) {
    auto hidden = inner.eval_ex(in);
    hidden.insert_cols(hidden.size(), arma::vec{1.0});
    return outer.eval_ex(hidden);
  }
  
  std::pair<arma::rowvec, arma::rowvec>
  eval_full(arma::rowvec in) {
    in.insert_cols(in.size(), arma::vec{1.0});
    return eval_full_ex(in);
  }

  std::pair<arma::rowvec, arma::rowvec>
  eval_full_ex(const arma::rowvec& in) {
    auto hidden = inner.eval_full_ex(in);
    auto out = outer.eval_full_ex(hidden);
    return {hidden, out};
  }

  perceptron_layer outer;
  perceptron_layer inner;
};

struct lut_layer {
  lut_layer(std::size_t categories, std::size_t outputs)
    : m(arma::randu<arma::mat>(categories, outputs)), learning_rate(0.1) {}

  arma::vec train(std::size_t in,
                  const arma::rowvec& desired) {
    return train(in, eval(in), desired);
  }

  arma::vec train(std::size_t in,
                  const arma::rowvec& out,
                  const arma::rowvec& desired) {
    return backpropagate(in, out, (desired - out).t());
  }

  arma::rowvec eval(std::size_t in) {
    arma::rowvec out = m.row(in);
    for (auto& e : out)
      e = function(e);

    return out;
  }

  static double function(double x) {
    // If this was the sigmoid functon,
    return 1 / (1 + exp(-x));
    //return x > 0 ? 1 : 0;
  }

  static double derivative(double y) {
    // If this was the sigmoid functon,
    return y * (1 - y);
    // return 1;
  }

  arma::vec backpropagate(std::size_t in,
                          const arma::rowvec& out,
                          const arma::vec& err) {

    auto err2 = err;
    for (std::size_t i = 0; i < err2.size(); ++i)
      err2[i] *= derivative(out[i]);
    
    m.row(in) += err2.t() * learning_rate;

    return err;
  }

  arma::mat m;
  double learning_rate;
};

// struct multilut {
//   multilut(std::size_t categories, std::size_t outputs)
//     : ll(categories, outputs) {}

//   arma::vec train(const std::vector<std::size_t>& in,
//                   const arma::rowvec& desired) {
//     auto desired2 = desired.reshape(in.size(), desired.size() / in.size(), 1);
//     arma::vec r;
//     for (std::size_t i = 0; i < in.size(); ++i) {
//       r = arma::join_cols(r, ll.train(in[i], desired2[i]));
//     }
//     return r;
//   }

//   arma::vec train(const std::vector<std::size_t>& in,
//                   const arma::rowvec& out,
//                   const arma::rowvec& desired) {
//     auto desired2 = desired.reshape(in.size(), desired.size() / in.size(), 1);
//     auto out2 = out.reshape(in.size(), out.size() / in.size(), 1);

//     arma::vec r;
//     for (std::size_t i = 0; i < in.size(); ++i) {
//       r = arma::join_cols(r, ll.train(in[i], out2[i], desired2[i]));
//     }
//     return r;
//   }

//   arma::rowvec eval(const std::vector<std::size_t>&in) {
//     arma::rowvec out = m.row(in);
//     for (auto& e : out)
//       e = function(e);

//     return out;
//   }

//   static double function(double x) {
//     // If this was the sigmoid functon,
//     return 1 / (1 + exp(-x));
//     //return x > 0 ? 1 : 0;
//   }

//   static double derivative(double y) {
//     // If this was the sigmoid functon,
//     return y * (1 - y);
//     // return 1;
//   }

//   arma::vec backpropagate(const std::vector<std::size_t>& in,
//                           const arma::rowvec& out,
//                           const arma::vec& err) {
//     auto out2 = out.reshape(in.size(), out.size() / in.size(), 1);

//     auto err2 = err;
//     for (std::size_t i = 0; i < err2.size(); ++i)
//       err2[i] *= derivative(out[i]);
    
//     m.row(in) += err2.t() * learning_rate;

//     return err;
//   }

//   lut_layer ll;
// };

#endif
