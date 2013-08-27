namespace neural {

  template<class L1, class L2>
  typename multilayer<L1,L2>::cref_output_type
  multilayer<L1,L2>::final_output(multilayer<L1,L2>::cref_full_output_type o)
  {
    return L2::final_output(o.second);
  }


  template<class L1, class L2>
  multilayer<L1,L2>::multilayer(std::size_t inner,
                                std::size_t middle,
                                std::size_t outputs)
    : l1(inner, middle), l2(middle, outputs)
  {}

  template<class L1, class L2>
  typename multilayer<L1,L2>::out_error_type
  multilayer<L1,L2>::train(multilayer<L1,L2>::cref_input_type in,
                           multilayer<L1,L2>::cref_output_type desired)
  {
    return train(in, eval_full(in), desired);
  }

  template<class L1, class L2>
  typename multilayer<L1,L2>::out_error_type
  multilayer<L1,L2>::train(multilayer<L1,L2>::cref_input_type in,
                           multilayer<L1,L2>::cref_full_output_type out,
                           multilayer<L1,L2>::cref_output_type desired)
  {
    auto err = (desired - final_output(out)).t();
    return backpropagate(in, out, err);
  }

  template<class L1, class L2>
  typename multilayer<L1,L2>::output_type
  multilayer<L1,L2>::eval(multilayer<L1,L2>::cref_input_type in)
  {
    auto imed = l1.eval(in);
    return l2.eval(imed);
  }

  template<class L1, class L2>
  typename multilayer<L1,L2>::full_output_type
  multilayer<L1,L2>::eval_full(multilayer<L1,L2>::cref_input_type in)
  {
    auto imed = l1.eval_full(in);
    return { imed, l2.eval_full(imed) };
  }

  template<class L1, class L2>
  std::size_t multilayer<L1,L2>::inputs()
  {
    return l1.inputs();
  }

  template<class L1, class L2>
  std::size_t multilayer<L1,L2>::outputs()
  {
    return l2.outputs();
  }

  template<class L1, class L2>
  typename multilayer<L1,L2>::out_error_type
  multilayer<L1,L2>::backpropagate(multilayer<L1,L2>::cref_input_type in,
                                   multilayer<L1,L2>::cref_full_output_type out,
                                   multilayer<L1,L2>::cref_in_error_type err)
  {
    auto imed_err = l2.backpropagate(out.first, out.second, err);
    return l1.backpropagate(in, out.first, imed_err);
  }

}
