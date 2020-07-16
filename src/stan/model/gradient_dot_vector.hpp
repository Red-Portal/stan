#ifndef STAN_MODEL_GRADIENT_DOT_VECTOR_HPP
#define STAN_MODEL_GRADIENT_DOT_VECTOR_HPP

#include <stan/math/mix.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>
#include <stan/analyze/mcmc/model_profiling.hpp>

namespace stan {
namespace model {

template <class M>
void gradient_dot_vector(const M& model,
                         const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                         const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
                         double& f, double& grad_f_dot_v,
                         std::ostream* msgs = 0) {
  auto scope_measurer = prof::global_profiler.measure_scope_gradlike();
  scope_measurer.start();
  stan::math::gradient_dot_vector(model_functional<M>(model, msgs), x, v, f,
                                  grad_f_dot_v);
}

}  // namespace model
}  // namespace stan
#endif
