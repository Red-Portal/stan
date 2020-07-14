
#ifndef MODEL_PROFILING_HPP
#define MODEL_PROFILING_HPP

#include <atomic>
#include <chrono>
#include <utility>

namespace prof
{
  using duration = std::chrono::microseconds;

  class ScopeTimer
  {
    using clock = std::chrono::steady_clock;

    std::atomic<uint64_t>& _dest;
    std::atomic<uint64_t>& _counter;
    std::chrono::time_point<clock> _start;

  public:
    inline ScopeTimer(std::atomic<uint64_t>& dest,
		      std::atomic<uint64_t>& counter) noexcept
      : _dest(dest), _counter(counter), _start(clock::now())
    {}

    inline ~ScopeTimer() noexcept
    {
      auto stop  = clock::now();
      auto dur   = stop - _start;
      auto ticks = std::chrono::duration_cast<prof::duration>(dur).count();
      _dest.fetch_add(ticks);
      _counter.fetch_add(1);
    }
  };

  class GlobalProfiler
  {
    using clock = std::chrono::steady_clock;

  private:
    std::chrono::time_point<clock> _global_start;
    std::atomic<uint64_t>          _total_like;
    std::atomic<uint64_t>          _total_gradlike;
    std::atomic<uint64_t>          _num_like_eval;
    std::atomic<uint64_t>          _num_gradlike_eval;

  public:
    inline GlobalProfiler() noexcept
    : _total_like(0),
      _total_gradlike(0),
      _num_like_eval(0),
      _num_gradlike_eval(0)
    {}

    inline void start_measure() noexcept
    {
      _global_start = clock::now();
    }

    inline ScopeTimer
    measure_scope_like() noexcept
    {
      return ScopeTimer(_total_like, _num_like_eval);
    }

    inline ScopeTimer
    measure_scope_gradlike() noexcept
    {
      return ScopeTimer(_total_gradlike, _num_gradlike_eval);
    }

    inline void
    dump_result(uint64_t* like_total,
		uint64_t* grad_total,
		double* like_avg,
		double*   grad_avg,
		uint64_t*   total_time) noexcept
    {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
	_global_start - clock::now());
      *like_total = _total_like.load();
      *grad_total = _total_gradlike.load();
      *like_avg   = static_cast<double>(*like_total) / _num_like_eval.load();
      *grad_avg   = static_cast<double>(*grad_total) / _num_gradlike_eval.load();
      *total_time = duration.count();
    }
  };

  extern GlobalProfiler global_profiler;
}

#endif
