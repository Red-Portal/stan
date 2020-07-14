
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
    std::chrono::time_point<clock> _start;

  public:
    inline ScopeTimer(std::atomic<uint64_t>& dest) noexcept
      : _dest(dest), _start(clock::now())
    {}

    inline ~ScopeTimer() noexcept
    {
      auto stop  = clock::now();
      auto dur   = stop - _start;
      auto ticks = std::chrono::duration_cast<prof::duration>(dur).count();
      _dest.fetch_add(ticks);
    }
  };

  class GlobalProfiler
  {
    using clock = std::chrono::steady_clock;

  private:
    std::chrono::time_point<clock> _global_start;
    std::atomic<uint64_t>          _total_like;
    std::atomic<uint64_t>          _total_gradlike;

  public:
    inline GlobalProfiler() noexcept
      : _total_like(0), _total_gradlike(0)
    {}

    inline ScopeTimer
    measure_scope_like() noexcept
    {
      return ScopeTimer(_total_like);
    }

    inline ScopeTimer
    measure_scope_gradlike() noexcept
    {
      return ScopeTimer(_total_gradlike);
    }

    inline std::tuple<uint64_t, uint64_t, uint64_t>
    dump_result() noexcept
    {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
	_global_start - clock::now()).count();
      return {_total_like.load(), _total_gradlike.load(), duration};
    }
  };

  extern GlobalProfiler global_profiler;
}
