#pragma once

#include <condition_variable>
#include <mutex>
#include <random>
#include <queue>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>

#include "game.hpp"

namespace needletail {
  template<class T, class U = typename T::value_type>
  inline T boltzman(const T& xs, float temperature) noexcept {
    if (temperature == 0) {
      return boost::copy_range<T>(
        boost::irange(static_cast<size_t>(0), xs.size()) |
        boost::adaptors::transformed(std::function([index = std::distance(std::begin(xs), boost::max_element(xs))](const float& i) { return i == index ? 1.0 : 0.0; })));
    }

    auto result = boost::copy_range<T>(
      xs |
      boost::adaptors::transformed(std::function([&](const U& x) { return pow(x, 1.0F / temperature); })));

    boost::transform(result, std::begin(result), [total = boost::accumulate(result, 0.0F)](const U& x) { return x / total; });

    return result;
  }

  template<class T, class U, class V = typename T::const_iterator>
  inline V choice(const T& xs, const U& probabilities) noexcept {
    auto random_engine       = std::default_random_engine(std::random_device()());
    auto random_distribution = std::uniform_real_distribution<float>(0, 1);

    const auto& threshold = random_distribution(random_engine);
    auto total = 0.0f;

    for (auto it_1 = std::begin(xs), it_2 = std::begin(probabilities); it_1 != std::end(xs); ++it_1, ++it_2) {
      if (total < threshold && threshold <= total + *it_2) {
        return it_1;
      }

      total += *it_2;
    }

    return --std::end(xs);
  }

  template<class T>
  class blocking_queue {
    std::queue<T>           _queue;
    std::mutex              _mutex;
    std::condition_variable _condition;

  public:
    blocking_queue() noexcept: _queue(), _mutex(), _condition() {
      ;
    }

    auto enqueue(const T& x) noexcept {
      auto lock = std::unique_lock(_mutex);

      const auto& need_to_notify = std::empty(_queue);

      _queue.emplace(x);

      if (need_to_notify) {
        _condition.notify_one();
      }
    }

    auto dequeue() noexcept {
      auto lock = std::unique_lock(_mutex);

      if (std::empty(_queue)) {
        _condition.wait(lock);
      }

      auto result = _queue.front();

      _queue.pop();

      return result;
    }
  };
}
