#pragma once

#include <boost/container/static_vector.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include "game.hpp"

namespace needletail {
  boost::container::static_vector<float, MAX_POINT_SIZE> one_hot(size_t size, size_t v) noexcept {
    auto result = boost::container::static_vector<float, MAX_POINT_SIZE>(size);

    result[v] = 1.0;

    return result;
  }

  boost::container::static_vector<float, MAX_POINT_SIZE> boltzman(const boost::container::static_vector<float, MAX_POINT_SIZE>& xs, float temperature) noexcept {
    if (temperature == 0) {
      return one_hot(xs.size(), std::distance(std::begin(xs), boost::max_element(xs)));
    }

    auto xs_1 = [&]() {
      auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

      boost::copy(
        xs |
        boost::adaptors::transformed([&](const auto& x) { return pow(x, 1.0F / temperature); }),
        std::back_inserter(result));

      return result;
    }();

    return [&]() {
      auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

      const auto& total = boost::accumulate(xs_1, 0.0F);

      boost::copy(
        xs_1 |
        boost::adaptors::transformed([&](const auto& x) { return x / total; }),
        std::back_inserter(result));

      return result;
    }();
  }

  template<typename T>
  T::const_iterator choice(const T& xs, const boost::conatiner::static_vector<float, MAX_POINT_SIZE>& p) noexcept {

  }
}
