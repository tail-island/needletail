#pragma once

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include "game.hpp"
#include "util.hpp"

namespace needletail {
  class greedy_play final {
    game  _game;
    float _temperature;

    auto p(needletail::state state, boost::container::static_vector<int, MAX_POINT_SIZE> legal_actions) const noexcept {
      const auto& scores_1 = [&]() {
        auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

        boost::copy(
          legal_actions |
          boost::adaptors::transformed([&](const auto& a) { return distance(_game.points()[state.route().back()], _game.points()[a]); }),
          std::back_inserter(result));

        return result;
      }();

      const auto& scores_2 = [&]() {
        auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

        const auto& total = boost::accumulate(scores_1, 0.0F);

        boost::copy(
          scores_1 |
          boost::adaptors::transformed([&](const auto& s) { return s / total; }),
          std::back_inserter(result));

        return result;
      }();

      const auto& scores_3 = [&]() {
        auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

        const auto& max = *boost::max_element(scores_2);

        boost::copy(
          scores_2 |
          boost::adaptors::transformed([&](const auto& s) { return max - s + 0.01; }),
          std::back_inserter(result));

        return result;
      }();

      return [&]() {
        auto result = boost::container::static_vector<float, MAX_POINT_SIZE>();

        const auto& total = boost::accumulate(scores_3, 0.0F);

        boost::copy(
          scores_3 |
          boost::adaptors::transformed([&](const auto& s) { return s / total; }),
          std::back_inserter(result));

        return result;
      }();
    }

    auto greedy(needletail::state state) const noexcept {
      if (state.is_end()) {
        return state;
      }

      const auto& legal_actions = state.legal_actions();

      const auto& p_1 = greedy_play::p(state, legal_actions);
      const auto& p_2 = boltzman(p_1, _temperature);

      return greedy(state.next(legal_actions[std::distance(std::begin(p_2), boost::max_element(p_2))]));
    }

  public:
    greedy_play(needletail::game game, float temperature): _game(game), _temperature(temperature) {
      ;
    }

    auto operator()() {
      return greedy(state(_game));
    }
  };
}
