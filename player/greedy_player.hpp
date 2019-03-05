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

    auto probabilities(needletail::state state, boost::container::static_vector<int, MAX_POINT_SIZE> legal_actions) const noexcept {
      auto result = boost::copy_range<boost::container::static_vector<float, MAX_POINT_SIZE>>(
        legal_actions |
        boost::adaptors::transformed(std::function([&](const int& a) { return distance(_game.points()[state.route().back()], _game.points()[a]); })));

      boost::transform(result, std::begin(result), [total = boost::accumulate(result, 0.0F)](const auto& s) { return s / total; });
      boost::transform(result, std::begin(result), [max = *boost::max_element(result)](const auto& s) { return max - s + 0.01F; });
      boost::transform(result, std::begin(result), [total = boost::accumulate(result, 0.0F)](const auto& s) { return s / total; });

      return result;
    }

    auto greedy(needletail::state state) const noexcept {
      if (state.is_end()) {
        return state;
      }

      const auto& legal_actions = state.legal_actions();

      return greedy(state.next(*choice(legal_actions, boltzman(probabilities(state, legal_actions), _temperature))));
    }

  public:
    greedy_play(needletail::game game, float temperature) noexcept: _game(game), _temperature(temperature) {
      ;
    }

    auto operator()() noexcept {
      return greedy(state(_game));
    }
  };
}
