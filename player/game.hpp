#pragma once

#include <cmath>
#include <random>

#include <boost/container/static_vector.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>

namespace needletail {
  const int MAX_POINT_SIZE = 32;

  class point final {
    int _y;
    int _x;

  public:
    point(int y, int x) noexcept: _y(y), _x(x) {
      ;
    }

    auto y() const noexcept {
      return _y;
    }

    auto x() const noexcept {
      return _x;
    }
  };

  float distance(const point& point_1, const point& point_2) noexcept {
    return std::sqrt(std::pow(point_1.y() - point_2.y(), 2) + std::pow(point_1.x() - point_2.x(), 2));
  }

  class game final {
    boost::container::static_vector<point, MAX_POINT_SIZE> _points;

  public:
    game(boost::container::static_vector<point, MAX_POINT_SIZE> points): _points(points) {
      ;
    }

    const auto& points() const noexcept {
      return _points;
    }
  };

  auto random_game() noexcept {
    const auto& points = [&]() {
      auto random_engine       = std::default_random_engine(std::random_device()());
      auto random_distribution = std::uniform_int_distribution<>(0, 128);

      return boost::copy_range<boost::container::static_vector<point, MAX_POINT_SIZE>>(
        boost::irange(0, 8) |
        boost::adaptors::transformed(std::function([&](const int& _) { return point(random_distribution(random_engine), random_distribution(random_engine)); })));
    }();

    return game(points);
  }

  class state final {
    game                                                 _game;
    boost::container::static_vector<int, MAX_POINT_SIZE> _route;

  public:
    state(const needletail::game& game, const boost::container::static_vector<int, MAX_POINT_SIZE>& route = {0}): _game(game), _route(route) {
      ;
    }

    auto is_end() const noexcept {
      return _game.points().size() == _route.size();
    }

    auto legal_actions() const noexcept {
      return boost::copy_range<boost::container::static_vector<int, MAX_POINT_SIZE>>(
        boost::irange(0, static_cast<int>(_game.points().size())) |
        boost::adaptors::filtered(std::function([&](const int& i) { return boost::find(_route, i) == std::end(_route); })));
    }

    const auto& route() const noexcept {
      return _route;
    }

    auto distance() const noexcept {
      return boost::accumulate(
        boost::irange(0, static_cast<int>(_route.size())),
        0.0F,
        [&](const auto& acc, const auto& i) { return acc + needletail::distance(_game.points()[_route[i]], _game.points()[_route[(i + 1) % _route.size()]]); });
    }

    auto next(int action) {
      auto next_route = [&]() {
        auto result = _route;

        result.emplace_back(action);

        return result;
      }();

      return state(_game, next_route);
    }
  };
}
