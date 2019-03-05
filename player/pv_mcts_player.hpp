#pragma once

#include <cmath>
#include <queue>
#include <thread>

#include <boost/container/static_vector.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include "game.hpp"
#include "tf_predict.hpp"
#include "util.hpp"

namespace needletail {
  inline auto pset(std::array<float, 128 * 128>& bitmap, const point& point) noexcept {
    bitmap[point.y() * 128 + point.x()] = 1.0F;
  }

  inline auto line(std::array<float, 128 * 128>& bitmap, const point& point_1, const point& point_2) noexcept {
    const auto& delta_y = std::abs(point_2.y() - point_1.y());
    const auto& delta_x = std::abs(point_2.x() - point_1.x());

    const auto& step_y = point_1.y() < point_2.y() ? 1 : -1;
    const auto& step_x = point_1.x() < point_2.x() ? 1 : -1;

    auto error = delta_y - delta_x;

    auto y = point_1.y();
    auto x = point_1.x();

    for (;;) {
      pset(bitmap, point(y, x));

      if (y == point_2.y() && x == point_2.x()) {
        break;
      }

      const auto& error_2 = error * 2;

      if (error_2 > -delta_x) {
        error -= delta_x;
        y += step_y;
      }

      if (error_2 <  delta_y) {
        error += delta_y;
        x += step_x;
      }
    }
  }

  inline auto create_x(const state& state) noexcept {
    const auto& create_blank_channel = [&]() {
      return std::array<float, 128 * 128>();
    };

    const auto& create_point_channel = [&](const auto& action) {
      auto result = create_blank_channel();

      pset(result, state.game().points()[action]);

      return result;
    };

    const auto& create_route_channel = [&]() {
      auto result = create_blank_channel();

      for (auto i = 0; i < static_cast<int>(state.route().size()) - 1; ++i) {
        line(result, state.game().points()[state.route()[i]], state.game().points()[state.route()[i + 1]]);
      }

      return result;
    };

    const auto& legal_actions = state.legal_actions();

    auto channels = std::array<std::array<float, 128 * 128>, MAX_POINT_SIZE + 2>{
      create_route_channel(),
      create_point_channel(state.route().back()),
      create_point_channel(state.route().front())
    };

    for (auto i = 1; i < MAX_POINT_SIZE; ++i) {
      channels[i + 2] = boost::find(legal_actions, i) != std::end(legal_actions) ? create_point_channel(i) : create_blank_channel();
    }

    auto result = std::array<float, 128 * 128 * 34>();

    for (auto i = 0; i < 128 * 128; ++i) {
      for (auto j = 0; j < 34; ++j) {
        result[i * 34 + j] = channels[j][i];
      }
    }

    return result;
  }

  class pv_mcts_node final {
    const pv_mcts_node*  _previous_node;
    needletail::state _state;

    float _p;
    float _w;
    float _n;

    std::vector<pv_mcts_node> _child_nodes;

  public:
    pv_mcts_node(const pv_mcts_node* previous_node, const state& state, float p) noexcept: _previous_node(previous_node), _state(state), _p(p), _w(0), _n(0), _child_nodes() {
      ;
    }

    const auto& previous_node() const noexcept {
      return _previous_node;
    }

    const auto& state() const noexcept {
      return _state;
    }

    auto p() const noexcept {
      return _p;
    }

    auto w() const noexcept {
      return _w;
    }

    auto n() const noexcept {
      return _n;
    }

    const auto& child_nodes() const noexcept {
      return _child_nodes;
    }

    auto& w() noexcept {
      return _w;
    }

    auto& n() noexcept {
      return _n;
    }

    auto& child_nodes() noexcept {
      return _child_nodes;
    }
  };

  template<int BATCH_SIZE>
  class pv_mcts final {
    tf_predict<BATCH_SIZE>&       _tf_predict;
    pv_mcts_node                  _root_node;
    blocking_queue<pv_mcts_node*> _queue;
    bool                          _finished;

    auto& next_node(const pv_mcts_node& node) const noexcept {
      const auto& pucb = [t = boost::accumulate(node.child_nodes(), 0, [](const auto& acc, const auto& node) { return acc + node.n(); })](const auto& node) {
        return (node.n > 0 ? node.w() / node.n() : 0.0) + (1.0F * node.p() * sqrt(t) / (1 + node.n()));
      };

      return *boost::max_element(node.child_nodes(), [&](const auto& node_1, const auto& node_2) { return pucb(node_1) < pucb(node_2); });
    }

    auto evaluate(pv_mcts_node& node) noexcept {
      if (node.state().is_end() || std::empty(node.child_nodes())) {
        _queue.enqueue(&node);
        return;
      }

      evaluate(next_node(node));
    }

    auto maintain_nodes() noexcept {
      static auto requests = boost::container::static_vector<pv_mcts_node*, BATCH_SIZE>();
      static auto x        = std::array<float, BATCH_SIZE * 128 * 128 * 34>();

      for (;;) {
        requests.emplace_back(_queue.dequeue());

        if (requests.size() < BATCH_SIZE) {
          continue;
        }

        for (auto i = 0; i < BATCH_SIZE; ++i) {
          std::memcpy(x.data() + i * 128 * 128 * 34, create_x(requests[i]->state()).data(), 128 * 128 * 34);
        }

        std::cout << "*** 0" << std::endl;
        auto [y_1, y_2] = _tf_predict(x);
        std::cout << "*** 1" << std::endl;

        if (_finished) {
          break;
        }

        requests.clear();
      }
    }

  public:
    pv_mcts(tf_predict<BATCH_SIZE>& tf_predict, const state& state) noexcept: _tf_predict(tf_predict), _root_node(nullptr, state, 0.0F), _queue(), _finished(false) {
      ;
    }

    auto operator()() noexcept {
      auto t = std::thread(&pv_mcts::maintain_nodes, this);

      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);

      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);
      _queue.enqueue(&_root_node);

      t.join();
    }
  };

  template<int BATCH_SIZE>
  class pv_mcts_play final {
    game                   _game;
    float                  _temperature;
    tf_predict<BATCH_SIZE> _tf_predict;

  public:
    pv_mcts_play(const game& game, float temperature, const std::string& model_path) noexcept: _game(game), _tf_predict(model_path) {
      pv_mcts<BATCH_SIZE>(_tf_predict, state(game))();
    }
  };
}

    // auto maintain_puct_nodes() noexcept {
    //   static auto requests = boost::container::static_vector<boost::container::static_vector<puct_node&, MAX_POINT_SIZE>, BATCH_SIZE>();
    //   static auto x        = std::array<float, BATCH_SIZE * 128 * 128 * 34>();

    //   for (;;) {
    //     if (!_queue.pop(*std::end(requests))) {
    //       continue;
    //     }

    //     if (requests.size() < BATCH_SIZE) {
    //       continue;
    //     }

    //     for (auto i = 0; i < BATCH_SIZE; ++i) {
    //       std::memcpy(x + i * 128 * 128 * 34, create_x(requests[i].back()).data(), 128 * 128 * 34);
    //     }

    //     auto [y_1, y_2] = _tf_predict(x);

    //     for (auto i = 0; i < BATCH_SIZE; ++i) {
    //       auto& nodes = requests[i];
    //       auto& node  = nodes.back();

    //       const auto& legal_actions = node.legal_actions();

    //       const auto& ps = [&]() {
    //         auto result = boost::copy_range<boost::container::static_vector<float, MAX_POINT_SIZE>()>(
    //           legal_actions |
    //           boost::adaptors::transformed(std::function([&](const int& action) { return y_1[i * MAX_POINT_SIZE + action]; })));

    //         if (!std::empty(result) && *boost::min_element(result) <= 0.01) {
    //           boost::transform(result, std::begin(result), [min = *boost::min_element(result)](const auto& p) { return p - min + 0.01; });
    //         }

    //         boost::transform(result, std::begin(result), [total = boost::accumulate(result, 0.0F)](const auto& p) { return p / total; });

    //         return result;
    //       }();

    //       const auto& v = y_2[i];

    //       if (std::empty(node.child_nodes())) {
    //         for (auto i = 0; i < legal_actions.size(); ++i) {
    //           node.child_nodes().emplace_back(node.state().next(legal_actions[i]), ps[i]);
    //         }
    //       }

    //       for (const auto& past_node: nodes) {
    //         past_node.w() += v;
    //         past_node.n() += 1;
    //       }

    //       node.w() += 0.1;  // virtual lossを元に戻します。
    //     }

    //     requests.clear();
    //   }
    // }
