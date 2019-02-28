#include <iostream>

#include "game.hpp"
#include "greedy_player.hpp"
#include "predict.hpp"

constexpr auto x = std::array<float, 20 * 128 * 128 * 34>{};

int main(int argc, char** argv) {
  auto g = needletail::random_game();

  {
    auto p = needletail::greedy_play(g, 1.0);
    auto s = p();
    for (auto i: s.route()) {
      std::cout << i << std::endl;
    }
    std::cout << s.distance() << std::endl;
  }

  {
    auto p = needletail::greedy_play(g, 1.0);
    auto s = p();
    for (auto i: s.route()) {
      std::cout << i << std::endl;
    }
    std::cout << s.distance() << std::endl;
  }

  {
    auto p = needletail::greedy_play(g, 1.0);
    auto s = p();
    for (auto i: s.route()) {
      std::cout << i << std::endl;
    }
    std::cout << s.distance() << std::endl;
  }

  return 0;
}
