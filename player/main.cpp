#include <iostream>

#include "game.hpp"
#include "greedy_player.hpp"
#include "pv_mcts_player.hpp"

constexpr auto x = std::array<float, 20 * 128 * 128 * 34>{};

int main(int argc, char** argv) {
  auto game         = needletail::random_game();
  auto greedy_play  = needletail::greedy_play(game, 1.0F);
  auto pv_mcts_play = needletail::pv_mcts_play<20>(game, 1.0F, "../data/model/0000.pb");

  auto s = greedy_play();
  std::cout << s.distance() << std::endl;

  // auto b = std::array<float, 128 * 128>();

  // auto random_engine       = std::default_random_engine(std::random_device()());
  // auto random_distribution = std::uniform_int_distribution<>(0, 127);

  // for (auto i = 0; i < 1000000; ++i) {
  //   const auto y_1 = random_distribution(random_engine);
  //   const auto x_1 = random_distribution(random_engine);

  //   const auto y_2 = random_distribution(random_engine);
  //   const auto x_2 = random_distribution(random_engine);

  //   needletail::line(b, needletail::point(y_1, x_1), needletail::point(y_2, x_2));
  // }

  // for (auto y = 0; y < 128; ++y) {
  //   for (auto x = 0; x < 128; ++x) {
  //     std::cout << (b[y * 128 + x] == 0 ? "□" : "■");
  //   }

  //   std::cout << std::endl;
  // }

  // std::cout << std::endl;


  // auto g = needletail::random_game();
  // auto p = needletail::greedy_play(g, 0.0);
  // auto s = p();

  // const auto& bs = needletail::create_x(s);

  // for (auto i = 0; i < 34; ++i) {
  //   for (auto y = 0; y < 128; ++y) {
  //     for (auto x = 0; x < 128; ++x) {
  //       std::cout << (bs[y * 128 * 34 + x * 34 + i] == 0 ? "□" : "■");
  //     }

  //     std::cout << std::endl;
  //   }

  //   std::cout << std::endl;
  // }


  // {
  //   auto p = needletail::greedy_play(g, 0.0);
  //   auto s = p();
  //   for (auto i: s.route()) {
  //     std::cout << s.game().points()[i].y() << ", " << s.game().points()[i].x()<< std::endl;
  //   }
  //   std::cout << s.distance() << std::endl;

  //   const auto& bs = needletail::to_x(s);

  //   for (auto y = 0; y < 128; ++y) {
  //     for (auto x = 0; x < 128; ++x) {
  //       std::cout << (bs[0][y * 128 + x] == 0 ? "□" : "■");
  //     }

  //     std::cout << std::endl;
  //   }

  //   std::cout << std::endl;
  // }

  // {
  //   auto p = needletail::greedy_play(g, 0.1);
  //   auto s = p();
  //   for (auto i: s.route()) {
  //     std::cout << i << std::endl;
  //   }
  //   std::cout << s.distance() << std::endl;
  // }

  // {
  //   auto p = needletail::greedy_play(g, 0.2);
  //   auto s = p();
  //   for (auto i: s.route()) {
  //     std::cout << i << std::endl;
  //   }
  //   std::cout << s.distance() << std::endl;
  // }

  return 0;
}
