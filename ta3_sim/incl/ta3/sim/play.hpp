#pragma once
#include "ta3/sim/tetris.hpp"

#include <iostream>
#include <optional>
#include <print>
#include <string>
inline void play() {

    ta3::sim::Tetris tetris{};
    std::optional<ta3::sim::Tetris> prev{};
    while(!tetris.gameOver()) {
        std::println("{}", tetris.string());
        std::println("type: {}", tetris.pieceQueue()[0]);
        std::println("orientation {}", tetris.pieceOrientation());
        std::println("lines cleared: {}", tetris.linesCleared());

        std::string input{};
        std::cin >> input;

        prev = tetris;
        if(input == "l") tetris.update(ta3::sim::Instruction::LEFT);
        else if(input == "r") tetris.update(ta3::sim::Instruction::RIGHT);
        else if(input == "rr") tetris.update(ta3::sim::Instruction::RRIGHT);
        else if(input == "ll") tetris.update(ta3::sim::Instruction::RLEFT);
        else if(input == "p") tetris.update(ta3::sim::Instruction::PLACE);
        else if(input == "d") tetris.update(ta3::sim::Instruction::DOWN);
        else std::println("unknown instruction");
    }
}