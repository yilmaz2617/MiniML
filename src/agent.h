#pragma once
#include "math_module.h"
#include "neural_net.h"
#include <vector>
#include <map>
#include <random>
#include <string>
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>

namespace miniml {

class QAgent {
public:
    double lr, gamma, epsilon, epsilon_decay, epsilon_min;
    int num_actions;
    std::map<int, std::vector<double>> q_table;
    std::mt19937 rng;
    std::vector<double> reward_history;
    int total_steps = 0;

    QAgent(int actions, double learning_rate = 0.1, double discount = 0.99,
           double eps = 1.0, double eps_decay = 0.995, double eps_min = 0.01)
        : num_actions(actions), lr(learning_rate), gamma(discount),
          epsilon(eps), epsilon_decay(eps_decay), epsilon_min(eps_min), rng(42) {}

    std::vector<double>& get_q(int state) {
        if (q_table.find(state) == q_table.end())
            q_table[state] = std::vector<double>(num_actions, 0.0);
        return q_table[state];
    }

    int choose_action(int state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) return rng() % num_actions;
        auto& q = get_q(state);
        return std::max_element(q.begin(), q.end()) - q.begin();
    }

    void learn(int state, int action, double reward, int next_state, bool done) {
        auto& q = get_q(state);
        auto& q_next = get_q(next_state);
        double target = reward;
        if (!done) target += gamma * *std::max_element(q_next.begin(), q_next.end());
        q[action] += lr * (target - q[action]);
        total_steps++;
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;
    }

    void print_stats() {
        printf("  Steps: %d | Q-table size: %zu | Epsilon: %.4f\n", total_steps, q_table.size(), epsilon);
    }
};

class GridWorld {
public:
    int width, height, agent_x, agent_y, goal_x, goal_y, start_x, start_y;
    std::vector<std::pair<int,int>> walls;
    int max_steps, current_step;
    static const int NUM_ACTIONS = 4;

    GridWorld(int w = 5, int h = 5, int max_s = 50)
        : width(w), height(h), max_steps(max_s), current_step(0) {
        start_x = 0; start_y = 0; goal_x = w - 1; goal_y = h - 1; reset();
    }

    void add_wall(int x, int y) { walls.push_back({x, y}); }

    int reset() { agent_x = start_x; agent_y = start_y; current_step = 0; return get_state(); }

    int get_state() const { return agent_y * width + agent_x; }

    bool is_wall(int x, int y) const {
        for (auto& w : walls) if (w.first == x && w.second == y) return true;
        return false;
    }

    std::pair<double, bool> step(int action) {
        current_step++;
        int nx = agent_x, ny = agent_y;
        if (action == 0) ny--; else if (action == 1) ny++; else if (action == 2) nx--; else if (action == 3) nx++;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height && !is_wall(nx, ny)) { agent_x = nx; agent_y = ny; }
        if (agent_x == goal_x && agent_y == goal_y) return {10.0, true};
        if (current_step >= max_steps) return {-1.0, true};
        return {-0.1, false};
    }

    void print() const {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (x == agent_x && y == agent_y) std::cout << " A";
                else if (x == goal_x && y == goal_y) std::cout << " G";
                else if (is_wall(x, y)) std::cout << " #";
                else std::cout << " .";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};

class DQNAgent {
public:
    NeuralNet net;
    double lr, gamma, epsilon, epsilon_decay, epsilon_min;
    int num_states, num_actions;
    std::mt19937 rng;

    struct Experience { Matrix state, next_state; int action; double reward; bool done; };
    std::vector<Experience> replay_buffer;
    size_t buffer_max;
    int batch_size;
    std::vector<double> reward_history;

    DQNAgent(int states, int actions, int hidden = 32,
             double learning_rate = 0.001, double discount = 0.99,
             double eps = 1.0, double eps_decay = 0.998, double eps_min = 0.01,
             size_t buf_max = 5000, int batch = 32)
        : num_states(states), num_actions(actions), lr(learning_rate), gamma(discount),
          epsilon(eps), epsilon_decay(eps_decay), epsilon_min(eps_min),
          buffer_max(buf_max), batch_size(batch), rng(42) {
        net.add(states, hidden, "relu");
        net.add(hidden, hidden, "relu");
        net.add(hidden, actions, "linear");
    }

    Matrix state_to_matrix(int state) {
        Matrix m(1, num_states, 0.0);
        if (state < num_states) m(0, state) = 1.0;
        return m;
    }

    int choose_action(int state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) return rng() % num_actions;
        Matrix q = net.forward(state_to_matrix(state));
        int best = 0;
        for (int i = 1; i < num_actions; ++i) if (q(0, i) > q(0, best)) best = i;
        return best;
    }

    void store(int state, int action, double reward, int next_state, bool done) {
        Experience exp; exp.state = state_to_matrix(state); exp.action = action;
        exp.reward = reward; exp.next_state = state_to_matrix(next_state); exp.done = done;
        if (replay_buffer.size() >= buffer_max) replay_buffer.erase(replay_buffer.begin());
        replay_buffer.push_back(exp);
    }

    void learn() {
        if ((int)replay_buffer.size() < batch_size) return;
        std::vector<int> indices(replay_buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int b = 0; b < batch_size; ++b) {
            auto& exp = replay_buffer[indices[b]];
            Matrix q_current = net.forward(exp.state);
            Matrix q_next = net.forward(exp.next_state);
            double target = exp.reward;
            if (!exp.done) {
                double max_q = q_next(0, 0);
                for (int i = 1; i < num_actions; ++i) max_q = std::max(max_q, q_next(0, i));
                target += gamma * max_q;
            }
            Matrix q_target = q_current;
            q_target(0, exp.action) = target;
            Matrix grad = loss::mse_grad(q_current, q_target);
            net.backward(grad, lr);
        }
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;
    }
};

}