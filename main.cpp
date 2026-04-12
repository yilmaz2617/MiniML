#include "src/math_module.h"
#include "src/neural_net.h"
#include "src/data_module.h"
#include "src/agent.h"
#include <iostream>

using namespace miniml;

void demo_xor() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 1: XOR Problemi\n";
    std::cout << "========================================\n";
    Matrix X(4, 2);
    X(0,0)=0; X(0,1)=0; X(1,0)=0; X(1,1)=1; X(2,0)=1; X(2,1)=0; X(3,0)=1; X(3,1)=1;
    Matrix Y(4, 1);
    Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;
    NeuralNet net;
    net.add(2, 8, "tanh"); net.add(8, 8, "tanh"); net.add(8, 1, "sigmoid");
    net.train(X, Y, 5000, 0.5, "mse", 1000);
    std::cout << "\nSonuclar:\n";
    Matrix pred = net.forward(X);
    for (size_t i = 0; i < 4; ++i)
        printf("  [%.0f, %.0f] -> %.4f (beklenen: %.0f)\n", X(i,0), X(i,1), pred(i,0), Y(i,0));
    printf("  Accuracy: %.1f%%\n", net.accuracy(X, Y));
    net.save_loss("xor_loss.csv");
    net.save("xor_model.bin");
}

void demo_qlearning() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 2: Q-Learning Labirent\n";
    std::cout << "========================================\n";
    GridWorld env(5, 5, 100);
    env.add_wall(1, 1); env.add_wall(2, 1); env.add_wall(3, 1);
    env.add_wall(1, 3); env.add_wall(2, 3);
    std::cout << "Labirent:\n"; env.print();
    QAgent agent(GridWorld::NUM_ACTIONS, 0.1, 0.99, 1.0, 0.995, 0.01);
    int episodes = 1000, solved = 0;
    for (int ep = 0; ep < episodes; ++ep) {
        int state = env.reset(); double total_reward = 0; bool done = false;
        while (!done) {
            int action = agent.choose_action(state);
            auto [reward, d] = env.step(action);
            int next_state = env.get_state();
            agent.learn(state, action, reward, next_state, d);
            state = next_state; total_reward += reward; done = d;
        }
        agent.reward_history.push_back(total_reward);
        if (total_reward > 5.0) solved++;
        if ((ep + 1) % 200 == 0) {
            double avg = 0; int n = std::min(50, (int)agent.reward_history.size());
            for (int i = agent.reward_history.size() - n; i < (int)agent.reward_history.size(); ++i) avg += agent.reward_history[i];
            avg /= n;
            printf("  Episode %4d | Avg Reward: %6.2f | Epsilon: %.3f\n", ep + 1, avg, agent.epsilon);
        }
    }
    printf("\n  Cozum orani: %d/%d (%.1f%%)\n", solved, episodes, 100.0 * solved / episodes);
    agent.print_stats();
    std::cout << "\nOgrenmis ajanin yolu:\n";
    int state = env.reset(); env.print();
    bool done = false; int steps = 0;
    while (!done && steps < 20) {
        auto& q = agent.get_q(state);
        int action = std::max_element(q.begin(), q.end()) - q.begin();
        auto [reward, d] = env.step(action);
        state = env.get_state(); done = d; steps++;
    }
    env.print();
    printf("  %d adimda %s\n", steps, done && env.agent_x == env.goal_x ? "HEDEFE ULASTI!" : "ulasamadi.");
}

void demo_dqn() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 3: DQN Agent\n";
    std::cout << "========================================\n";
    GridWorld env(4, 4, 50);
    env.add_wall(1, 1); env.add_wall(2, 2);
    int total_states = env.width * env.height;
    DQNAgent agent(total_states, GridWorld::NUM_ACTIONS, 64, 0.001, 0.99, 1.0, 0.998, 0.01);
    int episodes = 500, solved = 0;
    for (int ep = 0; ep < episodes; ++ep) {
        int state = env.reset(); double total_reward = 0; bool done = false;
        while (!done) {
            int action = agent.choose_action(state);
            auto [reward, d] = env.step(action);
            int next_state = env.get_state();
            agent.store(state, action, reward, next_state, d);
            agent.learn();
            state = next_state; total_reward += reward; done = d;
        }
        agent.reward_history.push_back(total_reward);
        if (total_reward > 5.0) solved++;
        if ((ep + 1) % 100 == 0) {
            double avg = 0; int n = std::min(50, (int)agent.reward_history.size());
            for (int i = agent.reward_history.size() - n; i < (int)agent.reward_history.size(); ++i) avg += agent.reward_history[i];
            avg /= n;
            printf("  Episode %4d | Avg Reward: %6.2f | Epsilon: %.3f\n", ep + 1, avg, agent.epsilon);
        }
    }
    printf("\n  DQN Cozum orani: %d/%d (%.1f%%)\n", solved, episodes, 100.0 * solved / episodes);
}

int main() {
    std::cout << "============================================\n";
    std::cout << "  MiniML Framework v0.1\n";
    std::cout << "  Saf C++ | Sifir Bagimlilk | Her Yerde Calisir\n";
    std::cout << "============================================\n";
    demo_xor();
    demo_qlearning();
    demo_dqn();
    std::cout << "\n  Tum demolar tamamlandi.\n";
    return 0;
}