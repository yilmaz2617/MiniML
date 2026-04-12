#include "src/math_module.h"
#include "src/neural_net.h"
#include "src/data_module.h"
#include "src/agent.h"
#include "src/ollama_module.h"
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
    GridWorld env(4, 4, 100);
    env.add_wall(1, 1); env.add_wall(2, 2);

    // Soft update (tgt_freq=0) + Double DQN + Huber loss
    // tau=0.005: her adımda %0.5 blend → stabil, kademeli hedef değişimi
    // eps_min=0.1: min %10 keşif → catastrophic forgetting engellenir
    DQNAgent agent(GridWorld::NUM_ACTIONS, env.width, env.height,
                   64, 0.001, 0.99,
                   /*eps=*/1.0, /*eps_decay=*/0.99, /*eps_min=*/0.1,
                   /*buf_max=*/10000, /*batch=*/64,
                   /*tgt_freq=*/0, /*tau=*/0.005);

    // Öğrenmeye başlamadan önce buffer'ı çeşitli deneyimlerle doldur
    const int MIN_LEARN_SIZE = 500;

    int episodes = 1500, solved = 0;
    for (int ep = 0; ep < episodes; ++ep) {
        int state = env.reset();
        double total_reward = 0;
        bool done = false;

        while (!done) {
            int action = agent.choose_action(state);
            int prev_state = state;
            auto [reward, d] = env.step(action);
            int next_state = env.get_state();

            agent.store(prev_state, action, reward, next_state, d);
            if ((int)agent.replay_buffer.size() >= MIN_LEARN_SIZE)
                agent.learn();

            state = next_state;
            total_reward += reward;
            done = d;
        }

        agent.reward_history.push_back(total_reward);
        if (total_reward > 5.0) solved++;
        agent.decay_epsilon();  // episode başına bir kez

        if ((ep + 1) % 100 == 0) {
            // Son 100 episode'daki başarı oranı
            int n = 100, recent_solved = 0;
            for (int i = (int)agent.reward_history.size() - n; i < (int)agent.reward_history.size(); ++i)
                if (agent.reward_history[i] > 5.0) recent_solved++;
            double avg = 0;
            for (int i = (int)agent.reward_history.size() - n; i < (int)agent.reward_history.size(); ++i)
                avg += agent.reward_history[i];
            avg /= n;
            printf("  Episode %4d | AvgR: %6.2f | Eps: %.3f | Son100: %d%% | Toplam: %.1f%%\n",
                   ep + 1, avg, agent.epsilon,
                   recent_solved, 100.0 * solved / (ep + 1));
        }
    }
    printf("\n  DQN Cozum orani: %d/%d (%.1f%%)\n", solved, episodes, 100.0 * solved / episodes);
}

// ─── DEMO 4: Adam Optimizer karşılaştırması ───────────────────────────────
void demo_adam() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 4: Adam Optimizer (SGD vs Adam)\n";
    std::cout << "========================================\n";
    Matrix X(4, 2);
    X(0,0)=0;X(0,1)=0; X(1,0)=0;X(1,1)=1; X(2,0)=1;X(2,1)=0; X(3,0)=1;X(3,1)=1;
    Matrix Y(4, 1);
    Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;

    // SGD
    NeuralNet sgd_net;
    sgd_net.add(2, 16, "tanh"); sgd_net.add(16, 1, "sigmoid");
    sgd_net.train(X, Y, 1000, 0.1, "mse", 1000);
    printf("  SGD Accuracy:  %.1f%%\n", sgd_net.accuracy(X, Y));

    // Adam — aynı mimari, daha düşük LR
    NeuralNet adam_net;
    adam_net.add(2, 16, "tanh"); adam_net.add(16, 1, "sigmoid");
    adam_net.use_adam();
    adam_net.train(X, Y, 1000, 0.01, "mse", 1000);
    printf("  Adam Accuracy: %.1f%%\n", adam_net.accuracy(X, Y));
}

// ─── DEMO 5: Batch Normalization ─────────────────────────────────────────
void demo_batchnorm() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 5: Batch Normalization\n";
    std::cout << "========================================\n";

    // 8-sınıflı yapay sınıflandırma (spiral benzeri)
    const int N = 64, D = 4, C = 8;
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> noise(-0.3, 0.3);
    Matrix X(N, D), Y(N, C, 0.0);
    for (int i = 0; i < N; ++i) {
        int cls = i % C;
        double angle = cls * (2*3.14159/C) + noise(gen);
        X(i,0) = std::cos(angle) + noise(gen)*0.2;
        X(i,1) = std::sin(angle) + noise(gen)*0.2;
        X(i,2) = noise(gen); X(i,3) = noise(gen);
        Y(i, cls) = 1.0;
    }

    // BatchNorm katmanları elle zincir
    BatchNormLayer bn1(32), bn2(32);
    NeuralNet net;
    net.add(D, 32, "relu"); net.add(32, 32, "relu"); net.add(32, C, "linear");
    net.use_adam();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int e = 1; e <= 500; ++e) {
        // forward: dense → bn → dense → bn → dense
        Matrix h1  = net.layers[0].forward(X);
        Matrix h1n = bn1.forward(h1);
        Matrix h2  = net.layers[1].forward(h1n);
        Matrix h2n = bn2.forward(h2);
        Matrix out = net.layers[2].forward(h2n);

        // Softmax-CE grad (MSE ile yaklaşık)
        Matrix grad = loss::mse_grad(out, Y);
        if (e % 100 == 0 || e == 1) {
            double l = loss::mse(out, Y);
            printf("  Epoch %4d | Loss: %.4f\n", e, l);
        }

        // backward
        Matrix g2  = net.layers[2].backward_adam(grad, 0.001);
        Matrix g2n = bn2.backward(g2, 0.001);
        Matrix g1  = net.layers[1].backward_adam(g2n, 0.001);
        Matrix g1n = bn1.backward(g1, 0.001);
        net.layers[0].backward_adam(g1n, 0.001);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  (%.1f ms)\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
}

// ─── DEMO 6: CNN — Pattern Recognition ──────────────────────────────────
void demo_cnn() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 6: Conv2D + MaxPool + Dense\n";
    std::cout << "========================================\n";

    // 4×4 görüntü, 1 kanal, 2 sınıf: 'çarpı' vs 'artı'
    // Çarpı deseni (X):              Artı deseni (+):
    //  1 0 0 1                        0 0 0 0
    //  0 1 1 0                        0 1 1 0
    //  0 1 1 0                        0 1 1 0
    //  1 0 0 1                        0 0 0 0
    const int H=4, W=4, C=1, N=8;
    Matrix X_data(N, C*H*W, 0.0);
    Matrix Y_data(N, 2, 0.0);

    // 4 örnek çarpı, 4 örnek artı (küçük varyasyonlarla)
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> aug(-0.1, 0.1);
    double cross_pat[] = {1,0,0,1, 0,1,1,0, 0,1,1,0, 1,0,0,1};
    double plus_pat[]  = {0,0,0,0, 0,1,1,0, 0,1,1,0, 0,0,0,0};
    for (int i = 0; i < N; ++i) {
        double* pat = (i < N/2) ? cross_pat : plus_pat;
        for (int j = 0; j < H*W; ++j)
            X_data(i, j) = pat[j] + aug(gen);
        Y_data(i, i < N/2 ? 0 : 1) = 1.0;
    }

    Conv2DLayer conv(C, 4, 3, 1, 1);   // 1ch→4ch, 3×3, stride=1, pad=1 → 4×4
    MaxPool2DLayer pool(2, 2, 2);       // 2×2 max pool → 2×2
    NeuralNet dense;
    dense.add(4*2*2, 16, "relu");
    dense.add(16, 2, "sigmoid");
    dense.use_adam();

    printf("  Mimari: Conv2D(1→4,3×3) → MaxPool(2×2) → Dense(16) → Dense(2)\n");
    for (int e = 1; e <= 300; ++e) {
        Matrix c_out  = conv.forward(X_data, H, W);                   // (8, 4*4*4)
        Matrix p_out  = pool.forward(c_out, 4, H, W);                 // (8, 4*2*2)
        Matrix pred   = dense.forward(p_out);

        Matrix grad  = loss::mse_grad(pred, Y_data);
        // Dense backward → pool input'una doğru gradyanı döndürür
        Matrix dpool = dense.backward_with_grad(grad, 0.001);
        Matrix dconv = pool.backward(dpool);
        conv.backward(dconv, 0.005);

        if (e % 100 == 0 || e == 1) {
            double l = loss::mse(pred, Y_data);
            double acc = dense.accuracy(p_out, Y_data);
            printf("  Epoch %3d | Loss: %.4f | Acc: %.1f%%\n", e, l, acc);
        }
    }
}

// ─── DEMO 7: PER-DQN — Prioritized Experience Replay ────────────────────
void demo_per_dqn() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 7: PER-DQN (Prioritized Replay)\n";
    std::cout << "========================================\n";
    GridWorld env(4, 4, 100);
    env.add_wall(1, 1); env.add_wall(2, 2);

    PERDQNAgent agent(GridWorld::NUM_ACTIONS, env.width, env.height);
    const int MIN_LEARN = 500, episodes = 1000;
    int solved = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        int state = env.reset(); double total_reward = 0; bool done = false;
        while (!done) {
            int action = agent.choose_action(state);
            int prev   = state;
            auto [reward, d] = env.step(action);
            int next_state   = env.get_state();
            agent.store(prev, action, reward, next_state, d);
            if (agent.sum_tree.size() >= MIN_LEARN) agent.learn();
            state = next_state; total_reward += reward; done = d;
        }
        agent.reward_history.push_back(total_reward);
        if (total_reward > 5.0) solved++;
        agent.decay_epsilon();

        if ((ep + 1) % 100 == 0) {
            int n = 100, rs = 0;
            for (int i = (int)agent.reward_history.size()-n;
                 i < (int)agent.reward_history.size(); ++i)
                if (agent.reward_history[i] > 5.0) rs++;
            double avg = 0;
            for (int i = (int)agent.reward_history.size()-n;
                 i < (int)agent.reward_history.size(); ++i)
                avg += agent.reward_history[i];
            printf("  Episode %4d | AvgR: %6.2f | Son100: %d%% | beta: %.3f\n",
                   ep+1, avg/n, rs, agent.beta);
        }
    }
    printf("\n  PER-DQN Cozum orani: %d/%d (%.1f%%)\n",
           solved, episodes, 100.0*solved/episodes);
}

// ─── DEMO 8: Multi-Agent GridWorld ───────────────────────────────────────
void demo_multiagent() {
    std::cout << "\n========================================\n";
    std::cout << "  DEMO 8: Multi-Agent Kooperatif\n";
    std::cout << "========================================\n";

    MultiAgentGridWorld env(5, 5, 80);
    env.add_wall(2, 1); env.add_wall(2, 2); env.add_wall(2, 3);
    env.add_agent(0, 0, 4, 4);  // Ajan 1: sol üst → sağ alt
    env.add_agent(4, 0, 0, 4);  // Ajan 2: sağ üst → sol alt

    // Her ajan için ayrı Q-table (bağımsız öğrenim)
    QAgent a1(MultiAgentGridWorld::NUM_ACTIONS, 0.1, 0.99, 1.0, 0.995, 0.05);
    QAgent a2(MultiAgentGridWorld::NUM_ACTIONS, 0.1, 0.99, 1.0, 0.995, 0.05);

    std::cout << "Ortam:\n"; env.print();

    int episodes = 2000, both_solved = 0;
    for (int ep = 0; ep < episodes; ++ep) {
        auto states = env.reset();
        bool done = false;
        while (!done) {
            int act1 = a1.choose_action(states[0]);
            int act2 = a2.choose_action(states[1]);
            auto results = env.step({act1, act2});
            auto ns = std::vector<int>{env.get_state(0), env.get_state(1)};
            done = results[0].second;
            a1.learn(states[0], act1, results[0].first, ns[0], done);
            a2.learn(states[1], act2, results[1].first, ns[1], done);
            states = ns;
        }
        bool s1 = env.agents[0].reached, s2 = env.agents[1].reached;
        if (s1 && s2) both_solved++;

        if ((ep + 1) % 500 == 0)
            printf("  Episode %4d | Ikisi birden: %d/%d (%.1f%%) | Eps: %.3f\n",
                   ep+1, both_solved, ep+1, 100.0*both_solved/(ep+1), a1.epsilon);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "============================================\n";
    std::cout << "  MiniML Framework v0.2\n";
    std::cout << "  Adam | BatchNorm | CNN | PER-DQN | MultiAgent\n";
    std::cout << "============================================\n";
    // Argüman kontrolü: "agent" geçilirse sohbet moduna gir
    bool agent_mode = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "agent") agent_mode = true;

    if (agent_mode) {
        miniml::MiniMLAgent agent("qwen2.5:1.5b");
        agent.run();
        return 0;
    }

    demo_xor();
    demo_qlearning();
    demo_dqn();
    demo_adam();
    demo_batchnorm();
    demo_cnn();
    demo_per_dqn();
    demo_multiagent();
    std::cout << "\n  Tum demolar tamamlandi.\n";
    return 0;
}