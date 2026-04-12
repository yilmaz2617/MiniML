#pragma once
#include "math_module.h"
#include "neural_net.h"
#include <vector>
#include <deque>
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
    NeuralNet target_net;   // Sabit hedef ağ — "moving target" divergence'ı önler
    double lr, gamma, epsilon, epsilon_decay, epsilon_min;
    int num_actions, grid_width, grid_height;
    int target_update_freq, learn_count;
    std::mt19937 rng;

    // One-hot state encoding: her hücre bağımsız → duvarları zımni öğrenir (tabular Q'ya yakın)
    // Normalized (x,y) özellikler duvar bilgisi içermez — one-hot burada daha iyi
    int num_states;  // grid_width * grid_height

    struct Experience {
        std::vector<double> state, next_state;
        int action; double reward; bool done;
    };
    // FIX 3: vector + erase(begin) O(n) → deque + pop_front() O(1)
    std::deque<Experience> replay_buffer;
    size_t buffer_max;
    int batch_size;
    std::vector<double> reward_history;

    double tau;  // Soft update katsayısı

    DQNAgent(int actions, int width, int height, int hidden = 64,
             double learning_rate = 0.001, double discount = 0.99,
             double eps = 1.0, double eps_decay = 0.995, double eps_min = 0.01,
             size_t buf_max = 10000, int batch = 64,
             int tgt_freq = 0,   // 0 = soft update kullan
             double soft_tau = 0.005)
        : num_actions(actions), grid_width(width), grid_height(height),
          num_states(width * height),
          lr(learning_rate), gamma(discount),
          epsilon(eps), epsilon_decay(eps_decay), epsilon_min(eps_min),
          target_update_freq(tgt_freq), learn_count(0), tau(soft_tau),
          buffer_max(buf_max), batch_size(batch), rng(42) {
        net.add(num_states, hidden, "relu");
        net.add(hidden, hidden, "relu");
        net.add(hidden, actions, "linear");
        target_net = net;   // Başlangıçta aynı ağırlıklar
    }

    // Soft target update: θ_target = τ*θ_online + (1-τ)*θ_target
    // Hard update yerine bu kullanılınca ani politika değişimleri engellenir
    void soft_update_target() {
        for (size_t l = 0; l < net.layers.size(); ++l) {
            for (size_t i = 0; i < net.layers[l].weights.data.size(); ++i)
                target_net.layers[l].weights.data[i] =
                    tau * net.layers[l].weights.data[i] +
                    (1.0 - tau) * target_net.layers[l].weights.data[i];
            for (size_t i = 0; i < net.layers[l].biases.data.size(); ++i)
                target_net.layers[l].biases.data[i] =
                    tau * net.layers[l].biases.data[i] +
                    (1.0 - tau) * target_net.layers[l].biases.data[i];
        }
    }

    // One-hot encoding: state s → [0,0,...,1,...,0] (sadece s. index 1)
    std::vector<double> state_to_features(int state) const {
        std::vector<double> feats(num_states, 0.0);
        if (state >= 0 && state < num_states) feats[state] = 1.0;
        return feats;
    }

    int choose_action(int state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) return rng() % num_actions;
        auto feats = state_to_features(state);
        Matrix m(1, num_states, 0.0);
        for (int j = 0; j < num_states; ++j) m(0, j) = feats[j];
        Matrix q = net.forward(m);
        int best = 0;
        for (int i = 1; i < num_actions; ++i) if (q(0, i) > q(0, best)) best = i;
        return best;
    }

    void store(int state, int action, double reward, int next_state, bool done) {
        Experience exp;
        exp.state      = state_to_features(state);
        exp.next_state = state_to_features(next_state);
        exp.action = action; exp.reward = reward; exp.done = done;
        if (replay_buffer.size() >= buffer_max) replay_buffer.pop_front();
        replay_buffer.push_back(exp);
    }

    void learn() {
        if ((int)replay_buffer.size() < batch_size) return;

        // FIX 3: Tüm batch'i tek matrix'e yükle → bir forward + bir backward
        std::vector<int> indices(replay_buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        Matrix batch_s (batch_size, num_states, 0.0);
        Matrix batch_ns(batch_size, num_states, 0.0);
        for (int b = 0; b < batch_size; ++b) {
            auto& exp = replay_buffer[indices[b]];
            for (int j = 0; j < num_states; ++j) {
                batch_s (b, j) = exp.state[j];
                batch_ns(b, j) = exp.next_state[j];
            }
        }

        // Double DQN: online ağ aksiyon SEÇER, target ağ değer DEĞERLENDİRİR
        // → overestimation bias azalır, daha stabil Q değerleri
        Matrix online_q_next = net.forward(batch_ns);         // aksiyon seçimi için
        Matrix target_q_next = target_net.forward(batch_ns);  // değer için (stabil)
        // current state forward: layer cache'i backward için ayarlar
        Matrix q_current = net.forward(batch_s);

        Matrix q_target = q_current;
        for (int b = 0; b < batch_size; ++b) {
            auto& exp = replay_buffer[indices[b]];
            double target = exp.reward;
            if (!exp.done) {
                // Double DQN: online ağın en iyi aksiyonunu seç
                int best_a = 0;
                for (int i = 1; i < num_actions; ++i)
                    if (online_q_next(b, i) > online_q_next(b, best_a)) best_a = i;
                // Target ağ ile değerlendir (overestimation önlenir)
                target += gamma * target_q_next(b, best_a);
            }
            q_target(b, exp.action) = target;
        }

        // Tek backward pass: dqn_grad çift normalizasyonu önler
        // Layer::backward (1/batch_size) böler — bu yeterli, ikinci bölme gerekmez
        net.backward(loss::dqn_grad(q_current, q_target), lr);

        // Soft update: her adımda yavaşça blend — ani politika değişimi yok
        if (target_update_freq == 0)
            soft_update_target();
        else if (++learn_count % target_update_freq == 0)
            target_net = net;
        // FIX 1: epsilon artık burada DÜŞMİYOR — her adımda değil, her episode'da düşecek
    }

    // Her episode sonunda bir kez çağrılır
    void decay_epsilon() {
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;
    }
};

// ─────────────────────────────────────────────────
// SumTree — Prioritized Experience Replay için
// O(log n) öncelikli örnekleme sağlar
// Yapı: ikili ağaç, yapraklar öncelik, iç düğümler toplam
// ─────────────────────────────────────────────────
class SumTree {
public:
    int capacity;
    std::vector<double> tree;  // 2*capacity - 1 düğüm
    int write_pos = 0;
    int n_entries = 0;

    SumTree(int cap) : capacity(cap), tree(2 * cap - 1, 0.0) {}

    double total() const { return tree[0]; }
    int size() const { return n_entries; }

    void _propagate(int idx, double delta) {
        int parent = (idx - 1) / 2;
        tree[parent] += delta;
        if (parent != 0) _propagate(parent, delta);
    }

    // data_idx'in ağaç indeksi
    int leaf_idx(int data_idx) const { return data_idx + capacity - 1; }

    // Öncelik güncelle
    void update(int data_idx, double priority) {
        int ti = leaf_idx(data_idx);
        _propagate(ti, priority - tree[ti]);
        tree[ti] = priority;
    }

    // Yeni deneyim ekle, indeks döndür
    int add(double priority) {
        int idx = write_pos;
        update(idx, priority);
        write_pos = (write_pos + 1) % capacity;
        n_entries = std::min(n_entries + 1, capacity);
        return idx;
    }

    // s değerine göre yaprak bul → (data_idx, priority)
    std::pair<int, double> get(double s) const {
        int idx = 0;
        while (idx < capacity - 1) {
            int left = 2 * idx + 1, right = left + 1;
            if (s <= tree[left]) idx = left;
            else { s -= tree[left]; idx = right; }
        }
        return { idx - (capacity - 1), tree[idx] };
    }
};

// ─────────────────────────────────────────────────
// PERDQNAgent — Prioritized Experience Replay ile DQN
// TD hatası yüksek deneyimler daha sık örneklenir
// IS weights ile bias düzeltmesi yapılır
// ─────────────────────────────────────────────────
class PERDQNAgent {
public:
    NeuralNet net, target_net;
    double lr, gamma, epsilon, epsilon_decay, epsilon_min, tau;
    int num_actions, grid_width, grid_height, num_states;
    std::mt19937 rng;

    // PER hiperparametreleri
    double alpha;         // önceliklendirme gücü (0=uniform, 1=tam öncelik)
    double beta;          // IS düzeltme gücü (0→1 artar)
    double beta_increment;
    double per_eps;       // sıfır öncelik önlemek için küçük sabit
    double max_priority;

    struct Experience {
        std::vector<double> state, next_state;
        int action; double reward; bool done;
    };
    std::vector<Experience> replay_buffer;
    SumTree sum_tree;
    size_t buffer_max;
    int batch_size;
    std::vector<double> reward_history;

    PERDQNAgent(int actions, int width, int height, int hidden = 64,
                double learning_rate = 0.001, double discount = 0.99,
                double eps = 1.0, double eps_decay = 0.99, double eps_min = 0.1,
                size_t buf_max = 10000, int batch = 64, double soft_tau = 0.005,
                double a = 0.6, double b = 0.4, double b_inc = 0.001, double p_eps = 1e-6)
        : num_actions(actions), grid_width(width), grid_height(height),
          num_states(width * height), lr(learning_rate), gamma(discount),
          epsilon(eps), epsilon_decay(eps_decay), epsilon_min(eps_min), tau(soft_tau),
          alpha(a), beta(b), beta_increment(b_inc), per_eps(p_eps),
          max_priority(1.0), buffer_max(buf_max), batch_size(batch),
          sum_tree((int)buf_max), rng(42) {
        replay_buffer.reserve(buf_max);
        net.add(num_states, hidden, "relu");
        net.add(hidden, hidden, "relu");
        net.add(hidden, actions, "linear");
        target_net = net;
        net.use_adam();  // Adam optimizer ile daha stabil öğrenme
    }

    std::vector<double> state_to_features(int state) const {
        std::vector<double> f(num_states, 0.0);
        if (state >= 0 && state < num_states) f[state] = 1.0;
        return f;
    }

    int choose_action(int state) {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        if (d(rng) < epsilon) return rng() % num_actions;
        auto feats = state_to_features(state);
        Matrix m(1, num_states, 0.0);
        for (int j = 0; j < num_states; ++j) m(0, j) = feats[j];
        Matrix q = net.forward(m);
        int best = 0;
        for (int i = 1; i < num_actions; ++i) if (q(0, i) > q(0, best)) best = i;
        return best;
    }

    void store(int state, int action, double reward, int next_state, bool done) {
        Experience exp;
        exp.state      = state_to_features(state);
        exp.next_state = state_to_features(next_state);
        exp.action = action; exp.reward = reward; exp.done = done;

        int idx = sum_tree.add(std::pow(max_priority, alpha));
        if (idx < (int)replay_buffer.size())
            replay_buffer[idx] = exp;
        else
            replay_buffer.push_back(exp);
    }

    void learn() {
        if (sum_tree.size() < batch_size) return;
        beta = std::min(1.0, beta + beta_increment);

        // Öncelikli örnekleme: toplam önceliği batch_size'a böl
        double seg = sum_tree.total() / batch_size;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        Matrix batch_s (batch_size, num_states, 0.0);
        Matrix batch_ns(batch_size, num_states, 0.0);
        std::vector<int>    sampled_idx(batch_size);
        std::vector<double> is_weights(batch_size, 1.0);

        double min_p = sum_tree.total() / sum_tree.capacity;
        double max_w = std::pow(sum_tree.size() * min_p, -beta);

        for (int b = 0; b < batch_size; ++b) {
            double s = (b + uniform(rng)) * seg;
            auto [data_idx, priority] = sum_tree.get(s);
            sampled_idx[b] = data_idx;

            // IS ağırlıkları: düşük öncelikli örnekler daha yüksek ağırlık
            double p_norm = priority / sum_tree.total();
            is_weights[b] = std::pow(sum_tree.size() * p_norm, -beta) / max_w;

            auto& exp = replay_buffer[data_idx];
            for (int j = 0; j < num_states; ++j) {
                batch_s (b, j) = exp.state[j];
                batch_ns(b, j) = exp.next_state[j];
            }
        }

        // Double DQN + Target network
        Matrix online_next = net.forward(batch_ns);
        Matrix target_next = target_net.forward(batch_ns);
        Matrix q_current   = net.forward(batch_s);
        Matrix q_target    = q_current;

        for (int b = 0; b < batch_size; ++b) {
            auto& exp = replay_buffer[sampled_idx[b]];
            double target = exp.reward;
            if (!exp.done) {
                int best_a = 0;
                for (int i = 1; i < num_actions; ++i)
                    if (online_next(b, i) > online_next(b, best_a)) best_a = i;
                target += gamma * target_next(b, best_a);
            }
            // TD hatası → yeni öncelik
            double td_err = std::abs(target - q_current(b, exp.action));
            sum_tree.update(sampled_idx[b], std::pow(td_err + per_eps, alpha));
            max_priority = std::max(max_priority, td_err + per_eps);

            q_target(b, exp.action) = target;
        }

        // IS ağırlıklı gradient
        Matrix grad = loss::dqn_grad(q_current, q_target);
        for (int b = 0; b < batch_size; ++b)
            for (int a = 0; a < num_actions; ++a)
                grad(b, a) *= is_weights[b];

        net.backward(grad, lr);

        // Soft target update
        for (size_t l = 0; l < net.layers.size(); ++l) {
            for (size_t i = 0; i < net.layers[l].weights.data.size(); ++i)
                target_net.layers[l].weights.data[i] =
                    tau * net.layers[l].weights.data[i] +
                    (1.0 - tau) * target_net.layers[l].weights.data[i];
            for (size_t i = 0; i < net.layers[l].biases.data.size(); ++i)
                target_net.layers[l].biases.data[i] =
                    tau * net.layers[l].biases.data[i] +
                    (1.0 - tau) * target_net.layers[l].biases.data[i];
        }
    }

    void decay_epsilon() {
        if (epsilon > epsilon_min) epsilon *= epsilon_decay;
    }
};

// ─────────────────────────────────────────────────
// MultiAgentGridWorld — Kooperatif çok ajanlı ortam
// İki ajan kendi hedeflerine ulaşmaya çalışır
// Paylaşılan reward: ikisi de hedefe ulaşınca +20
// ─────────────────────────────────────────────────
class MultiAgentGridWorld {
public:
    int width, height, max_steps, current_step;
    struct Agent { int x, y, goal_x, goal_y; bool reached; };
    std::vector<Agent> agents;
    std::vector<std::pair<int,int>> walls;
    static const int NUM_ACTIONS = 4;

    MultiAgentGridWorld(int w = 5, int h = 5, int ms = 100)
        : width(w), height(h), max_steps(ms), current_step(0) {}

    void add_agent(int sx, int sy, int gx, int gy) {
        agents.push_back({sx, sy, gx, gy, false});
    }
    void add_wall(int x, int y) { walls.push_back({x, y}); }

    bool is_wall(int x, int y) const {
        for (auto& w : walls) if (w.first == x && w.second == y) return true;
        return false;
    }

    // Her ajan için flat state: kendi (x,y) + hedef (gx,gy) + diğer ajan (ox,oy)
    int get_state(int agent_id) const {
        const auto& a = agents[agent_id];
        // Basit encoding: kendi pozisyon × grid²
        return a.y * width + a.x;
    }

    std::vector<int> reset() {
        current_step = 0;
        for (auto& a : agents) a.reached = false;
        // Ajanları başlangıç pozisyonlarına geri al (add_agent ile verilen)
        if (agents.size() >= 1) { agents[0].x = 0; agents[0].y = 0; }
        if (agents.size() >= 2) { agents[1].x = width-1; agents[1].y = 0; }
        std::vector<int> states;
        for (size_t i = 0; i < agents.size(); ++i) states.push_back(get_state(i));
        return states;
    }

    // Tüm ajanlar için aksiyon al, her ajan için (reward, done) döndür
    std::vector<std::pair<double,bool>> step(const std::vector<int>& actions) {
        ++current_step;
        std::vector<std::pair<double,bool>> results(agents.size());

        for (size_t i = 0; i < agents.size(); ++i) {
            if (agents[i].reached) { results[i] = {0.0, true}; continue; }
            int nx = agents[i].x, ny = agents[i].y;
            if (actions[i] == 0) ny--;
            else if (actions[i] == 1) ny++;
            else if (actions[i] == 2) nx--;
            else nx++;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height && !is_wall(nx, ny))
                { agents[i].x = nx; agents[i].y = ny; }

            if (agents[i].x == agents[i].goal_x && agents[i].y == agents[i].goal_y) {
                agents[i].reached = true;
                results[i] = {10.0, false};
            } else if (current_step >= max_steps) {
                results[i] = {-1.0, true};
            } else {
                results[i] = {-0.1, false};
            }
        }

        // Tüm ajanlar hedefe ulaştıysa ekstra kooperatif ödül
        bool all_done = true;
        for (auto& a : agents) if (!a.reached) all_done = false;
        if (all_done) for (auto& r : results) r.first += 5.0;

        bool episode_done = all_done || current_step >= max_steps;
        if (episode_done) for (auto& r : results) r.second = true;

        return results;
    }

    void print() const {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool printed = false;
                for (size_t i = 0; i < agents.size() && !printed; ++i)
                    if (agents[i].x == x && agents[i].y == y)
                        { printf(" %zu", i+1); printed = true; }
                for (size_t i = 0; i < agents.size() && !printed; ++i)
                    if (agents[i].goal_x == x && agents[i].goal_y == y)
                        { printf(" G"); printed = true; }
                if (!printed && is_wall(x, y)) printf(" #");
                else if (!printed) printf(" .");
            }
            printf("\n");
        }
        printf("\n");
    }
};

} // namespace miniml
