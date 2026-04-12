#pragma once
// Windows 10+ gerekli (httplib için)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "../third_party/httplib.h"
#include "../third_party/json.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <chrono>
#include <ctime>

using json = nlohmann::json;

namespace miniml {

// ─────────────────────────────────────────────────
// Konuşma geçmişi — bellek için
// ─────────────────────────────────────────────────
struct Message {
    std::string role;     // "user" veya "assistant"
    std::string content;
};

// ─────────────────────────────────────────────────
// OllamaClient — Ollama REST API ile konuşur
// Varsayılan: http://localhost:11434
// ─────────────────────────────────────────────────
class OllamaClient {
public:
    std::string host;
    int port;
    std::string model;
    std::vector<Message> history;    // konuşma geçmişi
    std::string memory_file;         // kalıcı bellek dosyası
    bool verbose = false;

    OllamaClient(const std::string& m = "llama3.2:1b",
                 const std::string& h = "localhost", int p = 11434)
        : model(m), host(h), port(p),
          memory_file("miniml_memory.json") {
        load_memory();
    }

    // ── Bağlantı kontrolü ───────────────────────
    bool is_running() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    // ── Yüklü modelleri listele ─────────────────
    std::vector<std::string> list_models() {
        httplib::Client cli(host, port);
        auto res = cli.Get("/api/tags");
        std::vector<std::string> models;
        if (!res || res->status != 200) return models;
        auto j = json::parse(res->body);
        for (auto& m : j["models"])
            models.push_back(m["name"].get<std::string>());
        return models;
    }

    // ── Tek seferlik üretim (geçmiş yok) ────────
    std::string generate(const std::string& prompt, bool stream = false) {
        httplib::Client cli(host, port);
        cli.set_read_timeout(120);

        json req = {
            {"model",  model},
            {"prompt", prompt},
            {"stream", false}
        };

        auto res = cli.Post("/api/generate", req.dump(), "application/json");
        if (!res || res->status != 200) return "[Hata: Ollama yanıt vermedi]";
        auto j = json::parse(res->body);
        return j.value("response", "");
    }

    // ── Sohbet (geçmiş korunur) ─────────────────
    std::string chat(const std::string& user_msg,
                     const std::string& system_prompt = "") {
        // Kullanıcı mesajını ekle
        history.push_back({"user", user_msg});

        httplib::Client cli(host, port);
        cli.set_read_timeout(120);

        json messages = json::array();

        // Sistem promptu (MiniML bilgisi + kişilik)
        std::string sys = system_prompt.empty() ? default_system_prompt() : system_prompt;
        messages.push_back({{"role","system"}, {"content", sys}});

        // Geçmiş (son 10 mesaj — bağlam penceresi)
        int start = std::max(0, (int)history.size() - 10);
        for (int i = start; i < (int)history.size(); ++i)
            messages.push_back({{"role", history[i].role},
                                 {"content", history[i].content}});

        json req = {{"model", model}, {"messages", messages}, {"stream", false}};

        auto res = cli.Post("/api/chat", req.dump(), "application/json");
        if (!res || res->status != 200) {
            history.pop_back();
            return "[Hata: Ollama yanıt vermedi]";
        }

        auto j = json::parse(res->body);
        std::string reply = j["message"]["content"].get<std::string>();
        history.push_back({"assistant", reply});
        save_memory();
        return reply;
    }

    // ── Geçmişi temizle ─────────────────────────
    void clear_history() {
        history.clear();
        std::cout << "  [Geçmiş temizlendi]\n";
    }

    // ── Geçmişi yazdır ──────────────────────────
    void print_history() {
        if (history.empty()) { std::cout << "  [Geçmiş boş]\n"; return; }
        for (auto& m : history)
            printf("  [%s]: %s\n\n", m.role.c_str(), m.content.c_str());
    }

    // ── Kalıcı belleğe kaydet ───────────────────
    void save_memory() {
        json j;
        j["model"] = model;
        j["history"] = json::array();
        // Son 20 mesajı sakla
        int start = std::max(0, (int)history.size() - 20);
        for (int i = start; i < (int)history.size(); ++i)
            j["history"].push_back({{"role",history[i].role},
                                     {"content",history[i].content}});
        // Zaman damgası
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        j["last_updated"] = std::string(std::ctime(&t));

        std::ofstream f(memory_file);
        f << j.dump(2);
    }

    // ── Bellekten yükle ─────────────────────────
    void load_memory() {
        std::ifstream f(memory_file);
        if (!f.is_open()) return;
        try {
            json j = json::parse(f);
            history.clear();
            for (auto& m : j.value("history", json::array()))
                history.push_back({m["role"], m["content"]});
            if (verbose)
                printf("  [%zu mesaj bellekten yüklendi]\n", history.size());
        } catch (...) {}
    }

private:
    // MiniML hakkında varsayılan sistem promptu
    std::string default_system_prompt() {
        return R"(Sen MiniML Framework'ün akıllı asistanısın.
MiniML, saf C++ ile yazılmış, sıfır bağımlılıklı bir makine öğrenmesi kütüphanesidir.

Yeteneklerin:
- Dense katmanlar (SGD / Adam optimizer)
- BatchNormalization, Conv2D, MaxPool2D
- DQN, PER-DQN (Prioritized Experience Replay)
- Multi-Agent GridWorld ortamı
- Model kaydetme/yükleme (.bin)

Başarılar:
- XOR: %100 accuracy
- Q-Learning: %99.7
- DQN: %92.1
- PER-DQN: %99.5
- Multi-Agent: %89-93

Türkçe veya İngilizce sorulara kısa, net, teknik cevaplar ver.
Kod örneklerini C++ olarak göster.)";
    }
};

// ─────────────────────────────────────────────────
// MiniMLAgent — Dil + hesaplama birleşimi
// Sorulara cevap verir, deney çalıştırır, hafıza tutar
// ─────────────────────────────────────────────────
class MiniMLAgent {
public:
    OllamaClient llm;
    bool running = true;

    MiniMLAgent(const std::string& model = "llama3.2:1b")
        : llm(model) {}

    // ── Başlatma kontrolü ───────────────────────
    bool init() {
        printf("\n=== MiniML Agent ===\n");
        printf("  Model: %s\n", llm.model.c_str());

        if (!llm.is_running()) {
            printf("  [!] Ollama çalışmıyor. Lütfen 'ollama serve' çalıştırın.\n");
            return false;
        }

        auto models = llm.list_models();
        bool found = false;
        for (auto& m : models) if (m.find(llm.model) != std::string::npos) found = true;

        if (!found) {
            printf("  [!] '%s' modeli yok. Yükleniyor...\n", llm.model.c_str());
            printf("  Komut: ollama pull %s\n", llm.model.c_str());
            return false;
        }

        printf("  Ollama: çalışıyor\n");
        printf("  Geçmiş: %zu mesaj yüklendi\n", llm.history.size());
        printf("  'çıkış' veya 'exit' yazarak çıkabilirsin.\n");
        printf("  'temizle' ile geçmişi sıfırlayabilirsin.\n\n");
        return true;
    }

    // ── Sohbet döngüsü ──────────────────────────
    void run() {
        if (!init()) return;

        std::string input;
        while (running) {
            printf("Sen: ");
            std::getline(std::cin, input);
            if (input.empty()) continue;

            // Özel komutlar
            if (input == "çıkış" || input == "exit" || input == "quit") {
                printf("  [Agent kapatılıyor...]\n");
                break;
            }
            if (input == "temizle" || input == "clear") {
                llm.clear_history(); continue;
            }
            if (input == "geçmiş" || input == "history") {
                llm.print_history(); continue;
            }

            // LLM'e gönder
            auto t0 = std::chrono::high_resolution_clock::now();
            std::string reply = llm.chat(input);
            auto ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            printf("\nAgent: %s\n", reply.c_str());
            printf("  (%.0f ms)\n\n", ms);
        }
    }

    // ── Tek soru ────────────────────────────────
    std::string ask(const std::string& question) {
        return llm.chat(question);
    }
};

} // namespace miniml
