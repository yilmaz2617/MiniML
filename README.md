---
license: mit
---

# MiniML Framework v0.2

Sıfır bağımlılıklı, saf C++ makine öğrenmesi kütüphanesi.

## Özellikler

| Modül | Açıklama |
|-------|----------|
| **Dense Layer** | SGD / Adam optimizer |
| **BatchNormalization** | Eğitim/değerlendirme modları, running stats |
| **Conv2D + MaxPool2D** | Direkt konvolüsyon, argmax backward |
| **DQN** | Experience replay, Double DQN, soft target update |
| **PER-DQN** | Prioritized Experience Replay (SumTree), %99.5 |
| **Multi-Agent** | Kooperatif GridWorld, paylaşılan ödül |
| **Ollama Agent** | Yerel LLM tabanlı Q&A, kalıcı hafıza |

## Başarı Oranları

```
XOR              → %100.0
Q-Learning       → %99.7
DQN              → %92.1
PER-DQN          → %99.5
Multi-Agent      → %89–93
```

## Derleme

```bash
bash build.sh
```

Veya manuel:
```bash
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
g++ -O2 -std=c++17 -D_WIN32_WINNT=0x0A00 main.cpp -o miniml.exe -lws2_32 -lwsock32
```

## Kullanım

```bash
# Tüm demolar
./miniml.exe

# Ollama ile sohbet ajanı (qwen2.5:1.5b gerekli)
./miniml.exe agent
```

## Agent Komutları

| Komut | Açıklama |
|-------|----------|
| `temizle` | Konuşma geçmişini sıfırlar |
| `geçmiş` | Geçmişi yazdırır |
| `çıkış` / `exit` | Ajanı kapatır |

## Bağımlılıklar (header-only)

- `third_party/httplib.h` — cpp-httplib (Ollama API için)
- `third_party/json.hpp` — nlohmann/json

Ollama kurulu değilse agent çalışmaz; diğer tüm özellikler bağımsız çalışır.

## API Örneği

```cpp
#include "src/neural_net.h"
#include "src/agent.h"
using namespace miniml;

// Sinir ağı
NeuralNet net;
net.add(2, 8, "relu");
net.add(8, 1, "sigmoid");
net.use_adam(0.001);
net.train(X, Y, 1000, 32, "mse");

// DQN agent
DQNAgent agent(16, 4);  // 16 state, 4 action
agent.remember(state, action, reward, next_state, done);
agent.learn();
agent.decay_epsilon();  // her episode sonunda

// PER-DQN
PERDQNAgent per(16, 4);
per.remember(s, a, r, ns, done);
per.learn();

// Ollama sohbet ajanı
MiniMLAgent llm("qwen2.5:1.5b");
llm.run();  // interaktif döngü
```

## GitHub

[github.com/yilmaz2617/MiniML](https://github.com/yilmaz2617/MiniML)
