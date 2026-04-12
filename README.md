# MiniML - Lightweight C++ AI Framework

**Zero Dependencies | Runs Everywhere | Edge AI Ready**

MiniML is a pure C++ machine learning framework built from scratch with no external dependencies. Designed for edge devices, embedded systems, and anywhere lightweight AI inference is needed.

## Features

- **Math Module** - Matrix operations, vector math, activation functions (sigmoid, ReLU, tanh)
- **Neural Network Engine** - Dense layers, forward propagation, backpropagation, MSE/BCE loss
- **Data Processing** - CSV reader, normalization, train/test split
- **Q-Learning Agent** - Tabular reinforcement learning with epsilon-greedy exploration
- **DQN Agent** - Deep Q-Network with experience replay buffer
- **GridWorld Environment** - Built-in maze environment for agent training

## Quick Start

```bash
g++ -std=c++17 -O3 -o miniml main.cpp
./miniml
```

## Results

| Demo | Accuracy | Time |
|------|----------|------|
| XOR Neural Net | 100% | ~26ms |
| Q-Learning Maze | 99.7% | <1s |
| DQN Agent | WIP | - |

## Architecture

```
MiniML/
├── main.cpp              # Demo & entry point
├── src/
│   ├── math_module.h     # Matrix, Vector, Activations
│   ├── neural_net.h      # Layers, NeuralNet, Loss functions
│   ├── data_module.h     # CSV, Normalization
│   └── agent.h           # QAgent, DQNAgent, GridWorld
├── CMakeLists.txt
└── README.md
```

## Roadmap

- [ ] ONNX model parser
- [ ] Conv2D layer
- [ ] CUDA GPU backend
- [ ] WebAssembly support
- [ ] REST API server mode
- [ ] INT8/INT4 quantization
- [ ] Multi-agent support

## Requirements

- C++17 compatible compiler (GCC, Clang, MSVC)
- No external libraries needed

## License

MIT