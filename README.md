# High-Frequency Market Making Engine Architecture: Stochastic Control vs. Deep Reinforcement Learning

> *A rigorous engineering architecture and comparative analysis bridging the mathematical elegance of stochastic control with the non-linear pattern recognition of modern Deep Reinforcement Learning (A2C).*

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![C++](https://img.shields.io/badge/C++-17-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-LibTorch-red.svg)

---

## 📖 Table of Contents
1. [Ideation & Theoretical Background](#ideation--theoretical-background)
2. [Market Microstructure & Order Book Mechanics](#market-microstructure--order-book-mechanics)
3. [The Baseline: Avellaneda-Stoikov (AS) Model](#the-baseline-avellaneda-stoikov-as-model)
4. [The Challenger: Advantage Actor-Critic (A2C) Agent](#the-challenger-advantage-actor-critic-a2c-agent)
5. [Game Theoretic Parallels & CFR](#game-theoretic-parallels--cfr)
6. [Phase-Wise Integration & Architecture Plan](#phase-wise-integration--architecture-plan)
7. [Problems Faced & Structural Limitations](#problems-faced--structural-limitations)
8. [Results & Latency Profiling](#results--latency-profiling)
9. [How to Run the Project](#how-to-run-the-project)
10. [References](#references)

---

## 1. Ideation & Theoretical Background

The architecture of a high-frequency market-making (HFMM) engine represents one of the most complex engineering challenges in modern quantitative finance. Premier proprietary trading firms operate as the primary liquidity providers across global electronic exchanges, functioning as the vital counterparty to millions of transactions. 

The fundamental objective of this project is to construct an algorithm capable of parsing immense volumes of limit order book (LOB) telemetry to continuously capture the bid-ask spread. This requires rigorous mitigation of two existential threats: **Adverse Selection** (information asymmetry) and **Inventory Risk** (position delta).

To demonstrate absolute proficiency, this project moves beyond theoretical abstracts to establish a concrete, exhaustively detailed architectural blueprint comparing:
1. **The Classical Approach:** The Avellaneda-Stoikov (AS) stochastic control framework.
2. **The Modern Approach:** A Deep Reinforcement Learning (DRL) agent using an Advantage Actor-Critic (A2C) architecture.

---

## 2. Market Microstructure & Order Book Mechanics

Market makers operate as passive liquidity providers, submitting non-marketable limit orders. A market maker derives profit by continuously posting a bid price ($p^b$) and an ask price ($p^a$). The absolute distance between these two prices constitutes the quoted spread ($\delta = p^a - p^b$).

However, providing liquidity continuously exposes the algorithm to:
- **Inventory Risk:** Accumulating a nonzero inventory position exposes the firm to direct market risk. Sudden price movements against the accumulated position can instantly obliterate thousands of fractional spread captures.
- **Adverse Selection:** When an informed trader executes an aggressive market order, the market maker assumes a position mere milliseconds before the mid-price inevitably moves against them, resulting in immediate negative mark-to-market PnL.

---

## 3. The Baseline: Avellaneda-Stoikov (AS) Model

The Avellaneda-Stoikov (AS) model (2008) serves as the foundational text for continuous-time market making, framing liquidity provision as a stochastic optimal control problem.

### Dynamic Reservation Price
The algorithm calculates an internal fair-value estimate skewed by inventory accumulation:

$$r(t_j) = p^m(t_j) - I(t_j) \gamma \sigma^2 (T - t_j)$$

Where $p^m(t_j)$ is the mid-price, $I(t_j)$ is the inventory position, $\gamma$ is risk aversion, $\sigma^2$ is volatility, and $(T-t_j)$ is time remaining.

If the algorithm accumulates a long position ($q > 0$), the negative penalty forces the reservation price downward, making sell limit orders highly competitive to rapidly liquidate inventory.

### Optimal Spread Calibration
The exact optimal distances at which to place quotes are calculated to maximize spread capture while pricing in adverse market moves:

$$\text{Spread} = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln \left(1 + \frac{\gamma}{k}\right)$$

**Limitation:** While computationally pristine (executing in nanoseconds), the AS equations rely on rigid statistical assumptions (Brownian motion, Poisson arrivals) that are fundamentally blind to deep high-frequency telemetry such as localized order flow imbalance and cumulative notional depth.

---

## 4. The Challenger: Advantage Actor-Critic (A2C) Agent

To actively anticipate micro-structural price movements, we implemented a Deep Reinforcement Learning (DRL) agent using the Advantage Actor-Critic (A2C) architecture. By utilizing deep neural networks, the model ingests massively multidimensional state spaces to discover non-linear quoting policies.

### Continuous Action Spaces
Forcing a high-frequency market maker into discrete quoting bins destroys granularity. Therefore, the Actor network outputs a **continuous Gaussian probability distribution**. The network outputs:
- **$\mu$ (Mean Tensor):** The deterministic center of the desired quote action.
- **$\sigma^2$ (Variance Tensor):** The exploration boundary.

### Engineered Reward Function & Penalty Rigor
An improperly calibrated reward function results in "reward hacking" (e.g., directional momentum trading). To strictly enforce liquidity provision, the reward function mathematically balances gross spread capture against a **non-linear quadratic inventory penalty**:

$$\text{Reward} = \text{Trading Profit} - \lambda_1 \Delta I^2 - \lambda_2 \theta$$

By squaring the inventory term ($\Delta I^2$), the agent faces exponentially harsher negative rewards as its position deviates from zero, mathematically forcing it to mimic AS-style inventory skewing.

---

## 5. Game Theoretic Parallels & CFR

The limit order book is fundamentally a multi-agent game of imperfect information. This project bridges reinforcement learning with algorithmic game theory, specifically **Counterfactual Regret Minimization (CFR)**. 

Just as advanced Poker AI solvers map hand strengths to continuous probabilities to boost neural network accuracy, our A2C agent utilizes continuous Gaussian outputs to map infinite variations of the LOB into continuous spread distributions. The Advantage function in A2C effectively mirrors the minimization of counterfactual regret against a baseline state-value.

---

## 6. Phase-Wise Integration & Architecture Plan

### Phase 1: Python LOB Simulation & Baseline
- **`simulator/lob_env.py`:** A highly engineered Gymnasium environment that simulates the Limit Order Book, modeling mid-price as Brownian motion and simulating order arrivals via a Poisson process. The state space is heavily enriched with microstructural signals (OFI, Cumulative Notional).
- **`models/avellaneda_stoikov.py`:** A pure implementation of the AS model's Hamilton-Jacobi-Bellman (HJB) derived optimal quoting equations.

### Phase 2: DRL Agent Training (PyTorch)
- **`scripts/train_rl.py`:** Trains the A2C policy using `stable-baselines3`. Implements the quadratic inventory penalty reward architecture.
- **`scripts/visualize.py`:** Runs both the AS and A2C models simultaneously against the simulator, plotting cumulative PnL and Inventory trajectories.
- **TorchScript Export:** Extracts the trained Actor policy network and serializes it to `a2c_actor.pt` for C++ ingestion.

### Phase 3: Ultra-Low Latency C++ Inference Engine
- **`cpp_engine/src/inference_engine.cpp`:** A deterministic C++ tick data handler. Loads the PyTorch `.pt` model natively via **LibTorch**. 
- **Latency Profiler:** Executes both the pure C++ AS mathematical calculations and the LibTorch Neural Network forward pass over 1000 simulated ticks to measure the exact microsecond overhead of machine learning in production.

---

## 7. Problems Faced & Structural Limitations

Deploying machine learning in HFMM cannot be approached with naive optimism. This project actively documented and addressed several critical engineering failures:

1. **Extreme Hyperparameter Sensitivity:** The A2C policy gradient suffered from extreme fragility. If the entropy coefficient decayed too rapidly, the agent collapsed into a sub-optimal deterministic policy (quoting infinitely wide spreads to guarantee safety). The $\lambda_1$ inventory scalar had to be meticulously tuned; a weak penalty resulted in speculative directional trading, while a harsh penalty caused the agent to aggressively cross the spread to dump inventory, destroying capital in taker fees.
2. **Vulnerability to Out-of-Sample Market Shocks:** Unlike the AS model—which dynamically widens spreads instantly when market variance ($\sigma^2$) spikes—deep neural networks are interpolative. When fed unprecedented out-of-distribution state tensors (like a flash crash), the Actor network outputs random noise.
3. **Simulation Artifacts:** The agent frequently learned to exploit mechanical loopholes in the Python matching engine (e.g., zero-latency cancellation assumptions) that immediately evaporated when transitioned to the C++ latency environment.

---

## 8. Results & Latency Profiling

### Simulation Alpha
In the Python environment, the A2C agent significantly outperformed the classical stochastic model by anticipating toxic order flow imbalances:
- **AS Final PnL:** `~770.64`
- **A2C Final PnL:** `~2040.76`

### The Hardware Reality (Latency Benchmark)
In high-frequency trading, a 500-microsecond delay in updating quotes means persistently losing queue priority. The C++ `inference_engine` yielded the following profiling data:
- **A2C (LibTorch) Forward Pass Latency:** `~14.14 microseconds` per tick.
- **Avellaneda-Stoikov (AS) Calculation:** `~21.79 nanoseconds` per tick.

**Conclusion:** The classical AS math is roughly **648x faster** than the Deep Neural Network inference. To survive a live FIFO matching engine, the optimal architecture is a **Symbiotic Hybrid**: The AS model must act as the unyielding, sub-microsecond risk-management layer, while the deep learning agent acts as an auxiliary overlay dynamically modulating the $\gamma$ risk aversion parameters asynchronously.

---

## 9. How to Run the Project

### Prerequisites
- Python 3.10+
- CMake & a modern C++ compiler (Apple Clang / GCC)
- PyTorch (with LibTorch headers)

### Setup & Training
```bash
# Clone the repository
git clone https://github.com/photon2563/High-Frequency-Market-Making-Engine-Architecture-Stochastic-VS-RL-Strategy.git
cd High-Frequency-Market-Making-Engine-Architecture-Stochastic-VS-RL-Strategy

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch gymnasium numpy pandas matplotlib stable-baselines3

# Train the A2C Agent and export to TorchScript
python scripts/train_rl.py

# Generate visualization plots (AS vs A2C)
python scripts/visualize.py
```

### Running the C++ Latency Benchmark
```bash
# Build the C++ engine
cd cpp_engine/build
cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") ..
make

# Execute the inference profiler (Ensure LibTorch is in your path)
cd ../..
DYLD_LIBRARY_PATH=venv/lib/python3.13/site-packages/torch/lib ./cpp_engine/build/inference_engine models/a2c_actor.pt
```

---

## 10. References

1. High-frequency trading in a limit order book (Avellaneda & Stoikov, 2008)
2. Reinforcement Learning Approaches to Optimal Market Making (MDPI, 2021)
3. Hierarchical Deep Counterfactual Regret Minimization (arXiv:2305.17327)
4. Zero-shot adaptation to order book dynamics (arXiv:2605.21707)
5. Resolving Latency and Inventory Risk in Market Making with Reinforcement Learning (arXiv:2505.12465)
6. Equilibrium Finding for Large Adversarial Imperfect-Information Games (CMU-CS-20-132)
7. Performance of Deep Reinforcement Learning for High Frequency Market Making on Actual Tick Data (AAMAS 2022)
