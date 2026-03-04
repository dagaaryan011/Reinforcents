# ReinforCents: A Multi-Agent Financial Market Simulation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**ReinforCents** is an immersive, simulation-based project that integrates finance and artificial intelligence into a functional version of a real-world stock and options market. The environment is populated by intelligent trading agents trained using multi-agent reinforcement learning (MARL) techniques.

**[View the Full Documentation](https://dagaaryan011.github.io/Reinforcents)**
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8c95dc5e-eeea-4898-9694-64e4eddd913b" />

---

## Key Features

* **Realistic Market Simulation:** A digital trading exchange with live order placement, a price/time-priority matching engine, real-time trade execution, and dynamic order book management.
* **Intelligent Agent Ecosystem:** The market is populated with three types of AI-driven traders, each with unique behaviors and strategies:
    * **Market Makers:** Provide liquidity by quoting bid/ask prices.
    * **Institutional Traders:** Execute large, sophisticated, and long-term strategies.
    * **Retail Traders:** Emulate the behavior of individual, often sentiment-driven, investors.
* **Advanced Reinforcement Learning:** Agents are trained using state-of-the-art algorithms like **Soft Actor-Critic (SAC)** and **Deep Deterministic Policy Gradient (DDPG)** to learn and evolve through market interaction.
* **Sophisticated Feature Engineering:** Agents make decisions using a rich set of features, including:
    * **The Black-Scholes Model:** For theoretical options pricing.
    * **Option Greeks (Δ, Γ, Θ, ν):** To measure risk and sensitivity.
    * **Technical Indicators:** EMA, RSI, Stochastic Oscillator, ADX, and CMF are used to analyze market trends.

---

## How It Works

The project is built on three core components that work together to create a dynamic trading environment.

### 1. Stock & Options Market Simulation
The foundation is a custom-built trading exchange that replicates the core functionalities of modern electronic platforms. It manages a live **order book** with bids and asks, processes trades, and provides a realistic interface for the AI agents to operate in.

### 2. Option Pricing & Analysis
An integrated **Option Pricing Engine** uses the **Black-Scholes model** to calculate the theoretical value of options. Agents leverage this, along with **Option Greeks**, to assess risk, identify mispricings, and make informed trading decisions.

### 3. Multi-Agent Reinforcement Learning
This is the "battle arena" where agents learn. Each agent type is powered by a distinct RL algorithm tailored to its role:
* **Market Maker (SAC):** The SAC agent uses a selector network to choose an option to trade and then uses its actor network to determine the optimal buy/sell prices and amounts to place in the order book. It learns to maximize profit from the bid-ask spread while managing inventory.
* **Institutional Trader (DDPG):** The DDPG agent analyzes its portfolio, market data, and sentiment to output a continuous action vector for observed options. It executes a trade on the option with the highest action value, learning complex strategies over time through a replay buffer.

---

## Reinforcement Learning Algorithms

Our agents are trained using sophisticated RL algorithms designed for complex, continuous action spaces.

### Soft Actor-Critic (SAC)
SAC is an off-policy algorithm that maximizes a trade-off between expected reward and entropy. A higher entropy encourages more exploration, which can help the agent discover better strategies.

It uses a set of five networks:
* **Actor Network:** Decides on an action by sampling from a learned probability distribution.
* **Two Critic Networks:** Evaluate the value of a state-action pair (Q-value). Using two networks helps to stabilize training.
* **Value Network:** Estimates the overall value of a given state.
* **Target Value Network:** A slow-updating copy of the Value Network used to provide stable targets during training.

### Deep Deterministic Policy Gradient (DDPG)
DDPG is an actor-critic algorithm that learns a deterministic policy for continuous action spaces. It uses a replay buffer to store and sample past experiences, stabilizing the learning process.

It uses a set of four networks:
* **Actor Network:** Directly outputs the exact action to take in a given state.
* **Critic Network:** Estimates the Q-value of the action proposed by the actor.
* **Target Actor & Critic Networks:** Slowly-updated copies of their main counterparts. These target networks provide stable Q-value targets during critic training, preventing oscillations and improving learning stability.

---

## Tech Stack

Our simulation and agent training pipeline is built with a modern, robust tech stack.

* **Programming Language:**
    * ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* **Core Libraries:**
    * ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
    * ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
    * ![YFinance](https://img.shields.io/badge/yfinance-000000?style=for-the-badge)
* **Deep Learning & RL Frameworks:**
    * ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* **Visualization & UI:**
    * ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
    * ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
    * ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
* **Tools:**
    * ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)
    * ![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## Project Structure

The repository is organized with a clear separation of concerns, separating the core simulation logic, agent implementations, and configuration files.

```
ReinforCents/
│
├── data/
│   └── models/             # Saved model weights for trained agents
│       ├── mm_agents/
│       ├── retail_agents/
│       └── insti_agents/
│
├── src/
│   ├── agents/             # Source code for all agent types
│   │   ├── retail/
│   │   ├── market_maker/
│   │   └── insti/
│   │
│   ├── market/             # Core market simulation components
│   │   ├── blackscholes.py
│   │   ├── exchange.py
│   │   ├── orderbook.py
│   │   └── ...
│   │
│   └── tools/              # Utility functions and helper scripts
│       ├── async_creator.py
│       ├── dashboard_updater.py
│       └── functions.py
│
├── main.py                 # Main script to run the simulation
├── app.py                  # Streamlit dashboard application
├── config.py               # Global configuration settings
└── requirements.txt        # Project dependencies
```

---

## Installation

To get a local copy up and running, follow these simple steps.

### Prerequisites
* Python 3.8+
* pip

### Steps
1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/dagaaryan011/Reinforcents.git](https://github.com/dagaaryan011/Reinforcents.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Reinforcents
    ```
3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

---

## Future Scope

We are planning several key upgrades to enhance the simulation's realism and capabilities:
* **Distributed and Asynchronous System:** Re-architecting the core for parallel processing to enable faster, more scalable, and more realistic simulations.
* **Increased Agent Scalability:** The new architecture will support a much larger number of agents without performance degradation.
* **Smarter Trading Algorithms:** We plan to add support for more advanced trading strategies like hedging and stop-loss orders to make agents behave more like real traders.

---


Project Link: [https://github.com/dagaaryan011/Reinforcents](https://github.com/dagaaryan011/Reinforcents)
