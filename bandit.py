"""
K-Armed Bernoulli Bandit Algorithms (Object-oriented python implementation)
Implement: Greedy, Epsilon-Greedy, UCB1, Thompson Sampling
Plots results in the terminal using plotext.
"""

"""
1. Create a folder named ba2fullrollnumber (all in lowercase), and initialize git init with your full name and email id. 
2. Place this boilerplate code, and do first commit.
3. Commit 2: After implementing  greedy
4. Commit 3: After implementing epsilon-greedy (
5. Commit 4: After implementing UCB1
6. Commit 5: After implmenting Thompson sampling
7. Commit 6: After generating text-based plot (using plotext) for showing regret analysis plot1: Cumulative regret vs time steps (t). Plot name: plot1.txt
8. Commit 7: After generating text-based plot for showing average regret for (T=10000) over different bandit algorithms. Plot name: plot2.txt

Evaluation: 
1. File naming, and git commits and timings are used to verify effort of implmementation and authenticity.
2. Total marks: 10
3. For evaluation, your implementation should extend the given boiler plate code. All variables given in boilerplate code are self-explanatory. Use additional variables (justifying their implementation in comments).

Deliverables:
One zipped folder with 3 files: python program, and 2 plots.

"""

import numpy as np
import plotext as plt

# Bandit Environment

class BernoulliBandit:
    def __init__(self, means):
        self.means = np.array(means)
        self.K = len(means)
        self.best_mean = np.max(means)

    def pull(self, arm):
        return int(np.random.rand() < self.means[arm])

# Greedy bandit

def greedy(bandit, T):
    """Pure greedy: always exploit the current best estimated arm.
    Pulls each arm once to initialise, then always picks argmax(mean)."""
    K = bandit.K

    #estimated mean reward for each arm
    counts = np.zeros(K)  # Number of times each arm was pulled
    values = np.zeros(K)  # Estimated mean reward for each arm

    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    cumulative_regret = 0

    # Initialise: pull each arm once
    for arm in range(K):
        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] = reward  # Initial estimate is just the first reward

        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)

    # Main loop: pull the arm with the highest estimated mean reward
    for t in range(K, T):
        arm = np.argmax(values)
        reward = bandit.pull(arm)
        counts[arm] += 1

        #update mean estimate incrementally
        values[arm] += (reward - values[arm]) / counts[arm]  # Update estimated mean
        
        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)

    return np.array(rewards), np.array(regrets)


# Epsilon-Greedy

def epsilon_greedy(bandit, T, epsilon=0.1):
    """Epsilon-greedy: explore uniformly at random with probability epsilon,
    exploit the current best arm otherwise."""
    K = bandit.K 
    #track pulls and estimated mean rewards for each arm
    counts = np.zeros(K)  # Number of times each arm was pulled 
    values = np.zeros(K)  # Estimated mean reward for each arm

    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    cumulative_regret = 0

    # Initialise: pull each arm once
    for arm in range(K):
        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] = reward  # Initial estimate is just the first reward

        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)

    # main loop
    for t in range(K, T):
        #exploration vs exploitation
        if np.random.rand() < epsilon:
            arm = np.random.randint(K)  # Explore: choose random arm    
        else:
            arm = np.argmax(values)  # Exploit: choose best estimated arm

        reward = bandit.pull(arm)
        counts[arm] += 1

        # incremental update of estimated mean reward for the chosen arm
        values[arm] += (reward - values[arm]) / counts[arm]

        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)

    return np.array(rewards), np.array(regrets)

# UCB1
def ucb1(bandit, T):
    K = bandit.K

    # track pulls and estimated mean rewards for each arm
    counts = np.zeros(K)  # Number of times each arm was pulled
    values = np.zeros(K)  # Estimated mean reward for each arm

    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    cumulative_regret = 0

    # Pull each arm once
    for arm in range(K):
        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] = reward  # Initial estimate is just the first reward

        regret = bandit.best_mean - reward
        cumulative_regret += regret

        rewards.append(reward)
        regret = bandit.best_mean - bandit.means[arm]

    # Main loop: pull arm with highest UCB index
    for t in range(K, T):
        # compute UCB values
        ucb_values = values + np.sqrt((2 * np.log(t + 1)) / counts)

        arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        counts[arm] += 1

        # update estimated mean
        values[arm] += (reward - values[arm]) / counts[arm]

        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)
    return np.array(rewards), np.array(regrets)


# Thompson Sampling
def thompson_sampling(bandit, T):
    K = bandit.K

    #beta distribution parameters
    alpha = np.ones(K)  # Successes + 1
    beta = np.ones(K)   # Failures + 1

    #Initialize algorithm dependent variable here
    rewards, regrets = [], []
    cumulative_regret = 0

    for t in range(T):
        #sample from beta distribution for each arm
        samples = np.random.beta(alpha, beta)

        #select arm with highest sample
        arm = np.argmax(samples)

        reward = bandit.pull(arm)

        #update beta parameters based on observed reward
        if reward == 1:
            alpha[arm] += 1  # success
        else:
            beta[arm] += 1   # failure

        regret = bandit.best_mean - bandit.means[arm]
        cumulative_regret += regret

        rewards.append(reward)
        regrets.append(cumulative_regret)

    return np.array(rewards), np.array(regrets)

# ──────────────────────────────────────────────
# Run & Plot
# ──────────────────────────────────────────────

def run_experiment(means, T, n_runs):
    algorithms = {
        "Greedy":            greedy,
        "Eps-Greedy(0.1)":   lambda b, T: epsilon_greedy(b, T, epsilon=0.1),
        "UCB1":              ucb1,
        #"KL-UCB":            kl_ucb,
        "Thompson Sampling": thompson_sampling,
    }
    results = {name: [] for name in algorithms}

    for _ in range(n_runs):
        bandit = BernoulliBandit(means)
        for name, algo in algorithms.items():
            _, regret = algo(bandit, T)
            results[name].append(regret)

    # Average over runs
    avg_regrets = {name: np.mean(runs, axis=0) for name, runs in results.items()}

    # ── Plot 1: Cumulative Regret Line Chart ──
    markers = {
        "Greedy":            "o",
        "Eps-Greedy(0.1)":   "x",
        "UCB1":              "^",
        #"KL-UCB":            "s",
        "Thompson Sampling": "*",
}
    plt.clear_figure()
    plt.theme("pro")  

    for name, avg_regret in avg_regrets.items():
        plt.plot(avg_regret, label=name, marker=markers[name])

    plt.title("Cumulative Regret vs Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")


    plt.save_fig("plot1.txt")   # save FIRST
    plt.show()                  # then display

    # ── Plot 2: Final Average Regret Bar Chart ──



if __name__ == "__main__":
    np.random.seed(42)
    MEANS = [0.1, 0.3, 0.5, 0.6, 0.9]   # 5-armed bandit, best arm = 0.9
    T     = 10000
    n_RUNS  = 50
    run_experiment(MEANS, T, n_RUNS)
