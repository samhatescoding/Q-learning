import numpy as np
from maze import Maze

actions = ['up', 'down', 'left', 'right']
action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

class Agent:
    def __init__(self, maze, alpha, gamma, epsilon, stochasticity=0.01):
        """
        Initialize the Agent.

        Parameters:
        - maze (Maze): The maze environment where the agent will act.
        - alpha (float): The learning rate.
        - gamma (float): The discount factor.
        - epsilon (float): The exploration rate.
        - stochasticity (float): Probability of a random action (default is 0.01).
        """
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.stochasticity = stochasticity
        self.Q = np.zeros((maze.rows, maze.cols, len(actions))) 

    def choose_action(self, state):
        """
        Choose an action using ε-greedy policy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(actions)
        r, c = state
        return actions[np.argmax(self.Q[r, c])]

    def step(self, state, action):
        """
        Take a step in the maze with stochasticity.
        """
        # Introduce stochasticity: choose a random action with a small probability
        if np.random.uniform(0, 1) < self.stochasticity:
            action = np.random.choice(actions)
        
        return self.maze.step(state, action)

    def q_learning(self, episodes):
        """
        Perform Q-learning to learn the optimal policy.

        Parameters:
        - episodes (int): Number of episodes to run the Q-learning.
        """
        for episode in range(episodes):
            if episode == 0 or episode == episodes // 3 or episode == 2 * episodes // 3 or episode == episodes - 1:
                self.print_arrow_table(episode)
            
            state = (2, 0)  # Start state
            while state != (0, 3):  # Until goal is reached
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)
                r, c = state
                nr, nc = next_state
                a_idx = actions.index(action)
                
                # Q-value update rule
                self.Q[r, c, a_idx] += self.alpha * (reward + self.gamma * np.max(self.Q[nr, nc]) - self.Q[r, c, a_idx])
                state = next_state

    def print_arrow_table(self, ep):
        """
        Display the policy table with arrows indicating the best actions.

        Parameters:
        - ep (int): The current episode number.
        """
        print(f"\nPolicy Table at Episode {ep}:")
        for r in range(self.maze.rows):
            row_display = ""
            for c in range(self.maze.cols):
                if (r, c) in self.maze.blocked:  # Blocked cell
                    row_display += "  X  "
                elif (r, c) in self.maze.terminals:  # Goal state
                    row_display += "  G  "
                else:
                    best_action_idx = np.argmax(self.Q[r, c])
                    best_action = actions[best_action_idx]
                    row_display += f"  {action_arrows[best_action]}  "
            print(row_display)

    def find_best_parameters(self, alpha_vals, gamma_vals, epsilon_vals, episodes, tolerance=0.01):
        """
        Find the best parameters (alpha, gamma, epsilon) that result in the fastest Q-learning convergence.

        Parameters:
        - alpha_vals (list): List of alpha values to test.
        - gamma_vals (list): List of gamma values to test.
        - epsilon_vals (list): List of epsilon values to test.
        - episodes (int): Number of episodes for each test.
        - tolerance (float): Convergence tolerance.
        
        Returns:
        - best_params (tuple): The best parameters (alpha, gamma, epsilon).
        """
        def has_converged(Q_old, Q_new, tolerance):
            return np.max(np.abs(Q_new - Q_old)) < tolerance

        best_params = None
        fastest_convergence = episodes  # Initialize with max episodes

        for alpha in alpha_vals:
            for gamma in gamma_vals:
                for epsilon in epsilon_vals:
                    self.Q.fill(0)  # Reset Q-table
                    prev_Q = np.copy(self.Q)  # Initialize previous Q-table for convergence check

                    for episode in range(episodes):
                        state = (2, 0)  # Start state
                        while state != (0, 3):  # Until goal is reached
                            action = self.choose_action(state)
                            next_state, reward = self.step(state, action)
                            r, c = state
                            nr, nc = next_state
                            a_idx = actions.index(action)
                            
                            self.Q[r, c, a_idx] += alpha * (reward + gamma * np.max(self.Q[nr, nc]) - self.Q[r, c, a_idx])
                            state = next_state
                        
                        # Check for convergence
                        if has_converged(prev_Q, self.Q, tolerance):
                            if episode < fastest_convergence:
                                fastest_convergence = episode
                                best_params = (alpha, gamma, epsilon)
                            break  # Stop the current test when converged
                        prev_Q = np.copy(self.Q)

        if best_params:
            print(f"Best parameters: Alpha: {best_params[0]}, Gamma: {best_params[1]}, Epsilon: {best_params[2]}")
            print(f"Convergence achieved in {fastest_convergence} episodes.")
        else:
            print("No convergence achieved with current parameters.")

        return best_params
