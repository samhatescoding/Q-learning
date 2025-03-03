import numpy as np

class Maze:
    def __init__(self, rows=None, cols=None, blocked=None, goals=None, failures=None, rewards=None):
        """
        Initialize the Maze environment.

        Parameters:
        - rows (int): Number of rows in the maze (optional).
        - cols (int): Number of columns in the maze (optional).
        - blocked (list of tuples): List of blocked positions (row, col) (optional).
        - goals (list of tuples): List of terminal positions (row, col) (optional).
        - failures (list of tuples): List of failure positions (row, col) (optional).
        - rewards (2D list or numpy array): Predefined rewards grid (optional).

        Usage:
        - Pass a rewards grid directly: Maze(rewards=rewards)
        - Or specify rows, cols, blocked, and terminal states: 
          Maze(rows=3, cols=4, blocked=[(1, 1)], goals=[(0, 3)])
        """
        if rewards is not None:
            # Parse rewards grid
            self._parse_rewards_grid(rewards)
        else:
            # Define grid from rows, cols, blocked, and goals
            self._define_grid(rows, cols, blocked, goals, failures)
        self.action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def _parse_rewards_grid(self, rewards):
        """
        Parse a predefined rewards grid with 'G' for goal states, 'F' for failure states, and 'X' for blocked cells.
        """
        self.rows, self.cols = rewards.shape
        self.rewards = np.zeros((self.rows, self.cols), dtype=float)
        self.blocked = []
        self.terminals = []

        for r in range(self.rows):
            for c in range(self.cols):
                cell = rewards[r, c]
                if cell == 'X':
                    self.rewards[r, c] = None  # Blocked cell
                    self.blocked.append((r, c))
                elif cell == 'G':
                    self.rewards[r, c] = 100  # Default goal reward
                    self.terminals.append((r, c))
                elif cell == 'F': 
                    self.rewards[r, c] = -100 # Default failure reward
                    self.terminals.append((r, c))
                else:
                    self.rewards[r, c] = float(cell)

    def _define_grid(self, rows, cols, blocked, goals, failures):
        """
        Define a grid based on rows, cols, blocked cells, goal and failure states.
        """
        self.rows = rows
        self.cols = cols
        self.rewards = np.full((self.rows, self.cols), -1, dtype=float)  # Default reward is -1
        self.blocked = blocked if blocked else []
        self.terminals = goals + failures if goals or failures else []

        for cell in self.blocked:
            self.rewards[cell[0], cell[1]] = None  # Mark blocked cells
        for goal in goals:
            self.rewards[goal[0], goal[1]] = 100  # Default goal reward
        for fail in failures:
            self.rewards[fail[0], fail[1]] = -100  # Default failure reward

    def step(self, state, action):
        """
        Take a step in the maze based on the current state and action.
        """
        r, c = state
        dr, dc = self.action_map[action]
        new_r, new_c = r + dr, c + dc

        # Check for invalid moves: blocked cells or boundaries
        if (new_r, new_c) in self.blocked or not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            return state, self.rewards[r, c]  # Stay in place
        
        return (new_r, new_c), self.rewards[new_r, new_c]

    def is_terminal(self, state):
        return state in self.terminals

    def display(self):
        print("Maze Layout:")
        for r in range(self.rows):
            row_display = ""
            for c in range(self.cols):
                if (r, c) in self.blocked:
                    row_display += "   X "
                elif (r, c) in self.terminals and self.rewards[r, c] > 0:
                    row_display += "   G " 
                elif (r, c) in self.terminals and self.rewards[r, c] < 0:
                    row_display += "   F "
                else:
                    row_display += f" {self.rewards[r, c]:3.0f} "
            print(row_display)
        print()