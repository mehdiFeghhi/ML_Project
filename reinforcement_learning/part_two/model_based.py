from HW4_Model import *
import numpy as np

REWARD = -1  # constant reward for non-terminal states
DISCOUNT = 0.90
MAX_ERROR = 10
NUM_ROW = 4
NUM_COL = 4
# Set up the initial environment
NUM_ACTIONS = 4
ACTIONS = [0, 1, 2, 3]


# Visualization
def printEnvironment(arr):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):

            if r == 3 and c == 3:
                val = 'G'
            else:
                val = ["<-", "down", "->", "up"][arr[r][c]]
            res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)


# Calculate the utility of a state given an action
def calculateU(env, u, row, c, action):
    states, probs, fail_probs, dones = env.possible_consequences(action, state_now=(row, c))

    list_of_next_idx_we_can_go = np.arange(len(states))
    list_of_reward_each_states_near = []
    for i in range(len(list_of_next_idx_we_can_go)):

        next_idx = list_of_next_idx_we_can_go[i]
        r = REWARD
        done = dones[next_idx]
        if done:
            r += 50
        elif np.random.rand() < fail_probs[next_idx]:
            r -= 10

        list_of_reward_each_states_near.append(probs[i] * (r + DISCOUNT * u[row][c]))

    return sum(list_of_reward_each_states_near)


# https://github.com/SparkShen02/MDP-with-Value-Iteration-and-Policy-Iteration/blob/main/valueIteration.py#LL12C1-L30C15
def valueIteration(env):
    print("During the value iteration:\n")
    U = np.zeros((4, 4))
    nextU = np.zeros((4, 4))
    while True:
        # nextU = np.zeros((4, 4))
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                nextU[r][c] = max([calculateU(env, U, r, c, action) for action in range(NUM_ACTIONS)])  # Bellman update
                error = max(error, abs(nextU[r][c] - U[r][c]))

        # env.map = nextU
        U = nextU.copy()
        print(U)
        print(error)
        if error < 0.25:
            break
    return U


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def utility_next_step(env, r, c, action,u):
    x = r
    y = c
    states = np.array([[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y]])

    if action == UP:
        selected = states[2]
    if action == DOWN:
        selected = states[3]
    if action == RIGHT:
        selected = states[1]
    if action == LEFT:
        selected = states[0]

    if selected[0] < 0 or selected[1] < 0 or selected[1] > NUM_COL - 1 or selected[0] > NUM_ROW - 1:
        return -float("inf")

    else:
        return u[selected[0]][selected[1]]


# Get the optimal policy from U
def getOptimalPolicy(env,u_map):
    policy = [[-1, -1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):

            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = utility_next_step(env, r, c, action,u_map)
                if u >= maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


env = NSFrozenLake(401722136)
env.render()
res = valueIteration(env)

policy = getOptimalPolicy(env,res)
print("The optimal policy is:\n")
printEnvironment(policy)
