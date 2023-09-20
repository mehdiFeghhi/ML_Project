import random
from pettingzoo.atari import boxing_v2
import numpy as np
import pickle
import hashlib





def find_info_of_player(first_player, last_row_player_first, first_row_player_first, last_column_player_first,
                        first_column_player_first):
    body_of_agent = [8, 8, 8, 10, 10, 10, 8, 8, 8]

    # position_head :
    # left = 2
    # right = 3
    # up = 0
    # down = 1
    if (last_row_player_first - first_row_player_first) > (last_column_player_first - first_column_player_first):
        list_of_first = []
        positions = []
        for d in range(first_row_player_first, last_row_player_first + 1):
            list_of_first.append(sum(first_player[d]))

        # Iterate over the list, looking for the subset
        for i in range(len(list_of_first) - len(body_of_agent) + 1):
            if list_of_first[i:i + len(body_of_agent)] == body_of_agent:
                positions.append(i)
        # print(positions)
        try:
            position = positions[0]
        except:
            position = 19

        head_row_up = first_player[first_row_player_first + position]
        head_row = first_player[first_row_player_first + position + 4]
        head_row_down = first_player[first_row_player_first + position + 7]

        where_is_head_row_first = first_row_player_first + position

        position_head_first = 3
        for i in range(len(head_row)):

            if head_row[i] and head_row_up[i] and head_row_down[i]:

                position_head_first = 3
                break
                # right

            elif head_row[i] and not head_row_up[i] and not head_row_down[i]:
                position_head_first = 2
                break
                # lift

        for i in range(len(head_row)):

            if head_row[i] and head_row_up[i] and head_row_down[i]:
                if position_head_first == 2:
                    where_is_head_column_first = i
                    break
                else:
                    where_is_head_column_first = i



    else:
        list_of_first = []
        positions = []
        for d in range(first_column_player_first, last_column_player_first + 1):
            list_of_first.append(sum(first_player[:, d]))

        # Iterate over the list, looking for the subset
        for i in range(len(list_of_first) - len(body_of_agent) + 1):
            if list_of_first[i:i + len(body_of_agent)] == body_of_agent:
                positions.append(i)

        # print(positions)
        try:
            position = positions[0]
        except:
            position = 19

        where_is_head_column_first = first_column_player_first + position

        head_column_right = first_player[:, first_column_player_first + position]
        head_column = first_player[:, first_column_player_first + position + 4]
        head_column_left = first_player[:, first_column_player_first + position + 7]
        for i in range(len(head_column)):

            if head_column[i] and head_column_right[i] and head_column_left[i]:
                position_head_first = 1
                break
                # down

            elif head_column[i] and not head_column_right[i] and not head_column_left[i]:
                position_head_first = 0
                break
                # up

        position_head_first = 0
        for i in range(len(head_column)):

            if head_column[i] and head_column_right[i] and head_column_left[i]:

                if position_head_first == 0:
                    where_is_head_row_first = i
                    break
                else:
                    where_is_head_row_first = i

    try:
        head_first_pos = np.array([where_is_head_row_first, where_is_head_column_first])
    except:
        where_is_head_row_first = (last_row_player_first + first_row_player_first) / 2
        where_is_head_column_first = (last_column_player_first + first_column_player_first) / 2
        head_first_pos = np.array([(last_row_player_first + first_row_player_first) / 2,
                                   (last_column_player_first + first_column_player_first) / 2])

    if position_head_first == 0:

        head_first_affine_pos = np.array([where_is_head_row_first, where_is_head_column_first - 1])

    elif position_head_first == 1:

        head_first_affine_pos = np.array([where_is_head_row_first, where_is_head_column_first + 1])

    elif position_head_first == 2:

        head_first_affine_pos = np.array([where_is_head_row_first - 1, where_is_head_column_first])

    elif position_head_first == 3:

        head_first_affine_pos = np.array([where_is_head_row_first + 1, where_is_head_column_first])

    return head_first_pos, head_first_affine_pos, position_head_first


def map_observation_three(observation, player=1):
    # main choice result - >[position_head_first,dif_row,dif_col,position_head_second]
    # one choice result - > [position_head_first,distance,degree_line,position_head_second]

    # Capture observation
    img = observation[30:180, 30:130, 0]

    # Find player white
    first_player = img == 214
    indices_first_player = np.where(first_player)

    first_row_player_first = indices_first_player[0][0]

    first_column_player_first = indices_first_player[1][0]

    last_row_player_first = indices_first_player[0][-1]

    last_column_player_first = indices_first_player[1][-1]

    # Find player black
    second_player = img == 0
    indices_second_player = np.where(second_player)
    row_first_player_second = indices_second_player[0][0]

    column_first_player_second = indices_second_player[1][0]

    row_last_player_second = indices_second_player[0][-1]

    column_last_player_second = indices_second_player[1][-1]

    head_first_pos, head_first_affine_pos, position_head_first = find_info_of_player(first_player,
                                                                                     last_row_player_first,
                                                                                     first_row_player_first,
                                                                                     last_column_player_first,
                                                                                     first_column_player_first)


    head_second_pos, head_second_affine_pos, position_head_second = find_info_of_player(second_player,
                                                                                        row_last_player_second,
                                                                                        row_first_player_second,
                                                                                        column_last_player_second,
                                                                                        column_first_player_second)

    row_head_first = head_first_pos[0]
    column_head_first = head_first_pos[1]

    row_head_second = head_second_pos[0]
    column_head_second = head_second_pos[1]

    if player == 1:
        result = np.array(
            [position_head_first, row_head_first - row_head_second, column_head_first - column_head_second,
             position_head_second])

        if row_head_first <= 31 or row_head_first >= 114 or column_head_first <= 14 or column_head_first >= 86:
            result = np.array([position_head_second, (row_head_first - row_head_second) * -1,
                               (column_head_first - column_head_second) * -1, position_head_first, row_head_first,
                               column_head_first])


    else:
        result = np.array([position_head_second, (row_head_first - row_head_second) * -1,
                           (column_head_first - column_head_second) * -1, position_head_first])

        if row_head_second <= 31 or row_head_second >= 114 or column_head_second <= 14 or column_head_second >= 86:
            result = np.array([position_head_second, (row_head_first - row_head_second) * -1,
                               (column_head_first - column_head_second) * -1, position_head_first, row_head_second,
                               column_head_second])

    return str(result.astype(int))


def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return random.randint(0, 17)
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    if s not in Q.keys():
        Q[s] = [0 for _ in range(18)]

    most_value = np.argmax(Q[s])
    index_max = [_ for _ in range(18) if Q[s][_] == Q[s][most_value]]
    # print(index_max)
    index = random.randint(0, len(index_max) - 1)

    return index_max[index]


def run_episodes(env, Q, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state = env.reset()
    tot_win = 0
    tot_eq = 0
    for _ in range(num_episodes):
        env.reset()
        print(_)
        done = False
        game_rew = 0

        # while not done:
        for agent in env.agent_iter():
            if agent == 'first_0':
                # select a greedy action
                # next_state, rew, done, _ = env.step(greedy(Q, state))
                env.step(greedy(Q, state))
                observation, reward, termination, truncation, info = env.last()
                done = termination or truncation

                next_state = map_observation_three(observation)

                state = next_state
                game_rew += reward

            elif agent == 'second_0':
                rand_num = random.randint(0, 17)
                env.step(rand_num)
                observation, reward, termination, truncation, info = env.last()
                # game_rew += reward
                done = termination or truncation

            if done:
                if (-1*game_rew) == 0:
                    tot_eq += 1
                elif (-1*game_rew) > 0:
                    tot_eq += 1
                tot_rew.append((-1*game_rew))
                break

    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))

    print(f'total win : {tot_win}')
    print(f'total equal : {tot_eq}')

    return np.mean(tot_rew)


def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005, file_read=None, number_epoch_run=0,
          number_action=18):
    # # Initialize the Q matrix
    Q = {}
    if file_read is not None:
        with open(file_read, 'rb') as f:
            Q = pickle.load(f)
    games_reward = []
    test_rewards = []

    for ep in range(number_epoch_run, num_episodes):
        env.reset()
        observation, reward, termination, truncation, info = env.last()
        state = map_observation_three(observation)
        done = termination or truncation
        tot_rew = 0
        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        action = eps_greedy(Q, state, eps)

        # loop the main body until the environment stops
        # while not done:
        for agent in env.agent_iter():

            if agent == 'first_0':
                env.step(action)  # Take one step in the environment
                observation, reward, termination, truncation, info = env.last()

                next_state = map_observation_three(observation)

                if next_state not in Q.keys():
                    Q[next_state] = [0 for _ in range(18)]
                done = termination or truncation
                # choose the next action (needed for the SARSA update)
                next_action = eps_greedy(Q, next_state, eps)
                reward = reward * -1
                # SARSA update
                Q[state][action] = Q[state][action] + lr * (
                        reward + gamma * Q[next_state][next_action] - Q[state][action])

                state = next_state
                action = next_action
                tot_rew += reward
            elif agent == 'second_0':
                rand_num = random.randint(0, 17)
                env.step(rand_num)
                observation, reward, termination, truncation, info = env.last()

                done = termination or truncation

            if done:
                games_reward.append(tot_rew)
                break
        # Test the policy every 300 episodes and print the results
        if (ep % 100) == 99:
            with open(f'checkpoint_main/v4_Q_Sarsa_learn_by_random_agent{ep}.pkl', 'wb') as f:
                pickle.dump(Q, f)
        if (ep % 300) == 299:
            print("I am in testing mood")
            with open('checkpoint_main/v4_Q_Sarsa_learn_by_random_agent.pkl', 'wb') as f:
                pickle.dump(Q, f)
            test_rew = run_episodes(env, Q, 10)

            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            with open('checkpoint_main/v4_test.pkl', 'wb') as f:
                pickle.dump(test_rew, f)

            print("Testing mood is end")

    return Q


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = boxing_v2.env(render_mode="human")
    eps = 0.02
    Q_sarsa = SARSA(env, lr=.1, num_episodes=10000, eps=eps, gamma=0.95, eps_decay=0.001,
                    file_read='checkpoint_main/v4_Q_Sarsa_learn_by_random_agent8099.pkl', number_epoch_run=6300)
    with open('checkpoint_main/v4_Q_Sarsa_learn_by_random_agent.pkl', 'wb') as f:

        pickle.dump(Q_sarsa, f)
