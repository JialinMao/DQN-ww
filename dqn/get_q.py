import numpy as np
import gym

DISCOUNT=0.96

def get_q(obs, last_act):
    # obs: [board_size, board_size, 1] matrix, obs[x, y] = 1 if there is something at (x, y), 0 otherwise
    # last_act: [board_size, board_size, 1] matrix, last_act[prev] = 1
    obs = obs.squeeze()
    last_act = last_act.squeeze()
    board_size, _ = obs.shape

    prev_x_0, prev_y_0 = np.where(last_act == 1)[0][0], np.where(last_act == 1)[1][0]
    a = np.vstack(np.where(obs==1))
    act = [a[:, i] for i in range(a.shape[1])]

    q = np.zeros_like(obs)

    for i in range(obs.size):
        r = []
        prev_x, prev_y = prev_x_0, prev_y_0
        actions = list(act)
        x = i // board_size
        y = i % board_size
        reward = 0 if (x, y) != (prev_x, prev_y) else -3
        
        while len(actions) >= 1:
            if obs[x, y] == 0:
                reward -= 1
            reward -= int(dist([[prev_x, prev_y]], [[x, y]]))
            r.append(reward)
            prev_x, prev_y = x, y
            x, y = actions.pop(np.argmin(dist([[x, y] for j in range(len(actions))], actions)))
        r.append(10-int(dist([[prev_x, prev_y]], [[x, y]])))
        q[i//board_size, i%board_size] = discount_r(r)

    return q.ravel()

def discount_r(r):
    reward = 0.0
    for i in range(len(r)):
        reward += r[i] * (DISCOUNT ** i)
    return reward

def dist(prev, curr):
    return np.sum(abs(np.array(prev) - np.array(curr)), axis=1)

def main():
    for i in range(10):
        env = gym.make('Game-v1')
        env.render()
        obs, _, terminate, _ = env.step(env.first_act)
        env.render()
        while not terminate:
            q = get_q(obs[0], obs[1])
            print q.reshape([12,12])
            obs, _, terminate, _=env.step(np.argmax(q))
            env.render()
main()
