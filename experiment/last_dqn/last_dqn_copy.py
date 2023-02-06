import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import math

# %%
df_name = 'nov_nine_var.xlsx'

# %%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

np.set_printoptions(precision=6, suppress=True)

# %%
dnn_model = tf.keras.models.load_model('dnn.h5')

# %%
ACTION_LIST = [
    0.05372157772621804,
    0.06143633695924037,
    0.05213242358253655,
    0.05617721888256303,
    0.06891978903661669,
    0.06451126527380139,
    0.04095511726333548,
    0.02829305913764111,
    0.039145989505637474,
    0.04215891490678202,
    0.03756490995470125,
    0.03903407009402856,
    0.0270467351673848,
    0.03764159956325903,
    0.037671849568861077,
    0.030854766253640736,
    0.05383277656911071,
    0.03601719716524669,
    0.03442764100663552,
    0.026549798057918694,
    0.04307480810973876
]

# %%
# dqn paramater
GAMMA = 0.9
BATCH_SIZE = 128
TRAIN_FLAG = 4000
EPISODE_DONE = 1000
EPS_DECAY = 0.99

# %%
df = pd.read_excel(df_name).iloc[:,1::]

scaler = MinMaxScaler()
X = scaler.fit_transform(df.iloc[:,0:21].to_numpy())

starting_state = X[-1].reshape(1, 21)

# %%
def set_goal(goal_df_name):
    """ set goal destination
    Args:
        goal_df_name(str): df_name in documents/result/
    Returns:
        goal_state(ndArray, (1, 21)): the state of lowest rate in df
    """
    goal_df = pd.read_excel(goal_df_name).iloc[:,1::].to_numpy()
    index = goal_df[:,-1].argmin()

    goal_state = goal_df[:,0:21][index].reshape(1, 21)
    goal_state = scaler.transform(goal_state)

    return goal_state

# %%
def return_action(i):
    a = np.zeros((1, 21))
    j = i // 2

    if i % 2 == 0:
        a[0][j] = -ACTION_LIST[j]
    
    else:
        a[0][j] = ACTION_LIST[j]
    
    return a

# %%
def return_state(s, a):
    ns = s + a
    return ns

# %%
def return_reward(ns, gs):
    dist = np.sqrt(np.sum(np.square(gs - ns)))
    loss = dist
    
    return loss

def check_end(ns, gs):
    end = [0 for i in range(21)]
    for i in range(21):
        if ns[0][i] == gs[0][i]:
            end[i] = 1
    return end


# %%
class DQN_Network(tf.keras.models.Model):
    def __init__(self):
        super(DQN_Network, self).__init__()
        self.input_layer = tf.keras.layers.Dense(128, input_shape=(21, ), activation='relu')

        self.hidden_layer = tf.keras.models.Sequential()
        self.hidden_layer.add(tf.keras.layers.Dense(128, activation='relu'))
        self.hidden_layer.add(tf.keras.layers.Dense(128, activation='relu'))

        self.ouput_layer = tf.keras.layers.Dense(42, activation='linear')

    def call(self, x):
        i = self.input_layer(x)
        h = self.hidden_layer(i)
        o = self.ouput_layer(h)
        return o

# %%
class DQN_Agent:
    def __init__(self):
        self.train_model = self.set_model()
        self.target_model = self.set_model()

        self.memory = deque(maxlen=60000)
        self.episode = 1

        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-10)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def set_model(self):
        net = DQN_Network()
        net.build(input_shape=(1, 21))

        optim = tf.keras.optimizers.Adam(learning_rate=1e-10)
        net.compile(optimizer=optim, loss='mse')
        return net

    def update_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def memorize(self, cs, a_i, r, ns, d):
        if d:
            self.episode += 1
        
        self.memory.append(
            (
                tf.convert_to_tensor(tf.cast(cs, tf.float32)),
                a_i,
                tf.convert_to_tensor(tf.cast(r, tf.float32)),
                tf.convert_to_tensor(tf.cast(ns, tf.float32)),
                d
            )
        )

    def convert_memory_to_input(self):
        batch = rand.sample(self.memory, BATCH_SIZE)
        s, a_i, r, ns, d = zip(*batch)

        states = tf.convert_to_tensor(s).reshape(BATCH_SIZE, 21)
        action_indexs = tf.convert_to_tensor(a_i)
        rewards = tf.convert_to_tensor(r)
        next_states = tf.convert_to_tensor(ns).reshape(BATCH_SIZE, 21)
        dones = tf.convert_to_tensor(d)

        return states, action_indexs, rewards, next_states, dones

    def act(self, state, end):
        # if self.episode >= 0 and self.episode < 20:
        #     eps_threshold = 0.991 ** self.episode
        # else:
        #     eps_threshold = EPS_DECAY ** self.episode

        eps_threshold = 0.05 + (1 - 0.05) * math.exp(-1. * self.episode / 100)

        a_r = np.array(self.train_model(state))[0]

        if rand.random() > eps_threshold:
            a_i = np.argmin(a_r)
            while end[a_i//2] != 1:
                a_r[a_i] = np.max(a_r)
                a_i = np.argmin(a_r)
            c = 1

        else:
            a_i = rand.randint(0, 41)
            c = 0

        a = return_action(a_i)

        return a, a_i, c, eps_threshold

    def run(self):
        if len(self.memory) < TRAIN_FLAG:
            return 1

        states, action_indexs, rewards, next_states, dones = self.convert_memory_to_input()
        loss = self.learn(states, action_indexs, rewards, next_states, dones)
    
        return loss.numpy()
        
    @tf.function
    def learn(self, states, action_indexs, rewards, next_states, dones):
        q_target = self.target_model(next_states)
        target_q = rewards + (1 - dones) * GAMMA * tf.reduce_min(q_target, axis=1, keepdims=True)

        with tf.GradientTape() as tape:
            current_q = self.train_model(states) # 현재 상황에서 할 수 있는 행동들의 q value
            current_q = tf.reduce_sum(current_q[action_indexs], axis=1, keepdims=True) # 실제 한 행동에 대한 q value

            loss = self.loss_fn(current_q, target_q)

        grads = tape.gradient(loss, self.train_model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.train_model.trainable_weights))

        return loss

# %%
agent = DQN_Agent()
rewards_hist = []
st_hist = []

goal_state = set_goal('basic_formula.xlsx').reshape(1, 21)
for e in range(5000):
    counter = [0 for i in range(42)]
    state = starting_state
    steps = 0
    rewards = 0
    c = 0
    print(e)
    if e % 200 == 0:
        agent.update_model()
        print("===update===")

    while True:
        end = check_end(state, goal_state)
        action, idx, t, eps = agent.act(state, end)
        counter[idx] += 1
        c += t
        next_state = return_state(state, action)
        reward = return_reward(next_state, goal_state)

        if steps == EPISODE_DONE or all(state[0][i] == goal_state[0][i] for i in range(21)):
            done = 1
        else:
            done = 0

        agent.memorize(state, idx, reward, next_state, done)
        loss = agent.run()
        
        state = next_state
        rewards += reward
        steps += 1

        # if steps == 1:
        #     print(f'steps: {steps}, reward: {reward}, a: {idx}')

        if done:
            print(e)
            rewards_hist.append(rewards)
            st_hist.append(state)

            if e % 50 == 0:
                print(f'============={e}=============')
                print(f"rewards: {round(rewards, 3)}, net_loss: {round(loss, 3)}, number of most decision: {max(counter)}, desicion tendecy: {c}, eps: {round(eps, 5)}")

            break

agent.train_model.save('last_dqn_copy')
pd.DataFrame(st_hist.reshape(10000, 21)).to_excel('ldcp_st_hist.xlsx')

# %%
plt.plot(rewards_hist)
plt.savefig('ldcp_plot.png')

# %%



