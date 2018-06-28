import numpy as np


class HMM:

    def __init__(self, A, B, P, O):
        # 状态转移概率矩阵
        self.state_transfer_probability = np.array(A, np.float)
        # 观测概率矩阵
        self.observation_probability = np.array(B, np.float)
        # 初始状态概率
        self.init_state_probability = np.array(P, np.float)
        # 观测序列
        self.observation_series = O
        self.observation_series_length = len(O)
        # 状态值数目
        self.state_num = self.state_transfer_probability.shape[0]
        # 观测值数目
        self.observation_num = self.observation_probability.shape[1]

    def Viterbi(self):
        optimal_road = np.zeros(self.observation_series_length, np.float)

        observation_series_max_probability = np.zeros((self.observation_series_length, self.state_num), np.float)
        last_state_bayes = np.zeros((self.observation_series_length, self.state_num), np.float)

        for state_index in range(self.state_num):
            observation_series_max_probability[0, state_index] = self.init_state_probability[state_index] * self.observation_probability[state_index, self.observation_series[0]]

        for t in range(1, self.observation_series_length):
            for i in range(self.state_num):
                temp = np.array([observation_series_max_probability[t-1, j] * self.state_transfer_probability[j, i] for j in self.state_num])
                observation_series_max_probability[t, i] = temp.max() * self.observation_probability[i, self.observation_series[t]]
                last_state_bayes[t, i] = temp.argmax()

        optimal_road[self.observation_series_length - 1] = observation_series_max_probability[self.observation_series_length-1, :].argmax()

        for i in range(self.observation_series_length-2, -1, -1):
            optimal_road[i] = last_state_bayes[i+1, optimal_road[i+1]]

        return optimal_road

    def forward(self):
        prob_distribution = np.zeros(self.observation_series_length, self.state_num)

        for i in range(self.state_num):
            prob_distribution[0, i] = self.init_state_probability[i] * self.observation_probability[i, self.observation_series[0]]

        for t in range(1, self.observation_series_length):
            for i in range(self.state_num):
                summation = 0.0
                for j in range(self.state_num):
                    summation += prob_distribution[t-1, j] * self.state_transfer_probability[j, i]
                prob_distribution[t, i] = summation * self.observation_probability[i, self.observation_series[t]]

        pro_lambda = 0.0
        for i in range(self.state_num):
            pro_lambda += prob_distribution[self.observation_series_length - 1, i]
        return pro_lambda, prob_distribution

    def backward(self):
        prob_distribution = np.zeros(self.observation_series_length, self.state_num)
        for i in range(self.state_num):
            prob_distribution[self.observation_series_length - 1, i] = 1

        for t in range(self.observation_series_length-2, -1, -1):
            for i in range(self.state_num):
                summation = 0.0
                for j in range(self.state_num):
                    summation += self.state_transfer_probability[i, j] * self.observation_probability[j, self.observation_series[t+1]] * prob_distribution[t+1, j]
                prob_distribution[t, i] = summation

        pro_lambda = 0.0

        for i in range(self.state_num):
            pro_lambda += self.init_state_probability[i] * self.observation_probability[i, self.observation_series[0]]  * prob_distribution[0, i]

        return pro_lambda, prob_distribution


















