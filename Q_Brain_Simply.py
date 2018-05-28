
import numpy as np
import pandas as pd



class linear_Q():

    def __init__(self, numState,ActionSet,greedy,learnRate,discountFactor,Max_episode=None):
        """initial Value"""
        self.numState = numState - 1  # 减去terminal
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.Max_episode = Max_episode
        self.W = np.random.uniform(0,0.1,size=self.numState*len(self.ActionSet)+1)[:,np.newaxis]  # 初始化W (#states * #action+1, 1)
                                                                                                  # 加1是bias
        self.X_S_A = np.identity(self.numState*len(self.ActionSet)+1) # #states * I(#action+1 , #states * #action+1)
        self.target_error = 0
        """ parameter matrix """
        self.para_matrix = self.create_para_matrix()

    def create_para_matrix(self):
        para_matrix = pd.DataFrame(dtype=np.float64)
        return para_matrix

    def new_X(self, X, W):
        if str(X) not in self.para_matrix.index:
            new_state = pd.Series(data= W,
                                name=str(X))        # the name is the index name of the brain

            """
            append the new state
            """
            self.para_matrix = self.para_matrix.append(new_state) # 一定要以赋值形式返回Q table（self.brain）
            #print(self.para_matrix)

    def choose_action(self,state):
        random_choose = np.random.uniform()  # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
        # low default: 0 ; high default: 1

        """Act non-greedy action or all value of action is 0 -- random choose action"""
        if (random_choose > self.greedy):
            action = np.random.choice(self.ActionSet)
            return action

        """Act greedy acction """
        if (random_choose < self.greedy):
            q, action = self.easy_find_max_q(state)
            return action

    def test_choose_action(self, w, S_next):
        if S_next == 0:
            w_ = w[0:len(self.ActionSet),:]
        else:
            w_ = w[S_next * len(self.ActionSet):S_next * len(self.ActionSet) + 2, :]

        maxQ = w_.max()
        maxA = np.where(w_ == maxQ)[0][0]
        maxA += 1

        return maxA

    def easy_find_max_q(self,S_next):
        if S_next == 0:
            w = self.W[0:len(self.ActionSet),:]
        else:
            w = self.W[S_next*len(self.ActionSet):S_next*len(self.ActionSet)+2, :]

        maxQ = w.max()
        maxA = np.where(w == maxQ)[0][0]
        maxA += 1

        return maxQ, maxA

    def getIndicator(self, s, a):
        """
        Find the corresponding indicator from 1-hot matrix(X_S_A) according to the s and a
        :param s: the state
        :param a: the action
        :return: an indicator
        """
        x = 0
        if s == 0:
            if a == 1:
                x = self.X_S_A[:, s][:, np.newaxis]
            if a == 2:
                x = self.X_S_A[:, s + 1][:, np.newaxis]
        else:
            if a == 1:
                x = self.X_S_A[:, s * len(self.ActionSet)][:,
                    np.newaxis]  # the x's shape is (#states * #action+1, 1)
            elif a == 2:
                x = self.X_S_A[:, s * len(self.ActionSet) + 1][:,
                    np.newaxis]  # the x's shape is (#states * #action+1, 1)

        return x

    def calDyn_ER(self, memory, batchSize):
        """
        Calculate the dynamics of the parameters of standard ER case
        dw = m/N * alpha * integrate( td-error * gradient(Q) )
        :param memory: the whole memory buffer
        :param batchSize: the size of mini-batch memory
        :return: the dynamics of parameters: dw
        """
        memory_size = memory.shape[0]
        gdSum = 0

        def tdTarget(self, s_, r):
            if s_ == -1:
                td_target = r
            else:
                max_Q, _ = self.easy_find_max_q(s_)
                td_target = r + self.discountFactor * max_Q
            return td_target

        def dynFunction(self, t0):
            transition0 = memory[t0, :]

            # Transition 0
            s0 = int(transition0[0])
            a0 = int(transition0[1])
            r0 = int(transition0[2])
            s_0 = int(transition0[3])
            x0 = self.getIndicator(s0, a0)

            # Predicted Q-value
            q0 = np.dot(np.transpose(x0), self.W)

            # TD target
            td_target0 = tdTarget(self, s_0, r0)

            # TD error
            td_error_0 = td_target0 - q0

            return td_error_0

        for i in range(memory_size):
            gdSum += dynFunction(self, i)

        dw = (memory_size / batchSize) * self.learnRate * gdSum
        return dw[0][0]

    def batch_linear_train(self, memory, batchSize):
        """Training part"""

        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        #save_path = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_LFA.png".format(self.numState)
        """
        plotter = LossAccPlotter(title="Loss of Linear FA with {0} states".format(self.numState),
                                 save_to_filepath=None,
                                 show_acc_plot=False,
                                 show_plot_window=False,
                                 show_regressions=False,
                                 LearnType="LFA"
                                 )
        """

        batchIndex = np.random.choice(memory.shape[0], size=batchSize)
        batchSample = memory[batchIndex, :]
        w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))
        param_dynamics = 0
        td_error = 0
        for sample in batchSample:
            x = 0
            s = int(sample[0])
            a = int(sample[1])
            r = int(sample[2])
            s_ = int(sample[3])

            x = self.getIndicator(s, a)

            q_predict = np.dot(np.transpose(x), self.W)

            """
            Calculate the target
            """
            if s_ == -1:
                self.target_error = r - q_predict
            else:
                max_Q, max_A = self.easy_find_max_q(s_)
                self.target_error = r + self.discountFactor * max_Q - q_predict
            td_error += self.target_error
            w_increment += self.target_error * x

        """
        update 
        """
        self.W += self.learnRate * w_increment

        gradient = np.linalg.norm(w_increment) / batchSize
        td_error = td_error / batchSize
        print("--> Linear Q-learning's gradient: {0}\n".format(gradient))
        print("==============================================\n")

        return self.W, gradient, td_error[0][0]


class Memory:
    def __init__(self,memorysize):
        self.ms =memorysize
        self.memory = np.zeros((self.ms, 4))
        self.memory_counter = 0

    def store_exp(self,e):
        """
        Store the experience into the memory D
        :param e: e = (s,a,r,s_)
        :return: None
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.ms
        self.memory[index,:] = e

        self.memory_counter += 1


if __name__ == '__main__':
    N_STATES = 26  # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes









