import numpy as np
import random
from Q_Brain_Simply import linear_Q
import TimeTag as tt

class Uniform_Memory:
    def __init__(self,memorysize, assistant):
        self.ms =memorysize
        self.assistant = assistant
        self.memory = np.zeros((self.ms, 6))
        self.memory_counter = 0

    def store_exp(self,e):
        """
        Store the experience into the memory D
        :param e: e = (s,a,r,s_,t,idx)
        :return: None
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.ms

        if self.memory_counter >= self.ms:
            tmp = self.memory[index,:]
            T = int(tmp[4])
            idx = int(tmp[5])
            self.assistant.remove_transition(T, idx)
            #print()

        self.memory[index,:] = e

        self.memory_counter += 1


class dual_memory(object):
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.cur_capacity = 0

        # learning parameters
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

        # Dual memory
        self.tt_memory = tt.timeTag(self.max_capacity)
        self.master_memory = Uniform_Memory(self.max_capacity, self.tt_memory)

# ================ Learning =====================

    def store(self, transition, time_step):
        """
        Firstly, store the transitions to time table and get back a set of time-marked transitions
        :param transition: a transition at time_step
        :param time_step: the learning time step
        :return:
        """
        # Time Tag Part
        if self.tt_memory.get_node(time_step) is False:
            # Do not exist the node of the current learning time step
            marked_t = self.tt_memory.add_node(time_step, [transition])  # input a list; return a list
            marked_t = marked_t[0]  # convert back to a transition from a list
        else:
            # Exist the node of the current learning time step
            marked_t = self.tt_memory.insert_transition(transition)  # input a transition (array); output a transition

        # Master Memory
        self.master_memory.store_exp(marked_t)

    def sample(self, n):

        if self.master_memory.memory_counter >= self.master_memory.ms:
            batchIndex = np.random.choice(self.master_memory.memory.shape[0], size=n)
            batchSample = self.master_memory.memory[batchIndex, :]
        else:
            # Memory is not full
            batchIndex =  np.random.choice(self.master_memory.memory_counter, size=n)
            batchSample = self.master_memory.memory[batchIndex, :]
        return batchIndex, batchSample

    def tid_update(self, index, new_tid):
        self.master_memory.memory[index,:] = new_tid

# ================= Adaptive =====================

    def sample_k_oldest(self, k):
        """
        Find k oldest transitions for calculating the old target error before the adjustment of memory

        The sampled transitions do not need to be pop out from the original time node

        Since there are no any update for transition at the adjustment stage

        :param k: the number of samples
        :return: a list of oldest transitions
        """
        oldest_transitions = self.tt_memory.select_k_oldest(k)
        return oldest_transitions

    def memory_enlarge(self,k):
        tmp = np.zeros((k,6))
        self.master_memory.memory = np.vstack((self.master_memory.memory, tmp))

        # update max capacity
        self.master_memory.memory_counter = self.max_capacity
        self.max_capacity = self.master_memory.memory.shape[0]
        self.master_memory.ms = self.master_memory.memory.shape[0]

    def memory_shrink(self, del_trans):

        for dt in del_trans:
            time = dt[4]
            index = dt[5]

            for i in range(self.master_memory.memory.shape[0]):
                t = int(self.master_memory.memory[i, 4])
                idx = int(self.master_memory.memory[i, 5])
                if t == time and idx == index:
                    self.master_memory.memory = np.delete(self.master_memory.memory, i, 0)
                    break
                else:
                    continue

        self.max_capacity = self.master_memory.memory.shape[0]
        self.master_memory.ms = self.master_memory.memory.shape[0]
        self.master_memory.memory_counter = self.max_capacity


    def tt_shrink(self, del_trans):
        for i in del_trans:
            time = int(i[4])
            idx = int(i[5])
            self.tt_memory.remove_transition(time, idx)
            # if all transition in the time node have been deleted (i.e. all are None)
            # it will automatically delete the time node, but the sub-map is reserved


class aER_train(linear_Q):
    def __init__(self, numState, actions, learnRate,
                 discount, greedy, init_memorySize, batchSize,
                 k_old):

        super(aER_train, self).__init__(numState, actions, greedy, learnRate, discount)

        # General parameters
        self.numState = numState - 1  # exclude terminate
        self.alpha = learnRate  # learning rate
        self.gamma = discount  # discount factor
        self.epsilon = greedy  # epsilon-greedy policy
        self.ActionSet = actions  # the action set

        # Memory
        self.memorySize = init_memorySize  # the initial memory size
        self.lowest_memory_size = init_memorySize - k_old
        self.memory = dual_memory(self.memorySize)  # initialize the memory buffer with initial memory size
        self.batchSize = batchSize  # the size of mini batch

        # Learning time Step
        self.learning_steps = 0

        # Adaptive Part
        self.select_k = k_old
        self.tmp_oldest = []  # for store the k oldest transitions temporarily
        self.last_old_td_error = 0  # the last sum of TD errors of the k oldest transitions
        self.cur_old_td_error = 0

        # Optimization
        self.gradient = 0

    def learn(self):
        batch_idx, batch_memory = self.memory.sample(self.batchSize)
        w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))

        for sample in batch_memory:
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

            w_increment += self.target_error * x

        """
        update 
        """
        self.W += self.learnRate * w_increment
        gradient = np.linalg.norm(w_increment) / self.batchSize
        print("--> Linear Q-learning's gradient: {0}\n".format(gradient))
        print("==============================================\n")

        self.learning_steps += 1

        for id, sample in zip(batch_idx, batch_memory):
            T = int(sample[4])
            idx = int(sample[5])

            # =================== For updating Time Tag =====================
            # Update the transitions to the latest time node
            # Firstly, pop out the sampled transitions from old time step and put them to the latest time node
            is_none = self.memory.tt_memory.extract_transition(T, idx)
            if is_none is not None:
                self.memory.tt_memory.remove_transition(T, idx)
            else:
                continue
            # Secondly, insert them into the newest time node
            if self.memory.tt_memory.get_node(self.learning_steps) is False:
                # Do not exist the node of the current learning time step
                #test=sample[0:4]
                new_tid = self.memory.tt_memory.add_node(self.learning_steps, [sample[0:4]])  # input a list; return a list
                new_tid = new_tid[0]
            else:
                # Exist the node of the current learning time step
                new_tid = self.memory.tt_memory.insert_transition(sample[0:4])
            # ===============================================================
            self.memory.tid_update(id, new_tid)

        return self.W, gradient, self.memory.master_memory.ms

    def adaptive_Memroy(self, k):  # TODO: the part of adjust the memory size and delete some old transitions
        """
        Adjust the memory size
        :param k: increment k or decrement k
        :return:
        """
        # Step 1: sample n oldest transitions;
        # in the same time step, sample transitions randomly
        self.tmp_oldest = self.memory.tt_memory.select_k_oldest(self.select_k)

        # Step 2: Calculate the sum of TD-error of n-oldest transitions

        self.calc_sum_error()

        # Step 3: Compare the last n-oldest TD-error with the current n-oldest TD-error
        if self.cur_old_td_error > self.last_old_td_error:
            self.memory.memory_enlarge(k)
            self.last_old_td_error = self.cur_old_td_error
            print("--> learning step {0}: Enlarging the memory".format(self.learning_steps))
        else:
            # TODO: Think: how to decided the removal transitions? Randomly choose from k-oldest set? or using F-value
            if self.memory.master_memory.ms > self.lowest_memory_size:
                removal = random.sample(self.tmp_oldest, k)
                # shrink the master mamory
                self.memory.memory_shrink(removal)
                # shrink the time tag
                self.memory.tt_shrink(removal)

                # re-sample n-oldest
                self.tmp_oldest = self.memory.tt_memory.select_k_oldest(self.select_k)
                # recompute the error
                self.calc_sum_error()  # calculate a new self.cur_old_td_error
                self.last_old_td_error = self.cur_old_td_error
                print("--> learning step {0}: Shrinking the memory".format(self.learning_steps))
            else:
                print("--> learning step {0}: The memory size is approach to the lowest memory size".format(self.learning_steps))
                self.last_old_td_error = self.cur_old_td_error

            #k_compensation = self.memory.timetag.select_k_oldest(k)

    def calc_sum_error(self):
        self.cur_old_td_error = 0  # reset to 0; re-calculate
        for t in self.tmp_oldest:
            s = int(t[0])
            a = int(t[1])
            r = int(t[2])
            s_ = int(t[3])

            xx = 0
            if s == 0:
                if a == 1:
                    xx = self.X_S_A[:, s][:, np.newaxis]
                if a == 2:
                    xx = self.X_S_A[:, s + 1][:, np.newaxis]
                q_predict = np.dot(np.transpose(xx), self.W)
            else:
                if a == 1:
                    xx = self.X_S_A[:, s * len(self.ActionSet)][:,
                         np.newaxis]  # the xx's shape is (#states * #action+1, 1)
                elif a == 2:
                    xx = self.X_S_A[:, s * len(self.ActionSet) + 1][:,
                         np.newaxis]  # the xx's shape is (#states * #action+1, 1)
                q_predict = np.dot(np.transpose(xx), self.W)

            """
            Calculate the target
            """
            if s_ == -1:
                self.target_error = r - q_predict
            else:
                max_Q, max_A = self.easy_find_max_q(s_)
                self.target_error = r + self.discountFactor * max_Q - q_predict

            self.cur_old_td_error += np.absolute(self.target_error)[0][0]