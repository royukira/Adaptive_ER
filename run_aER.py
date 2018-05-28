import aER
import numpy as np
import Simply_Teasure_Game as stg
from matplotlib import pyplot as plt

def greedy_increment(greedy, total_learning_step, incrementRate=0.00004, greedy_max=0.9):
    """
    self-increment greeedy : epsilon += incrementRate / (epsilon + c)^2

    :param incrementRate: global increment rate  /  default: 0.001
    :param greedy_max: the max greedy  /  default: 0.95
    :return: None
    """
    if total_learning_step == 0:
        greedy = 0.1
    elif greedy < greedy_max:
        greedy += incrementRate / np.square(greedy + 0.00001)  # 0.00001 ensures the denominator will not be 0
    else:
        greedy = greedy_max
    return greedy


def train_with_play(numState, RL, init_capacity=100):
    Cstep = 5
    Kstep = 30
    greedy = greedy_increment(0,0)
    total_steps = 0
    steps = []
    episodes = []
    gradient = []
    gdSteps = []
    ms_ = []
    msSteps = []
    dws = []
    dSteps = []
    errors = []
    eSteps = []
    #memory = apER.apER_Memory(init_capacity)
    # this is only for ER; pER has a specific memory(function) in priorityTrain()
    for i_episode in range(20):
        s=0
        stg.update_env(s, i_episode, total_steps, numState-1)
        while True:
            # env.render()
            random_choose = np.random.uniform()

            if random_choose < greedy:
                q,a = RL.easy_find_max_q(s)
            else:
                a = np.random.choice(RL.ActionSet)

            S_Next, R = stg.get_env_feedback(s, a, numState-1)

            transition = np.hstack((s, a, R, S_Next))

            # store
            RL.memory.store(transition, RL.learning_steps)

            is_learn = total_steps % Cstep
            is_adjust = total_steps % Kstep

            if (total_steps > init_capacity) and (is_learn == 0):
                # Learning part
                print("--> Step {0} : Adaptive - Experience Replay...".format(total_steps))
                _, gd, ms = RL.learn()
                gradient.append(gd)
                gdSteps.append(total_steps)
                ms_.append(ms)
                msSteps.append(RL.learning_steps)

                # Adjustment part
                if is_adjust == 0:
                    print("--> Step {0} : Starting adjust the size of memory".format(total_steps))
                    RL.adaptive_Memroy(Kstep)
                    print("--> Step {0} : The size of memory: {1}".format(total_steps, RL.memory.max_capacity))

            if R == 1:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break
            else:
                s = S_Next
            #stg.update_env(s, i_episode, total_steps, numState)
            total_steps += 1
            greedy = greedy_increment(greedy,total_steps)

    epi_step = np.vstack((episodes, steps))
    gradient_step = np.vstack((gdSteps, gradient))
    ms_step = np.vstack((msSteps,ms_))

    """
    if len(dws) != 0:  # TESTING
        dw_step = np.vstack((dSteps, dws))
        return epi_step, gradient_step, dw_step

    if len(errors) !=0:
        error_step = np.vstack((eSteps, errors))
        return epi_step, gradient_step, error_step
    """
    return epi_step, gradient_step, ms_step


if __name__ == '__main__':

    """Setting"""
    N_STATES = 40  # the length of the 1 dimensional world, #states + terminate
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / (N_STATES - 1))  # discount factor

    batchSize = 60
    memorysize = 150

    rl = aER.aER_train(numState=N_STATES,actions=ACTIONS,learnRate=ALPHA,
                         discount=EPSILON,greedy=GAMMA,init_memorySize=memorysize,
                         batchSize=batchSize,k_old=30)

    step, gd, ms= train_with_play(RL=rl, numState=N_STATES, init_capacity=memorysize)

    plt.figure(1)
    plt.plot(step[0, :], step[1, :], c='r', label='Linear Q-learning with aER')
    plt.legend(loc='best')
    plt.ylabel('Total training step')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(gd[0, :], gd[1, :], c='r', label='Linear Q-learning with aER')
    plt.legend(loc='best')
    plt.ylabel('Gradient')
    plt.xlabel('Total training steps')
    #plt.ylim(0, 0.08)
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(ms[0, :], ms[1, :], c='r')
    plt.legend(loc='best')
    plt.ylabel('Memory Size')
    plt.xlabel('Learning steps')
    # plt.ylim(0, 0.08)
    plt.grid()
    plt.show()