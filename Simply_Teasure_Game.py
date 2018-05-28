"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

The initial position is random in my version.

"""
import time

def update_env(S, episode, step_counter, numState):
    env_list = ['-'] * numState + ['T']  # example:'---------T' our environment
    if S == -1:
        interaction = '==> Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        #time.sleep(2)
        #print('\r                                ', end='')
    else:
        env_list[S] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.3)
        return interaction


def get_env_feedback(S, A,numState):
    """
    The rule of rewarding
    :param S: Now State
    :param A: Action
    :return: Next state S_ (according to what Action it take), Reward R
    """
    # This is how agent will interact with the environment
    if A == 1:  # right move
        if S == numState - 1:  # terminate; 因为 numState-1 是 Treasure 的 position
            S_ = -1    # 因此再向右移一个就是terminal = -1
            R = 1              # 到达terminal即找到treasure，获得奖励 R=1
            return S_, R
        else:
            S_ = S + 1         # 如果没到terminal，根据 Action 到 Next State ， 往右移
            R = 0              # 无奖励
            return S_, R
    elif A==2:  # wrong move
        R = 0                  # 因为这个环境中 Treasure是在最右边 所以不管怎么往左移 都是无奖励 R=0

        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1         # 往左移

        #S_ = random.randint(-1,S)
        #S_ = -1

        return S_, R