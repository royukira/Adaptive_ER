# Adaptive_ER
## What are problems about standard ER and prioritized ER
  * The memory size of memory is a main factor affecting the reinforcement learning process.
  * Non-monotonic effect on the learning rate of both ERs
  * Too much or too little memory both slow down the rate of learning
  * An improper memory size causes redundant problem in pER
  * Both versions of ER exist overshooting problem.
  
## What is `Adaptive (memory) Experience Replay`?
A simple `adaptive experience replay (aER)`, an algorithm proposed for this problem recently, controls the size of memory comparing `the current TD-error of n-oldest transitions` with `the last one`. Obviously, the algorithm’s idea is quite intuitive and easy-to-understand, which treats TD-error as a criterion of the performance of learning and adjust the size of memory buffer based on the criterion. However, the trouble of implementation is that there is lack of memory structures storing the information of time step. In other words, the question is `how to define a transition’s age`. For solving the problem, I introduced a dual memory system for improving the aER algorithm.  

![](https://github.com/royukira/Adaptive_ER/blob/master/photo/algorithm.png)

## What is `Dual Memory Structure`?
`The dual memory system` consists of two single memory structures: `master memory` and `assistant memory`. Both memory structures serve for different roles. The master memory is an adaptive memory which in charge of sampling the transitions for learning. The assistant memory is the `Time-Tag memory` I proposed that takes charge of the adjustment stage. It mainly helps the agent to control the size of the master tree according to the performance of learning.

## What is `Time-Tag memory`?
In the aER algorithm and the apER algorithm I proposed, there are many operations for transitions which is highly based on the learning time step. For example, in the adjustment part of aER, we need to select n oldest transitions. The issue is how do we determine a transition that whether it is oldest. The structure of memory buffer utilized in standard ER is a sample FIFO stack. Besides, the sumTree structure used in priority experience replay only contain the information of priorities of transitions. Hence, the current memory structure is unfriendly for some algorithm which rely on the time information of transitions.

The TT memory is assembled with 3 data structure: hash table (the hash map), linear table (the sub-map item), and node (time chains)

![](https://github.com/royukira/Adaptive_ER/blob/master/photo/submap.png)  
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/hashmap.png)  
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/time_node.png)  
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/tt.png)

## `Using Dual Memory Structure On aER`

* Learning performance [total training steps] (`Left:aER with dual memory`,`right:ER and prioritized ER`)
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/result.png)

* Learning performance [Error]
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/result2.png)

* Change of memory size
![](https://github.com/royukira/Adaptive_ER/blob/master/photo/result3.png)

# FUTURE WORK
## apER
The detail of apER is discussed on the `part 6` of my report 
