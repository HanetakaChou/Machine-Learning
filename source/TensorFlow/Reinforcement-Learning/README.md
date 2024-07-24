# Reinforcement Learning  

## Markov Decision Process (MDP)  

The **state space** S is all possible **states** the **environment** can be in.  

The **state** $\displaystyle s_t \in S$ is the complete description of the **environment** at the specific step t.  

The **terminal state** $\displaystyle s_T$ is the **state** where the **episode** ends.  

The **observation space** O is all possible observations that the **agent** perceives from the **environment**. For the **fully observable environment**, the **observation space** is identical to the **state space**, namely, O = S.  

---  

The **action space** A is all possible **actions** that the **agent** can take.  

The **action** $\displaystyle a_t \in A$ is the decision or choice made by the **agent** at the specific step t.  

---  

The **transition probability** (namely, **transition function**) $\displaystyle  \mathop{\mathrm{P}}(s^{\prime} \mid s, a)$ is the probability that the environment transitions to **state** $\displaystyle s^{\prime}$ from **state** s after the **agent** takes **action** a.  

The **stochastic** (**non-deterministic**) **environment**: the **transition function** accounts for the inherent uncertainty in the **environment**. In many real-world situations, taking the same **action** a in the same **state** s does not always lead to the same next **state** $\displaystyle s^{\prime}$. The **transition function** captures this randomness by assigning probabilities to possible next **states** $\displaystyle s^{\prime}$.  

The **Markov property**: the probability of transitioning to the next **state** $\displaystyle s^{\prime}$ depends only on the current **state** s and the **action** a, and NOT on the history of any previous **states** and **actions**. In other words, the process is "memoryless": $\displaystyle \mathop{\mathrm{P}}(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = \mathop{\mathrm{P}}(s_{t+1} \mid s_t, a_t)$.  

---  

The **immediate reward** $\displaystyle r_t$ is the actual **reward** that the **agent** receives at a specific step t after taking the **action** $\displaystyle a_t$ in the **state** $\displaystyle s_t$, which can have multiple possible values, especially in the **stochastic** (**non-deterministic**) **environment**.  

The **reward function** $\displaystyle \mathop{\mathrm{R}}(s, a) = \mathbb{E} [r_t \mid s_t = s, a_t = a]$ is the **expected** **reward** that the **agent** will receive after taking the specific **action** a in the specific **state** s.

---  

The **discount factor** $\displaystyle \gamma$ is the **impatience** which determines the importance of future rewards.  

---  

The **episode** is the complete sequence of interactions between the **agent** and the **environment**, starting from the initial state $\displaystyle s_0$ and ontinuing until the **terminal state** $\displaystyle s_T$ is reached.  

The **return** (namely, **cumulative reward**) $\displaystyle G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T - t} r_T = \sum_{k = 0}^{T - t} \gamma^k r_{t+k}$ is the total sum of rewards starting from the specific state $\displaystyle s_t$ and continuing until the **terminal state** $\displaystyle s_T$ of the **episode**.     

---  

The **policy** $\displaystyle \mathop{\mathrm{\pi}} (s) = a$ is the mapping from **states** to **actions**, which specifies the **action** a that the **agent** should take in the specific **state** s. The **policy** can be **stochastic** (**non-deterministic**) $\displaystyle \mathop{\mathrm{\pi}} (a \mid s) = \mathop{\mathrm{P}} (a \mid s)$.  

The **Q-(Value) Function** (namely, the **(State) Action-Value Function**) $\displaystyle \mathop{\mathrm{Q^\pi}} (s, a) = \mathbb{E} [G_t \mid s_t = s, a_t = a, \pi]$ is the **expected** **return** (namely, **expected** **cumulative reward**) that the **agent** will receive, starting from the specific **state** s, taking the specific **action** a (once, which may NOT follow the **policy** $\displaystyle \pi$), and then following a specific **policy** $\displaystyle \pi$ thereafter.

The **Optimal Q-(Value) Function** $\displaystyle \mathop{\mathrm{Q^*}} (s, a) = \max \limits_\pi \mathop{\mathrm{Q^\pi}} (s, a)$ is the maximum **expected** **return** (namely, **expected** **cumulative reward**) that the **agent** can receive, starting from the specific **state** s, taking the specific **action** a, which is independent of any specific **policy** $\displaystyle \pi$.  

The **optimal policy** $\displaystyle \pi^* (s) = \arg \max \limits_{a} \mathop{\mathrm{Q^*}} (s, a)$ is the **policy** that for each **state** s, selects the **action** a that maximizes the **optimal Q-(Value) Function** $\displaystyle \mathop{\mathrm{Q^*}} (s, a)$.  

## Bellman Equation  

The **Q-(Value) Function** can also be expressed recursively by the **Bellman equation** $\displaystyle \mathop{\mathrm{Q^\pi}} (s, a) = \mathbb{E} [r + \gamma \mathop{\mathrm{Q^\pi}} (s^{\prime}, a^{\prime}) \mid s, a] = \mathop{\mathrm{R}}(s, a) + \gamma \sum_{s^{\prime}} \mathop{\mathrm{P}}(s^{\prime} \mid s, a) \sum_{a^{\prime}} \mathop{\mathrm{\pi}} (a \mid s) \mathop{\mathrm{Q^\pi}} (s^{\prime}, a^{\prime})$, and the recursion ends in the **terminal state** $\displaystyle \displaystyle \mathop{\mathrm{Q^\pi}} (s_T, a) = \mathop{\mathrm{R}}(s_T, a)$.  

The **Optimal Q-(Value) Function** can also be expressed recursively by the **Bellman equation** $\displaystyle \mathop{\mathrm{Q^*}} (s, a) = \mathbb{E} [r + \gamma \max \limits_{a} \mathop{\mathrm{Q^*}} (s^{\prime}, a^{\prime}) \mid s, a] = \mathop{\mathrm{R}}(s, a) + \gamma \sum_{s^{\prime}} \mathop{\mathrm{P}}(s^{\prime} \mid s, a) \max \limits_{a^{\prime}} \mathop{\mathrm{Q^*}} (s^{\prime}, a^{\prime})$, and the recursion ends in the **terminal state** $\displaystyle \displaystyle \mathop{\mathrm{Q^*}} (s_T, a) = \mathop{\mathrm{R}}(s_T, a)$.  

## Reference Environment  

[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)  

state (observation space)  

// reward  

// action  

for lunar lander https://gymnasium.farama.org/environments/box2d/lunar_lander/  

action  
0  
1  
2  
3  


reset  
current_state = env.reset()  

step  
next_state, reward, terminated, truncated, info, _ = env.step(action)  

## Deep Q-Network (DQN)  


## Epsilon-Greedy Policy (Exploitation-Exploration Tradeoff)  

The agent observes the current state and chooses an action using an  𝜖
 -greedy policy. Our agent starts out using a value of  𝜖=
  epsilon = 1 which yields an  𝜖
 -greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed state. As training progresses we will decrease the value of  𝜖
  slowly towards a minimum value using a given  𝜖
 -decay rate. We want this minimum value to be close to zero because a value of  𝜖=0
  will yield an  𝜖
 -greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the action that it believes (based on its past experiences) will maximize  𝑄(𝑠,𝑎)
 . We will set the minimum  𝜖
  value to be 0.01 and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in source we encourage you to take a look at the utils.get_action function in the utils module.

##  

off-policy algorithm  
target policy  
behabior policy  

target network  
soft update // Polyak averaging  

episode // end at terminal state  

state -> action  

reward function // only need to tell it what to do // do not need to tell it how to do  

// specify the reward function is much easier than specifying the exact right action to take from every single step  

// different values at different states  

// change state // receive reward (reward change)  

// terminal state // nothing more happens after gets reward at the terminal state  

// discount factor // reward * (discount factor)^n // n is the step  

// discount factor // \gamma // provide little impatient // get reward sooner results in high value in total return  
   
// higher discount factor -> less impatient  

// heavily discount factor -> incredibly impatient  

// financial application // dollar today better than dollay in the future  

// total return = R_1 + \gamma R_2 + \gamma^2 R_3 + ... + \gamma^{n-1} R_n // until terminal state  (no \gamma for the first state)  

// first reward is not discounted // n step 

// start off at terminal state can still have return  

// return depends on the actions you take // choose action depends on the return // with the highest return  

// negative //push out the negative reward as far into the future as possible   // financial // pay someone dollar // postpone will be better  

// policy function /pi // mapping from states to actions  

// states  
// actions  
// reward  
// discount factor \gamma  
// return  
// policy \pi // find action based on the state // \pi (s) = a  

// Markov Desicion Process (MDP) // the future only depends on the current state, and not on anything that might have occurred prior to that state // the future only depends on where your are now not on how you got here 

agent /pi // choose action a //-> environment / world // -> state s /reward R -> agent /pi 

Q-Learning  

// Q function (state action value function) Q(s, a) // start in state s, take action a (once //this action doesn't need to behave optimally), then bahave optimally (follow policy /pi // from Bellman Equation, find the maximum) after that  

// to learn the policy /pi according to the Bellman Equation  

// Q* / optimal Q function // the same meaning  

// picking action \max \limits_{a} \mathQ (s, a) // the best possible return / action   

// Bellman Equation  

// s current state  
// R(s) // reward of current state  
// a current action  
// \gamma discount factor  
// s^{\prime} state you get to after taking action a  
// a^{\prime} action that you take in state s' 

// Bellman Equation:  
// recursive form: $\displaystyle \mathop{\mathrm{Q}}(s, a) = \mathop{\mathrm{R}}(s) + \gamma \max \limits_{a^{\prime}} \mathop{\mathrm{Q}}(s^{\prime}, a^{\prime})$   

// no discount factor for current state  

// when s is terminal state ? // Q (s, a) = R (S) // no further recursive  
 
// episode ends (terminal state) // use OpenAI gym

// random (stochastic environment)  
// action may not success (may not follow the command)     

// expected return  
// mathematical expectation  
// considier the possibilit // miss step probabulity // probability of going in the wrong direction  
   
// maximize the expected return (instead of the return)  

// continuous state  

s -> state vector
a -> one-hot action vector  

[s a] = X // input to the neural network  

Q(s,a) = y // output  

// if we have that neural network // we can predict the Q(s,a) for each a at s and pick the action a that maximizes Q(s,a)  

// use Bellman Equation to get the Q(s,a) 

// take random action  
// (s^{i}, a^{i}, R(S)^{i}, S'^{i})  
// s,a -> X (input)  
// R(s), all possible of s^{\prime} -> Y (output) = R(s) + \gamma Q(s',a') 

// DQN  

// initialize neural network randomly as guess of Q(S,a)  
// train Q_new such that Q_new(S,a) \approx y (Q is similiar as the coefficent w and b in linear regression)  

// Replay Buffer // store the 10000 most recent (s^{i}, a^{i}, R(S)^{i}, S'^{i})  

// refinement: improved neural network achitecture  
// intput only the state // do not include a   
// output // vector // Q (s, a) for each a 
// more efficient // predict once and get all actions // pick the best one  

// refinement: \epsilon-greedy policy  
//
// even if Q(s,a) is not a good guess   

\epsilon-greedy policy  
// exploitation step: with probability 1 - \epsilon, pick the action a (called that greedy atcion) that maximizes Q(s, a)  
// exploration step: with probability \epsilon, pick an action a randomly // Q(s, a) randomly initialization // is actually not correct // sometimes can not convergence forever  

exploitation-exploration trade-pff

// start off at high epsilon // then gradually decrease it  

// refinement:  mini-batch  

batch learning vs mini-batch learning  

for both supervise learning and reforcement learning  

when m is large for example 100000000  

// batch learning (use all training examples)  

not use all training eamples  

m^{\prime} = 1000  
pick some subset m^{\prime} examples  // mini-batch learning  

// mini-batch gradient descent  
// each iteration // subset of the data   

// not use all examples from the replay buffer 

// compared with batch gradient descent  
// not reliably some times noisily  
// each iteration is much more computationally expensive  


// refinement: soft update  

// since mini-batch // not reliably some times noisily  
// Q_new may be worse than existing Q  
// Q = 0.01 Q_new + 0.99 Q  // converge more reliably  

// limitation  
// much easier to get to work in a simulation than a real world  
// far fewer applications than supervised or unsupervised learning  

// Deep Q-Learning Algorithm With Experience Replay  








