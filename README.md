## Computing on Wheels: A Deep Reinforcement Learning-Based Approach
Computing on Wheels: A Deep Reinforcement Learning-Based Approach

Official repository for the paper implementation of Computing on Wheels: A Deep Reinforcement Learning-Based Approach published at the IEEE Transactions on Intelligent Transportation Systems Journal.

### Abstract
Future generation vehicles equipped with modern technologies will impose unprecedented computational demand due to the wide adoption of compute-intensive services with stringent latency requirements. The computational capacity of the next generation vehicular networks can be enhanced by incorporating vehicular edge or fog computing paradigm. However, the growing popularity and massive adoption of novel services make the edge resources insufficient. A possible solution to overcome this challenge is to employ the onboard computation resources of close vicinity vehicles that are not resource-constrained along with the edge computing resources for enabling tasks offloading service. In this paper, we investigate the problem of task offloading in a practical vehicular environment considering the mobility of the electric vehicles (EVs). We propose a novel offloading paradigm that enables EVs to offload their resource hungry computational tasks to either a roadside unit (RSU) or the nearby mobile EVs, which have no resource restrictions. Hence, we formulate a non-linear problem (NLP) to minimize the energy consumption subject to the network resources. Then, in order to solve the problem and tackle the issue of high mobility of the EVs, we propose a deep reinforcement learning (DRL) based solution to enable task offloading in EVs by finding the best power level for communication, an optimal assisting EV for EV pairing, and the optimal amount of the computation resources required to execute the task. The proposed solution minimizes the overall energy for the system which is pinnacle for EVs while meeting the requirements posed by the offloaded task. Finally, through simulation results, we demonstrate the performance of the proposed approach, which outperforms the baselines in terms of energy per task consumption.

### Visual aspect of the problem

<img 
 style="text-align: center;"
  src="https://github.com/user-attachments/assets/0849a995-6145-4dec-8f9f-efa9464a76a2">

</img>

### Installation
```
$ git clone git@github.com:bayegaspard/Actor_Critic_RL_CoW.git
$ cd Actor_Critic_RL_CoW
$ pip install -r requirements.txt
```

Cite
```
@article{kazmi2022computing,
  title={Computing on wheels: A deep reinforcement learning-based approach},
  author={Kazmi, SM Ahsan and Ho, Tai Manh and Nguyen, Tuong Tri and Fahim, Muhammad and Khan, Adil and Piran, Md Jalil and Baye, Gaspard},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={11},
  pages={22535--22548},
  year={2022},
  publisher={IEEE}
}
```

