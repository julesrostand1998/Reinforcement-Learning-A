============================================ ASSIGNMENT 2 READ ME - Jules Rostand ========================================

The environment.py contains the code to create the basketball environment, and in the policy_gradient_learning.py there is
the code to run the Monte-Carlo REINFORCE policy-gradient algorithm on a priorly initiated domain.

The environment is currently set for the smallest field assigned to us (W=9, H=6, nb_opp=5), at the very beginning of the
 policy_gradient_learning.py code, you can also change those values. 

The environment is set as open AI Gym environments. We use classic control from which we import rendering to visualize
our training episodes. In the policy_gradient_learning.py algorithm, I added time.sleeps in the main function to better 
visualize the steps taken by the agent, that can be modified as well for faster execution. 

From the environment step function, we also output several values that we use to analyse our model's behavior and make
adjustments (see report). In the policy learning code, there are also several codes put in comments that served for creating
the plots. All of this is detailed in the code comments as well as the report. 

Steps to run the code:
1. Make sure that you have torch and gym installed on your machine
2. cd on your directory containing environment.py and policy_gradient_learning.py
3. Run the following command: python3 policy_gradient_learning.py
