from rl_dispatch.environment.environment import DisplibEnvV1
from rl_dispatch.data_handle.data_handle import load_data_file



input_path = 'data/toy_files/problems/line3_1.json'
data = load_data_file(input_path)
env = DisplibEnvV1(data, 'cpu',verbose=False)

# Reset the environment and render its initial state
obs, _ = env.reset()
env.render(mode='human')

num_steps = 10000  # Adjust as needed
for _ in range(num_steps):  # Arbitrary number of steps for demonstration
    print("\nChoose an action:")
    print("0 - Progress time step")
    print("1 - Pick current index feasible node")
    print("2 - Skip to next feasible node")
    try:
        action = int(input("Enter action (0,1,2): "))
    except ValueError:
        print("Invalid input. Defaulting to action 0.")
        action = 0
        
        
    obs, reward, done, _ = env.step(action)
    env.render(mode='human')
    if done:
        print("All nodes picked. Environment done.")
        break

env.close()

#TODO: Alter the existing to have resources
#TODO: Incorporate resources into the code so infeasible resource actions are blocked
#TODO: Add Currently picked as a picked_feature  
#TODO: Restructure to pick on node not on graph.
#TODO: Turn actions into the solution output
#TODO: Nodes are not added to Cant Pick correctly
#TODO: An agent can get stuck if it doesnt pick a node that passes its upper bound.