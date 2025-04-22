from rl_dispatch.environment.environment import DisplibEnvV1
from rl_dispatch.data_handle.data_handle import load_data_file, write_solution
from rl_dispatch.data_handle.verify import main
import time
import os


problems_path = 'data/problems/'
solutions_path = 'data/solutions/'
problem_files = [f for f in os.listdir(problems_path) if f.endswith('.json')]

max_steps = 10000  # Adjust as needed
max_time = 300 # Adjust as needed

for instance_path in problem_files:
    input_path = problems_path + instance_path
    data = load_data_file(input_path)
    env = DisplibEnvV1(data, 'cpu', verbose=False)

    # Reset the environment and render its initial state
    obs, _ = env.reset()
    env.render(mode='human')

    
    start_time = time.time()
    for i in range(max_steps):  # Arbitrary number of steps for demonstration
        print(f"Instance: {instance_path}, Step: {i}")
        action = 1
        try:
            obs, reward, done, solution = env.step(action)
        except Exception as e:
            print(f"Error occurred while processing {instance_path} at step {i}: {e}")
            break  # Restart with a different instance
        env.render(mode='human')
        if done:
            if solution:
                print(f"All nodes picked. Environment done for {instance_path}.")
                output_path = solutions_path + instance_path
                write_solution(solution, output_path)
                break
            else:
                break
        
        ran_time = time.time() - start_time

        if ran_time > max_time:  # Break if runtime exceeds 30 seconds
            print(f"Runtime exceeded {max_time} seconds for {instance_path}. Breaking.")
            break
                

    env.close()


for instance_path in problem_files:
    problem_file = problems_path + instance_path
    solution_file = solutions_path + instance_path

    if os.path.exists(solution_file):
        try:
            print(f"Verifying solution for {instance_path}...")
            main(problem_file, solution_file)
        except Exception as e:
            print(f"Verification failed for {instance_path}: {e}")

#TODO: Alter the existing to have resources
#TODO: Incorporate resources into the code so infeasible resource actions are blocked
#TODO: Add Currently picked as a picked_feature  
#TODO: Restructure to pick on node not on graph.
#TODO: Turn actions into the solution output
#TODO: Nodes are not added to Cant Pick correctly
#TODO: An agent can get stuck if it doesnt pick a node that passes its upper bound.