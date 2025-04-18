import json


def load_data_file(file_path:str) -> dict:
    """
    Load a data file from the specified path and return the data as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None

def read_objective(data:dict) -> list[dict]:
    """
    Extracts objective operations and unique trains from the given data.
    Args:
        data (dict): A dictionary containing 'objective' key with a list of operations.
    Returns:
        list: A list of dictionary items where each dictionary item is an objective component
    """
    
    objective_operations = data['objective']
    
    return objective_operations

def write_solution(solution: dict, filename: str):
    """
    Write the generated solution to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(solution, file, indent=2)

def compute_objective(data: dict, events: list) -> int:
    """
    Compute the objective value based on events and provided criteria.
    This function now consistently looks for 'objective' in data.
    """
    objective_components = data.get('objective', [])
    objective_value = 0

    for component in objective_components:
        if component['type'] == 'op_delay':
            train_id = component['train']
            operation_id = component['operation']
            threshold = component.get('threshold', 0)
            increment = component.get('increment', 0)
            coeff = component.get('coeff', 0)

            for event in events:
                if event['train'] == train_id and event['operation'] == operation_id:
                    delay = max(0, event['time'] - threshold)
                    objective_value += coeff * delay + increment * int(delay > 0)
                    break

    return objective_value


def sort_events(events: list) -> list:
    """
    Sort the events by time first and then by train.
    """
    return sorted(events, key=lambda e: (e['time'], e['train']))