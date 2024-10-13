from termcolor import colored
from models import AgentState, CodeReviewResult

def pretty_print_state_enhanced(agent_state: AgentState):
    print('-' * 50)
    print(colored('Agent State:', 'blue'))
    # print('-' * 50)
    for key, value in agent_state.items():

        if isinstance(value, CodeReviewResult):
            result = value.result  
            message = value.message  
            color_key = 'green' if result == 'correct' else 'red'
            print(colored(f'{key}:', 'cyan'))
            print(f'{colored("Result:", "cyan")} {colored(result, color_key)}')
            print(f'{colored("Message:", "cyan")}: {colored(message, "yellow")}')
            print('-' * 50)  # Add a separator for clarity

        elif isinstance(value, str) and '\n' in value:
            print(colored(f'{key}:', 'cyan'))
            print(colored(f'{value}', 'yellow'))

        else:
            # Default print for single-line key-value pairs
            print(colored(f'{key}:', 'cyan') + colored(f' {value}', 'yellow'))

    print('-' * 50)
