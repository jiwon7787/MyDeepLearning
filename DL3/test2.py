import re
import keyword

def input_program():
    program_lines = []
    print("Enter the Python program to analyze, line by line. Enter 'end' to finish.")
    while True:
        line = input()
        if line == 'end':
            break
        program_lines.append(line)
    return program_lines

def print_program(program_lines):
    print("Program:")
    for line in program_lines:
        print(line)

def list_variables(program_lines, display_variables=True):
    variables = set()
    variable_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    keywords_set = set(keyword.kwlist)

    for line in program_lines:
        for match in variable_pattern.findall(line):
            if match not in keywords_set:
                variables.add(match)

    sorted_variables = sorted(variables)

    if display_variables == True:
        print("Variables:")
        for var in sorted_variables:
            print(var)

    return sorted_variables

# Convert the input variable name to snake_case
def to_snake_case(variable_name):
    # Among the variable_name, put '_' every before an uppercase letter
    # Technically, the uppercase letters were replaced with themselves and '_'
    snake_case_name = re.sub(r'([A-Z])', r'_\1', variable_name).lower()
    if snake_case_name.startswith('_'): # some variables have a name starting with '_'(e.g _epic)
        snake_case_name = snake_case_name[1:] # exclude '_'
    return snake_case_name

# Format the selected variable to snake_case
def format_variable(program_lines, variables):
    while True:
        print("Pick a variable:")
        var_name = input().strip()
        if var_name in variables:
            snake_case_name = to_snake_case(var_name)
            for i in range(len(program_lines)):
                # Search for the selected variable name in each line and replace it with the snake_case version
                program_lines[i] = re.sub(rf'\b{var_name}\b', snake_case_name, program_lines[i])
            # print(f"{var_name} has been converted to {snake_case_name}")
            break
        else:
            print("This is not a variable name.")

    return program_lines


def main():
    program_lines = input_program()

    while True:
        print("==================================")
        print("Enter your choice:")
        print("1. Print program.")
        print("2. List.")
        print("3. Format.")
        print("0. Quit.")
        print("==================================")
        choice = input().strip()

        if choice == '1':
            print_program(program_lines)
        elif choice == '2':
            variables = list_variables(program_lines, display_variables=True)
        elif choice == '3':
            variables = list_variables(program_lines, display_variables=False)
            if variables:
                program_lines = format_variable(program_lines, variables)
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")

if __name__ == "__main__":
    main()
