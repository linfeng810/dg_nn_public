import re

# Define the regular expression pattern to match lines starting with "nits" followed by numbers (integers, floats, or scientific notation)
pattern = r'nits.*?([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
# r'^nits ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
number_pattern = "[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?"

fout = open('turek_fix_bug_log.csv', 'w')

# Open the text file for reading
with open('/data/linfeng/aicfd/z01-dg_nn/z05-ns/turek_fix_bug_log.txt', 'r') as file:
    for line in file:
        # Use regular expression to find matching lines
        match = re.match(pattern, line)
        if match:
            numbers = re.findall(number_pattern, line)
            if numbers:
                # Convert the extracted strings to floats
                numbers = [float(num[0]+num[-1]) for num in numbers]
                print(f'Found numbers: {numbers}')
                fout.write(f'{numbers}\n')

fout.close()
