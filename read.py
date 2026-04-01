# Open the file in read mode ('r')
with open('sources.txt', 'r') as file:
    # Read the very first line
    first_line = file.readline()
    # Print it to the console
    print(first_line.strip())