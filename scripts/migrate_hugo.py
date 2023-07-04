import sys
import re
import os

def insert_at_index(string, substring, index):
    return string[:index] + substring + string[index:]

def process_markdown_file(input_file_path):
    with open(input_file_path, 'r') as input_file:
        content = input_file.read()

    start = 0
    while True:
        start = content.find("$", start)



    # Replacing ${text}$ with `${text}$`
    content = re.sub(r'\${(.*?)}\$', r'`$\1$`', content)

    # Replacing $${text}$$ with `$${text}$$`
    content = re.sub(r'\${2}(.*?)\${2}', r'`$\1$`', content)

    # Generate output file path
    output_file_path = os.path.splitext(input_file_path)[0] + "_new.md"

    with open(output_file_path, 'w') as output_file:
        output_file.write(content)


# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    process_markdown_file(input_file_path)
