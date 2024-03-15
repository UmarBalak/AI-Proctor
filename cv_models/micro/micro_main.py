from micro_functions import *

while True:
    text = listen()
    # print(text)

    with open("paper.txt", "r") as file:
    # Read the content of the file
        paper = file.read()

    similarity = compare_files(text, paper)
    print(f"Similarity : {similarity:.2f}%")