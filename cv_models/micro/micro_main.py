from micro_functions import *

# while True:
#     # text = listen()
#     # print(text)
#     text = "what is DSA, define machine learning, explain merge sort, how to deploy a machine learning model"
#     paper = "what is dsa, define machine learning, explain merge sort, how to deploy a machine learning model"

#     similarity = compare_files(text, paper)
#     print(f"Similarity : {similarity:.2f}%")

text = "what is DSA, define machine learning, explain merge sort, how to deploy a machine learning model"
# paper = "what is dsa, define machine learning, explain merge sort, how to deploy a machine learning model"
paper = "what is MACHINE LEANING 1236521"

result = cosine(text, paper.lower())
print(result)