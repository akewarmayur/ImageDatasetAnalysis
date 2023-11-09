# import pandas as pd
# import glob
# df = pd.DataFrame(columns=["image_name"])
# # import re
# #
# # def atoi(text):
# #     return int(text) if text.isdigit() else text
# #
# #
# # def natural_keys(text):
# #     return [atoi(c) for c in re.split(r'(\d+)', text)]
# #
# # imli = []
# # for fi in glob.glob("data/*"):
# #     na = fi.split("/")[1]
# #     print(na)
# #     imli.append(na)
# # imli.sort(key=natural_keys)
# #
# # for i in imli:
# #     le = len(df)
# #     df.loc[le] = [i]
# # df.to_csv("yy.csv")
#
# classes = [
#            "Alcohol use",
#            "Drugs use",
#            "Smoking drugs",
#            "LGBTQ Content",
#            "Expressions of Nudity",
#            "Expressions of Sexuality",
#            "Expressions of Profanity",
#            "Expressions of Hate Speech",
#            "Expressions of Violence",
#            "Conversational Information",
#            "Factual Information",
#            "Opinions and Beliefs",
#           "Emotional Content",
#            "Narratives and Stories",
#            "Instructions and Directions",
#            "Questions and Answers",
#            "Problem Solving",
#            "Conflict and Resolution",
#            "Cultural and Societal Insights",
#            "Entertainment and Humor",
#            "Technical and Specialized Knowledge",
#            "Language and Linguistic Analysis",
#            "Personal Information",
#            "Cultural Expressions",
#            "Sarcasm and Irony",
#            "Hesitation and Uncertainty",
#            "Agreements and Disagreements",
#            "Intention and Purpose",
#            "Social Relationships",
#            "Expressions of Gratitude",
#            "Expressions of Apology",
#            "Learning and Education",
#            "Negotiation",
#            "Influence and Manipulation",
#            "Authority and Power Dynamics",
#            "Motivation",
#            "Politeness and Rudeness",
#            "Greetings and Farewells",
#            "Rituals and Traditions",
#            "Public Speaking",
#            "Affirmation and Confirmation",
#            "Criticism and Feedback",
#            "Expressions of Agreement",
#            "Speculation and Hypotheticals",
#            "Nostalgia",
#            "Abstract Concepts and Philosophical Discussions",
#            "Health and Wellness Information",
#            "Environment and Weather",
#            "Travel and Experiences",
#            "Technology and Innovation",
#            "Product and Service Recommendations",
#            "Legal and Ethical Discussions",
#            "Creative and Artistic Expression"
#            ]
# # for i in classes:
# #     le = len(df)
# #     df.loc[le] = [i]
# # df.to_csv("yy.csv")
#
# # df = pd.read_csv("tt.csv")
# # print(df.columns)
# # res = {}
# # df1_grouped = df.groupby('imagePath')
# #
# # # iterate over each group
# # res = {}
# # for group_name, df_group in df1_grouped:
# #     res[group_name] = df_group['facePath'].tolist()
# # print(res)
#
# rr = ["ball", "ball", "others", "others"]
# my_list = [x for x in rr if x != "others"]
# zz = (list(set(my_list)))
# print(zz)

# list_of_objects = []
# with open("objectFile.txt", 'r') as file:
#     lines = file.readlines()
#
# for line in lines:
#     list_of_objects.append(line.strip())  #
#
# print(list_of_objects)
# import os
# for filename in os.listdir("/home/manu/PycharmProjects/DatasetAnalysis/d7/"):
#     print(filename)

import pandas as pd

df = pd.read_csv("Results1/Faces.csv")
print(df.columns)

df1 = pd.read_csv("Results/DataAnalysis.csv")
ss = df1["image_name"].tolist()
others = {}
for s in ss:
    if s in df['FramesPath'].values:
        print(f"{s} is present in the column.")
    else:
        print(f"{s} is not present in the column^^^^^^^^^^^^^^^^^^^.")
        others[s] = "others"


print(others)