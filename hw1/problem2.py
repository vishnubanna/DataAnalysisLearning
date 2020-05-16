#!/usr/bin/python3

# A client has come to you asking how to perform version control operations with git on their file 'git_commands.py' so that they can work on and submit files to receive credit for a class. To help them, you will fill in variable values so that print statements print out correct output to show them what commands they need to enter.
#You'll be provided some instructions to accomplish the tasks/questions the client has for you.


#We define a variable called 'cmd' to hold the string 'git'
cmd='git'


# For the rest of the assignment, YOU MUST ASSIGN VALUES TO VARIABLES BY INDEXING THE LIST BELOW.
# DO NOT JUST TYPE OUT SOMETHING LIKE "' add '" FROM THE LIST.
# Using the commands in the list below, print out the answers to the following questions:
commands=[' (replace this) ',' add ',' clone ',' commit -m ', ' pull ',' push ']

####################
print('Question 1: What command should I use to download my assignment from GitHub?')
#Set variable 'cmd1' to the list variable that best indicates your answer.
#eg: cmd1=commands[pickTheCorrectNumber]
#Note that, unlike MATLAB, Python indexing begins at 0

#Fill in the line below:
cmd1= commands[2]
print(cmd + cmd1 + 'https://github.com/johndoe/ConfusedStudent.git')


#Define a variable called 'filename' that holds the string for the client's file, 'git_commands.py' that they need work done with.
#We have provided a variable to hold the filename, which will be used in the following questions.
filename ='git_commands.py'

####################

print('Question 2: If `git_commands.py` is a new working file that I`ve just created on my local machine, then what commands should I use to submit it to the remote repository?')
#Step 1: Stage the file for the next version
#Step 2: Create a new version (update local repository)
#Step 3: Update the remote repository

#Fill in the variable assignments below with appropriate choices from the list from earlier
#Step 1
#Set variable 'cmd1' to the list variable that best indicates your answer
cmd1= commands[1]
print(cmd + cmd1 + filename)

#Step 2
#Set variable 'cmd2' to the list variable that best indicates your answer
cmd2=commands[3]
print(cmd + cmd2+ "'First version of file is done. I should pay my consultant.'")

#Step 3
#Set variable 'cmd3' to the list variable that indicates your answer
cmd3=commands[5]
print(cmd + cmd3)

####################

print('Question 3: If I modified `git_commands.py` on my local machine, then what commands should I use to update the file on the remote repository?')
#Step 1: Stage the file for the next version
#Step 2: Create a new version (update local repository)
#Step 3: Update the remote repository

#Fill in the variable assignments below with appropriate choices from the list from earlier:
#Step 1
#Set variable 'cmd1' to the list variable that best indicates your answer
cmd1=commands[1]
print(cmd + cmd1 + filename)

#Step 2
#Set variable 'cmd2' to the list variable that best indicates your answer
cmd2=commands[3]
print(cmd + cmd2+ "'Problem fixed and file updated. My consultant is cool and deserves $20.'")

#Step 3
#Set variable 'cmd3' to the list variable that indicates your answer
cmd3=commands[5]
print(cmd + cmd3)

####################

print('Question 4: If `git_commands.py` has been removed from my local machine, then what commands should I use to remove it from the remote repository?')
#Step 0: Assume that you have deleted the file referenced by filename.
#Step 1: Stage the removal of the file
#Step 2: Create a new version
#Step 3: Update the repository

#Fill in the variable assignments below with appropriate choices from the list from earlier:
#Step 1
#Set variable 'cmd1' to the list of variable that indicates your answer
cmd1=commands[1]
print(cmd + cmd1 + filename)

#Step 2.
#Set variable 'cmd2' to the list variable that best indicate your answer
cmd2=commands[3]
print(cmd + cmd2+ "'Removed File from Repository.'")

#Step 3
#Set variable 'cmd3' to the list variable that indicate your answer
cmd3=commands[5]
print(cmd + cmd3)

####################

print('Question 5: If the remote repository on Github has new changes that are not on my local machine, what command should I use to update my local machine?')
#Fill in the line below by setting variable 'cmd1' to the list variable that best indicates your answer
cmd1=commands[4]
print(cmd + cmd1 )
