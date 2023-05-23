#!/usr/bin/env python
# coding: utf-8

# In[16]:


################################################################################
# Project 4 #
# Run hspice to determine the tphl of a circuit #
################################################################################
import numpy as np # package needed to read the results file
import subprocess # package needed to lauch hspice
import shutil # package needed to copy a file
################################################################################
# Start the main program here. #
# VARIABLES: #
# tphl: Propagation Time Delay High to Low #
################################################################################
# Create the node points of the circuit
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y','z']
# Define a perfect time propagation of 100ns
perfect_tphl = 100
for fan in range(2,15):
     # Create loop for each node, step size is 2 because you have to go to the
     # next inverter which has two new nodes
     # Write each value into the line file
    for stage in range(1,15,2):

         # Copy the header file into the InvChain file
        shutil.copy('header.sp', 'InvChain.sp')

         # We are going to open 'InvChain.sp' file and the 'a' means we are
         # going to append the file
         # 'w' means write to the file, 'r' means to read the file
        inv = open('InvChain.sp', "a")

         # Append the first line in 'InvChain.sp' to the following below
        stringFan = '.param fan = ' + str(fan)
        inv.write(stringFan)

 # For loop to determine what node you are dealing with
        for inverter in range(stage):
 # Are we at the first node?
            if(inverter == 0):
                firstNode = '\nXinv0 '+ str(nodes[0]) + ' ' + str(nodes[1])+ 'inv M=1'

             # Write the line with the following pointer
                inv.write(firstNode)


             # Are we at a node in between of the chain?
             # If inv = 2:
             # InverterNum fromNode toNode fanValue
             # betweenNode = Xinv2 2 3 fan^2
             # Write each value into the line file
            elif(inverter > 0 and inverter < stage):
                betweenNode = '\nXinv' + str(inverter) + ' ' +str(nodes[inverter])+' ' + str(nodes[inverter+1])+ ' inv M='+ str(fan**inverter)

 # Write the line with the following pointer
                inv.write(betweenNode)

 # Are we at a the last node of the chain?
            elif(inverter == len(stage)):
                endNode = '\nXinv'+ str(inverter) + ' '+ str(nodes[inverter])+ ''+ str(nodes[25])+ ' inv M=' + str(fan**inverter)

 # Write the line with the following pointer
                inv.write(endNode)

 # Write the command .end into the hspice file
                inv.write('\n.end')

 # Close the file
                inv.close()

 # Launch hspice. Note that both stdout and stderr are captured so
 # they do NOT go to the terminal!
            proc = subprocess.Popen(["hspice","InvChain.sp"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            output, err = proc.communicate()
 # extract tphl from the output file
            data = np.recfromcsv("InvChain.mt0.csv",comments="$",skip_header=3)
            tphl = data["tphl_inv"]
            if(tphl < perfect_tphl):
                tphl = perfect_tphl
                bestStageCount = stage
                bestFanCount = fan
print("The overrided InvChain.sp file after performing 'python project4.py' ")
inv = open('InvChain.sp', 'r')
print(inv.read())
print('\n')
print("The best combination includes ", bestStageCount, "number of inverters and", bestFanCount, "number of fans.")
print("\n")


# In[ ]:




