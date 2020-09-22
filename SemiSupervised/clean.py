#!/usr/bin/env python
# coding: utf-8
import sys
with sys.stdin as f:
    with sys.stdout as w:
        for line in f:
            rightside = ""
            leftside = ""
            right = False
            rightskips = 0
            for i, c in enumerate(line):
                if c == "\n":
                    break
                if right == True:
                    if rightskips<2:
                        rightskips+=1
                        continue
                if c == "|":
                    if line[i+1] == "|" and line[i+2] == "|":
                        right = True
                else: 
                    if not right:
                        leftside+=c
                    else:
                        rightside+=c
            if rightside == " " or rightside == "" or leftside == " " or leftside == "":
                continue
            else:
                w.write(line)
   
