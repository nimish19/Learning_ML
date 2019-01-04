
# Database handling using sqlite

import sqlite3
"""


# File based database ( connects if exits or creates a new one if it does not exists ) 
"""
conn = sqlite3.connect ( 'employee.db' )
"""
"""
# creating cursor
c = conn.cursor()


# STEP 1
# www.sqlite.org/datatype3.html


c.execute ("""CREATE TABLE employees(
  first  TEXT,
  last TEXT,
  pay INTEGER
  );""")
"""
"""
# STEP 2
c.execute("INSERT INTO employees VALUES ('Sylvester', 'Fernandes', 50000)")
c.execute("INSERT INTO employees VALUES ('abc', 'genocide', 500)")

c.execute("INSERT INTO employees VALUES ('def', 'dnA', 50000)")
c.execute("INSERT INTO employees VALUES ('ghi', 'Fernandes', 50000)")
c.execute("INSERT INTO employees VALUES ('jkl', 'Fernandes', 70000)")
c.execute("INSERT INTO employees VALUES ('mno', 'Fernandes', 60000)")
c.execute("INSERT INTO employees VALUES ('pqr', 'Fernandes', 40000)")
c.execute("INSERT INTO employees VALUES ('stu', 'Fernandes', 30000)")
c.execute("INSERT INTO employees VALUES ('vwx', 'Fernandes', 20000)")
c.execute("INSERT INTO employees VALUES ('yz', 'Fernandes', 10000)")

# STEP 3
c.execute("SELECT * FROM employees WHERE last = 'Fernandes' ")

""
##
c.fetchone()
c.fetchmany(5)
c.fetchall()
###
""
# returns one or otherwise None as a tuple
print ( c.fetchone()) 

# returns a list of tuples
print ( c.fetchall() )



# commits the current transaction 
conn.commit()

# closing the connection 
conn.close()
"""



# Database handling using MySQL
# conda install mysql-python
# pip install --upgrade pip 
# pip install -U setuptools
# pip install -U wheel
# pip install protobuf
# pip install mysql-connector-python-rf

import mysql.connector


# File based database ( connects if exits or creates a new one if it does not exists ) 
conn = mysql.connector.connect ( user='', password='', host='localhost' )
# database = ‘’ can be used if we want to connect to already defined database


# creating cursor
c = conn.cursor()


# STEP 1
c.execute("CREATE DATABASE mydb")

# STEP 2
c.execute(“USE mydb")

# STEP 3
c.execute ("""CREATE TABLE employees(
  first  TEXT,
  last TEXT,
  pay INTEGER
  );""")


# STEP 4
c.execute("INSERT INTO employees VALUES ('Sylvester', 'Fernandes', 50000)”)


# STEP 5
c.execute("SELECT * FROM employees")

# STEP 6
my_data_list = c.fetchall()





"""

# Database handling using MongoDB

from pymongo import MongoClient

client = MongoClient('localhost', 27017)

# create the database if does not exists
mydb = client.employees




# adding the employee in the employee collection
def add_employee(id, first, last, pay):
    unique_employee = mydb.employee.find_one({"id":id}, {"_id":0})
    if unique_employee:
        return "Employee already exists"
    else:
        mydb.employee.insert(
                {
                "id" : id,
                "first" : first,
                "last" : last,
                "pay" : pay
                })
        return "Employee added successfully"

def fetch_all_employee():
    user = mydb.forsk_clients.find_all()
    for i in user:
        print "employee information : - ", i


id = raw_input("Enter id for employee: ")
first = raw_input("Enter first name of employee: ")
last = raw_input("Enter last name of employee: ")
pay = raw_input("Enter salary of employee: ")

print add_employee(id,first,last,pay)


fetch_all_employee()
    

# Database handling using MongoDB on Cloud ( mLab )
# Steps to create DB and Account online mLab



import requests
import json


data_dict = {}

# adding the employee in the employee collection
def add_employee(id, first, last, pay):
        global data_dict
        data_dict = {                
                "id" : id,
                "first" : first,
                "last" : last,
                "pay" : pay
                }
        print "data dictionary is ", data_dict
        print add_data_to_mlab(data_dict)
        
        

res = ""
def add_data_to_mlab(data_dict):
    global res
    url = "https://api.mlab.com/api/1/databases/employees/collections/employee?apiKey=65mczz6BHJHLMxUayNO2VXNYWedu11q4"
    response = requests.post(url, json =data_dict)
    
    res = response.status_code
    if res == 200:
        print "data added successfully"
    else:
        print "response is something else:"
        print res



    
def fetch_all_employee():
    
    url = "https://api.mlab.com/api/1/databases/employees/collections/employee?apiKey=65mczz6BHJHLMxUayNO2VXNYWedu11q4"
    response = requests.get(url)
    
    res = json.loads(response.text)
    print res
    
    
    
    
id = raw_input("Enter id for employee: ")
first = raw_input("Enter first name of employee: ")
last = raw_input("Enter last name of employee: ")
pay = raw_input("Enter salary of employee: ")

print add_employee(id,first,last,pay)
fetch_all_employee()




"""