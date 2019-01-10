from pymongo import MongoClient
from datetime import datetime
#import json

client = MongoClient('localhost', 27017)
mydb = client.db_Forsk

def add_client(company_name, ceo_name, ceo_email, ceo_number):
    unique_client = mydb.forsk_clients.find_one({"CEO Email":ceo_email}, {"_id":0})
    if unique_client:
        return "Client already exists"
    else:
        mydb.forsk_clients.insert_many(
                {
                "Company Name" : company_name,
                "CEO Name" : ceo_name,
                "CEO Email" : ceo_email,
                "CEO Number" : ceo_number,
                "Date-Time" : datetime.now()
                })
        return "Client added successfully"

def view_client(ceo_email):
    user = mydb.forsk_clients.find_one({"CEO Email":ceo_email}, {"_id":0})
    if user:
        company = user["Company Name"]
        ceo = user["CEO Name"]
        email = user["CEO Email"]
        number = user["CEO Number"]
        time = user["Date-Time"]
        return {"Company Name":company,"CEO Name":ceo,"CEO Email":email,"CEO Number":number}
    else:
        return "Sorry, No such user exists"


company =input("Enter Company Name: ")
ceo_name =input("Enter CEO Name: ")
ceo_email = input("Enter CEO Email: ")
ceo_number = input("Enter CEO Number: ")

print (add_client(company,ceo_name,ceo_email,ceo_number))

user =input("Enter CEO Email to find: ")
user_data = view_client(user)

print (user_data)