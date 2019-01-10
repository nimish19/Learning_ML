#submission 33

import pymongo

client = pymongo.MongoClient('localhost',27017)
my_db = client["db_University"]
collection = my_db['university_clients']
def add_client(*data):
    unique_client = my_db.university_clients.find_one({"Student Roll no": Student_Roll_no}, {"_id":0})
    if unique_client:
        return 'Client already exist'
    else:
        my_db.university_clients.insert_many(l1)
        return 'Client added successfully'
def collection(Student_Name,Student_Age,Student_Roll_no,Student_Branch):
    dict={"Student Name" : Student_Name,
                "Student Age": Student_Age,
                "Student Roll no": Student_Roll_no, 
                'Student Branch': Student_Branch}
    l1.append(dict)
    
l1=[]
for _ in range(10):
    Student_Name = input('Enter Student Name: ')
    Student_Age = input("Enter Student Age: ")
    Student_Roll_no = input("Enter Student Roll no: ")
    Student_Branch = input("Enter Student Branch: ")  
    collection(Student_Name,Student_Age,Student_Roll_no,Student_Branch)
add_client(l1)

