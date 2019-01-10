#submission 28

def check_valid(email):
    try:
        user_name,domain = email.split('@')
        industry,extension = domain.split('.')
    except Exception:
        return False
    #replace '_' and '-' with ''
    if user_name.replace('_','').replace('-','').isalnum() is False:
        return False
    elif industry.isalnum() is False:
        return False
    elif len(extension)>3 is False:
        return False
    else :
        return True
emails = []
for _ in range(int(input("No. of emails: "))):
    emails.append(input("enter email: "))
check = list(filter(check_valid, emails))
print(check)    