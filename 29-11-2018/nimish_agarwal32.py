#submission 32

import re
data = input("Enter card numbers: ")
correct = re.findall(r'([456]\d{15})|([456]\d{3}\-\d{4}\-\d{4}\-\d{4})', data)
print (correct[0][0])
