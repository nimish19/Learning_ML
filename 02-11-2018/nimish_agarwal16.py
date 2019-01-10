#Submission 16
text = input("Sting: ")

def translate(a):
    vowels = 'a,e,i,o,u'
    newText=''
    for i in text:
        #checks if char is not vowel or whitespace i.e., Consonants
        if(i not in vowels and i!=' '):
            #Concatenate consonant i to double i and 'o' in-between
            newText += i + 'o' + i
        else:
            newText += i
    return newText

print(translate(text))            