#this question was asked to me in an interview and  i failed to answer it 
#properly. I found it after the interview was done
#this program asks used to enter the desired no. of names and prints them in Ascending Order

NumList = []

Number = int(input("Please enter the Total Number of List Elements: "))
for i in range(1, Number + 1):
    value = input("Please enter the Value of %d Element : " %i)
    NumList.append(value)

NumList.sort()

print("Element After Sorting List in Ascending Order is : ", NumList)