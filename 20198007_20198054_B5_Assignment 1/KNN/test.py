def sort_by_ind(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

x=sort_by_ind([2, 3, 1, 4, 5])
print(x)




def sorts(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][0] > sub_li[j + 1][0]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li
s = [2, 3, 1, 4, 5]
li=[]
 
for i in range(len(s)):
      li.append([s[i],i])
li = sorts(li)
print(li)
sort_index = []
 
for x in li:
      sort_index.append(x[1])
 
print(sort_index)