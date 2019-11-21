import matplotlib.pyplot as plt
with open('text.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content] 
print(content)
tariff = []
profit = []

for i in range(len(content)):
    if i%3 == 1:
        tariff.append(float(content[i][8:]))
    elif i%3 == 2:
        profit.append(abs(float(content[i][15:])))

time = range(0,len(tariff))

plt.plot(time,tariff)
plt.show()
