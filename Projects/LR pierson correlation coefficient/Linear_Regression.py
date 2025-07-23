import matplotlib.pyplot as plt

import numpy as np
import math

class LinearRegression:

    def gradient_descent(self, x, y):
        xHat = 0
        yHat = 0

        for i in range(len(x)):
            xHat += x[i]
            yHat += y[i]
        
        xHat /= len(x)
        yHat /= len(y)

        upperFunc = 0
        for j in range(len(x)):
            upperFunc += (x[j] -xHat) * (y[j] - yHat)
        lowerFunc2 = 0
        lowerFunc = 0
        for k in range(len(y)):
            lowerFunc += (x[k] - xHat ) ** 2
            lowerFunc2 += (y[k] - yHat ) ** 2
        res = math.sqrt(lowerFunc*lowerFunc2)

        r = upperFunc/res

        SDY = math.sqrt(lowerFunc2) / (len(y) -1)
        SDX = math.sqrt(lowerFunc) / (len(x) -1)

        M = r * SDY/SDX

        b = yHat - M*xHat
        return M, b
    def predict(self, m,b,x):
        lstY= []
        lstX = []
        for i in range(len(x)):
            lstY.append(m*x[i] + b)
            lstX.append(x[i])
        return lstY



x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

plt.scatter(x,y)
model = LinearRegression()
m, b = model.gradient_descent(x,y)
new_y = model.predict(m,b,x)

plt.plot(x,new_y, color='blue')
plt.show()

print(new_y)
print(f"Slope (m): {m}, Intercept (b): {b}")


        
        

      