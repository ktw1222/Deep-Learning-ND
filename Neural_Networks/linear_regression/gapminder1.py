import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Mak a prediction using the model
laos_life_exp = bmi_life_model.predict(21.07931)


# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_values, y_values)

# print(model.predict([ [127], [248] ]))
# [[ 438.94308857, 127.14839521]]
