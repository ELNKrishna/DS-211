import numpy as np
import pandas as pd

df = pd.read_csv('python\\copiolet\\Sem 6\\Numerical opt\\real_estate_dataset.csv')
n_samples, n_features = df.shape

columns = df.columns

# Save the columns to a text file
np.savetxt('column_names.txt', columns, fmt='%s')


# The columns are :-
# """
# ID Square_Feet Num_Bedrooms Num_Bathrooms Num_Floors Year_Built Has_Garden Has_Pool Garage_Size Location_Score Distance_to_Center Price  
# We only need Square_feet, Garage_size, Location_score, Distance_to_centre. Target is price
# """


X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]
y = df['Price']

print(y)



# Build a linear model to predict price using X
n_samples, n_features = X.shape

# Make an array of coeffs of the size of n_features + 1 and initialize to 1
coefs = np.ones(n_features + 1)  # n + 1 to account(absorb) the bias

# Predict the price for each sample X
predictions_by_defn = X @ coefs[1:] + coefs[0]

# Append a column of 1's in X (similar to what chiru did)
# Predictions must be the same
X = np.hstack((np.ones((n_samples, 1)), X))

predictions = X @ coefs

# Confirm if both are the same
is_same = np.allclose(predictions_by_defn, predictions)

# Calculate the errors
errors = y - predictions

print(f"L2 norm of errors {np.linalg.norm(errors)}")

# Use relative error
relative_errors = errors / y

print(f"L2 norm of relative_errors {np.linalg.norm(relative_errors)}")

# Find the MSE between predictions and y
mse = np.mean(errors ** 2)
print(f"Mean Squared Error: {mse}")

# Optimization Problem: Find coefs that minimize the MSE
# This is the Least-Squares Problem

# Solve the normal equation
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# Save the coefs in a CSV file
np.savetxt('coefficients.csv', coefs, delimiter=',')

# Make predictions using coefs
predictions = X @ coefs

# Find error, relative error, and their L2 norms
errors = y - predictions
relative_errors = errors / y

print(f"New L2 norm of errors {np.linalg.norm(errors)}")
print(f"New L2 norm of relative errors {np.linalg.norm(relative_errors)}")

##################################################################################################################################

# Use all the features in the dataset to build the linear model
X = df.drop(columns=['Price'])
y = df['Price']
 
# Append ones for the intercept term
X = np.hstack((np.ones((n_samples, 1)), X))

# Solve normal equation for all features
coefs_all = np.linalg.inv(X.T @ X) @ X.T @ y

# Save the coefficients for the bigger DataFrame
np.savetxt('coefficients_all_features.csv', coefs_all, delimiter=',')

#####################################################################################################################################

# Solve the normal equation using QR decomposition
Q, R = np.linalg.qr(X)
b = Q.T @ y

print("hi5")

# Back-substitution to solve R * coefs_qr_loop = b
# coefs_qr_loop = np.zeros(n_features + 1)
# for i in range(n_features, -1, -1):
#     coefs_qr_loop[i] = (b[i] - R[i, i+1:] @ coefs_qr_loop[i+1:]) / R[i, i]

coefs_qr_loop = np.zeros(len(R)-1)
print(n_features)
for i in range(len(R)-2, -1, -1):
    if i == n_features:  # Last row
        coefs_qr_loop[i] = b[i] / R[i, i]
    else:  # Remaining rows
        # print(len(b[i] - R[i, i+1:]))
        # print(coefs_qr_loop[i+1:])
        # print(i)
        # print(len(coefs_qr_loop))
        # print (len(R[i, i+1:]))
        # print(len(coefs_qr_loop[i:]))
        coefs_qr_loop[i] = (b[i] - (R[i, i+1:] @ coefs_qr_loop[i:])) / R[i, i]


# Save the coefficients obtained using QR decomposition
np.savetxt('coefficients_qr.csv', coefs_qr_loop, delimiter=',')

#####################################################################################################################################

# Solve using Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
coefs_svd = Vt.T @ np.diag(1 / S) @ U.T @ y

# Save the coefficients obtained using SVD
np.savetxt('coefficients_svd.csv', coefs_svd, delimiter=',')
