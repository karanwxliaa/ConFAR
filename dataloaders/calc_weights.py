from dataloaders.weights import calc_weights

weights = calc_weights(["AU"])
print("The Weighted matrix is: \n\n\n", [round(weight, 3) for weight in weights])
