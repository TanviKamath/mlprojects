import dill

with open('artifacts/model.pkl', 'rb') as f:
    model = dill.load(f)

print(type(model))  # ✅ should show <class 'sklearn.ensemble._forest.RandomForestRegressor'> or similar

