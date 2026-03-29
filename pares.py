import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv("diabetes.csv")
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

datos_std = (datos[cols] - datos[cols].mean()) / datos[cols].std()
datos_std['Outcome'] = datos['Outcome']

pares = [
    ('Glucose', 'BMI'),
    ('Glucose', 'Age'),
    ('Insulin', 'Glucose'),
    ('BMI', 'SkinThickness'),
    ('Age', 'Pregnancies'),
    ('Glucose', 'DiabetesPedigreeFunction')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (var_x, var_y) in enumerate(pares):
    sns.scatterplot(
        data=datos_std, x=var_x, y=var_y, hue='Outcome', 
        palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=axes[i]
    )
    axes[i].set_title(f"{var_y} vs {var_x}", fontweight='bold')

plt.suptitle("Analisis de Pares de Caracteristicas (Diabetes vs No Diabetes)", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()