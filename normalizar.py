import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

datos = pd.read_csv("diabetes.csv")
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

datos_std = (datos[cols] - datos[cols].mean()) / datos[cols].std()
datos_std['Outcome'] = datos['Outcome']

def dibujar_rostro(ax, fila, titulo):
    d = (fila[cols] - datos[cols].min()) / (datos[cols].max() - datos[cols].min())

    v_age      = d['Age']          # Tamaño vertical
    v_bmi      = d['BMI']          # Ancho cara
    v_glucose  = d['Glucose']      # Expresión (Boca)
    v_insulin  = d['Insulin']      # Ancho boca
    v_pedigree = d['DiabetesPedigreeFunction'] # Tamaño ojos
    v_preg     = d['Pregnancies']   # Altura ojos
    v_bp       = d['BloodPressure'] # Inclinación cejas
    v_skin     = d['SkinThickness'] # Tamaño Nariz

   # Cara 
    cara = patches.Ellipse((0.5, 0.5), 0.35 + v_bmi*0.25, 0.4 + v_age*0.25, 
                           color='#FFDBAC', ec='black', lw=1.5)
    ax.add_patch(cara)

    # Ojos
    ojo_r = 0.03 + v_pedigree*0.04
    ojo_y = 0.55 + v_preg*0.1
    ax.add_patch(patches.Circle((0.4, ojo_y), ojo_r, color='white', ec='black'))
    ax.add_patch(patches.Circle((0.6, ojo_y), ojo_r, color='white', ec='black'))

    # Nariz
    ancho_nariz = 0.02 + v_skin * 0.05
    puntos_nariz = [[0.5, 0.52], [0.5 - ancho_nariz, 0.44], [0.5 + ancho_nariz, 0.44]]
    nariz = patches.Polygon(puntos_nariz, closed=True, color='#F1C27D', ec='black', lw=1)
    ax.add_patch(nariz)

    # Boca (Glucosa)
    es_alerta = v_glucose > 0.5
    boca_y = 0.32 if es_alerta else 0.38
    boca_ang = (10, 170) if es_alerta else (190, 350)
    boca = patches.Arc((0.5, boca_y), 0.1 + v_insulin*0.2, 0.1, 
                       theta1=boca_ang[0], theta2=boca_ang[1], lw=2, color='red')
    ax.add_patch(boca)

   #Cejas
    y_ceja = ojo_y + 0.08
    ax.plot([0.3, 0.45], [y_ceja, y_ceja + (v_bp-0.5)*0.1], color='black', lw=1.5)
    ax.plot([0.55, 0.7], [y_ceja + (v_bp-0.5)*0.1, y_ceja], color='black', lw=1.5)

    ax.set_title(titulo, fontsize=7, fontweight='bold', pad=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


sanos = datos[datos['Outcome'] == 0].head(20)
enfermos = datos[datos['Outcome'] == 1].head(20)
muestra = pd.concat([sanos, enfermos])

fig, axes = plt.subplots(5, 8, figsize=(18, 11))
axes = axes.flatten()

for i, (idx, row) in enumerate(muestra.iterrows()):
    label = "x" if row['Outcome'] == 0 else "x"
    dibujar_rostro(axes[i], row, f"{label}-{idx}")




leyenda = (
    "Mapeo de Atributos: [Cara: Alto=Edad, Ancho=BMI] | [Boca: Curva=Glucosa (Triste=Alta), Ancho=Insulina]\n"
    "[Ojos: Tamaño=Pedigree, Altura=Embarazos] | [Cejas: Angulo=Presión Arterial] | [Nariz: Tamaño=Grosor Piel]"
)
plt.figtext(0.5, 0.93, leyenda, ha="center", fontsize=10, style='italic', bbox={'facecolor':'orange', 'alpha':0.1, 'pad':5})

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()