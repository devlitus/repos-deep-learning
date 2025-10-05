import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
titanic = sns.load_dataset('titanic')

# Eliminar los que no tienen edad para este análisis
titanic_con_edad = titanic.dropna(subset=['age'])

# Calcular tasa de supervivencia por rangos de edad
print("=" * 60)
print("📊 SUPERVIVENCIA POR EDAD")
print("=" * 60)

# Crear grupos de edad
titanic_con_edad['age_group'] = pd.cut(titanic_con_edad['age'], 
                                        bins=[0, 12, 18, 35, 60, 100],
                                        labels=['Niños (0-12)', 'Adolescentes (13-18)', 
                                               'Adultos Jóvenes (19-35)', 'Adultos (36-60)', 
                                               'Mayores (60+)'])

# Calcular tasa de supervivencia por grupo
for grupo in titanic_con_edad['age_group'].cat.categories:
    tasa = titanic_con_edad[titanic_con_edad['age_group'] == grupo]['survived'].mean() * 100
    cantidad = len(titanic_con_edad[titanic_con_edad['age_group'] == grupo])
    print(f"{grupo}: {tasa:.1f}% sobrevivió (n={cantidad})")

# Gráfico
plt.figure(figsize=(10, 5))

# Gráfico 1: Distribución de edad por supervivencia
plt.subplot(1, 2, 1)
titanic_con_edad[titanic_con_edad['survived']==0]['age'].hist(bins=30, alpha=0.5, label='No Sobrevivió', color='red')
titanic_con_edad[titanic_con_edad['survived']==1]['age'].hist(bins=30, alpha=0.5, label='Sobrevivió', color='green')
plt.xlabel('Edad')
plt.ylabel('Cantidad')
plt.title('Distribución de Edad por Supervivencia')
plt.legend()

# Gráfico 2: Tasa de supervivencia por grupo de edad
plt.subplot(1, 2, 2)
survival_by_age = titanic_con_edad.groupby('age_group', observed=False)['survived'].mean() * 100
survival_by_age.plot(kind='bar', color='steelblue', rot=45)
plt.ylabel('% de Supervivencia')
plt.title('Tasa de Supervivencia por Grupo de Edad')
plt.axhline(y=38.4, color='red', linestyle='--', label='Promedio General')
plt.legend()

plt.tight_layout()
plt.show()

# Comparar: edad promedio de sobrevivientes vs no sobrevivientes
edad_sobrevivientes = titanic_con_edad[titanic_con_edad['survived']==1]['age'].mean()
edad_no_sobrevivientes = titanic_con_edad[titanic_con_edad['survived']==0]['age'].mean()

print("\n" + "=" * 60)
print("📊 EDAD PROMEDIO")
print("=" * 60)
print(f"Sobrevivientes: {edad_sobrevivientes:.1f} años")
print(f"No sobrevivientes: {edad_no_sobrevivientes:.1f} años")
print(f"Diferencia: {abs(edad_sobrevivientes - edad_no_sobrevivientes):.1f} años")