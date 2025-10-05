import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
titanic = sns.load_dataset('titanic')

# Configurar el tamaño de los gráficos
plt.figure(figsize=(15, 5))

# ============================================
# GRÁFICO 1: ¿Cuántos sobrevivieron?
# ============================================
plt.subplot(1, 3, 1)  # 1 fila, 3 columnas, posición 1

# Contar cuántos sobrevivieron y cuántos no
survival_counts = titanic['survived'].value_counts()

# Crear gráfico de barras
plt.bar(['No Sobrevivió', 'Sobrevivió'], 
        survival_counts.values, 
        color=['#e74c3c', '#2ecc71'])

# Poner el número encima de cada barra
for i, valor in enumerate(survival_counts.values):
    plt.text(i, valor + 10, str(valor), 
             ha='center', fontweight='bold', fontsize=12)

plt.title('Supervivencia General', fontsize=14, fontweight='bold')
plt.ylabel('Cantidad de Pasajeros')

# ============================================
# GRÁFICO 2: Supervivencia por Sexo
# ============================================
plt.subplot(1, 3, 2)  # 1 fila, 3 columnas, posición 2

# Crear una tabla cruzada: sexo vs supervivencia
sex_survival = pd.crosstab(titanic['sex'], titanic['survived'])

# Crear gráfico de barras agrupadas
sex_survival.plot(kind='bar', 
                  ax=plt.gca(), 
                  color=['#e74c3c', '#2ecc71'], 
                  rot=0)

plt.title('Supervivencia por Sexo', fontsize=14, fontweight='bold')
plt.xlabel('Sexo')
plt.ylabel('Cantidad de Pasajeros')
plt.legend(['No Sobrevivió', 'Sobrevivió'])
plt.xticks([0, 1], ['Mujer', 'Hombre'])

# ============================================
# GRÁFICO 3: Supervivencia por Clase
# ============================================
plt.subplot(1, 3, 3)  # 1 fila, 3 columnas, posición 3

# Crear tabla cruzada: clase vs supervivencia
class_survival = pd.crosstab(titanic['pclass'], titanic['survived'])

# Crear gráfico de barras agrupadas
class_survival.plot(kind='bar', 
                    ax=plt.gca(), 
                    color=['#e74c3c', '#2ecc71'], 
                    rot=0)

plt.title('Supervivencia por Clase Social', fontsize=14, fontweight='bold')
plt.xlabel('Clase del Ticket')
plt.ylabel('Cantidad de Pasajeros')
plt.legend(['No Sobrevivió', 'Sobrevivió'])
plt.xticks([0, 1, 2], ['1ª Clase', '2ª Clase', '3ª Clase'])

# Ajustar espaciado
plt.tight_layout()

# Mostrar los gráficos
plt.show()

# ============================================
# CALCULAR PORCENTAJES
# ============================================
print("=" * 60)
print("📊 TASAS DE SUPERVIVENCIA")
print("=" * 60)

# Supervivencia general
survival_rate = titanic['survived'].mean() * 100
print(f"\n🌍 GENERAL: {survival_rate:.1f}% sobrevivió")

# Por sexo
print("\n👥 POR SEXO:")
for sex in titanic['sex'].unique():
    rate = titanic[titanic['sex'] == sex]['survived'].mean() * 100
    print(f"   {sex.capitalize()}: {rate:.1f}%")

# Por clase
print("\n🎫 POR CLASE:")
for pclass in sorted(titanic['pclass'].unique()):
    rate = titanic[titanic['pclass'] == pclass]['survived'].mean() * 100
    print(f"   Clase {pclass}: {rate:.1f}%")

print("\n" + "=" * 60)