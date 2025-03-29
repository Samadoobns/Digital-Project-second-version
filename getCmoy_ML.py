from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display



X = pd.read_csv('C:/Users/samad/OneDrive/Bureau/ml_2/Dataset_numerique_20000_petites_machines.csv', sep=';')
y = X.pop('Cmoy')
X_test = pd.read_csv('C:/Users/samad/OneDrive/Bureau/ml_2/Dataset_numerique_10000_petites_machines.csv', sep=';')
y_test = X_test.pop('Cmoy')
print(X.columns)
print(X.head())


print("train set dim",X.shape)

# Liste des modèles à tester
regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0),
    "HistGradient Boosting": HistGradientBoostingRegressor(max_iter=100, random_state=0),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=0),
    '''"SVR (RBF)": SVR(kernel='rbf', C=10, epsilon=0.1),'''
    "KNeighbors": KNeighborsRegressor(n_neighbors=5),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
}


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='most_frequent')
# Bundle preprocessing for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X.columns)])


my_pipline = [ Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)]) for model in regressors.values() ]

scoree = {}
importances = {}

for piplinee in my_pipline:
    model_name = next(k for k, v in regressors.items() if v == piplinee.named_steps['model'])

    print(f"\n➡️ Training model: {model_name}")
    with tqdm(total=1, desc=f"Fitting {model_name}", unit="model") as pbar:
        piplinee.fit(X, y)
        pbar.update(1)

    preds = piplinee.predict(X_test)
    score = r2_score(y_test, preds)

    scoree[model_name] = score
    print(f"✅ Score of model {model_name}: {score:.4f}")

    try:
        perm = PermutationImportance(piplinee.named_steps['model'], random_state=42)
        perm.fit(piplinee.named_steps['preprocessor'].transform(X_test), y_test)
        weights = eli5.explain_weights_df(perm, feature_names=X.columns.tolist())
        importances[model_name] = weights
        display(weights.head(10))  # top 10
    except Exception as e:
        print(f"⚠️ Importance non dispo pour {model_name} : {e}")
    


# Tri des scores du meilleur au moins bon
sorted_scores = dict(sorted(scoree.items(), key=lambda item: item[1], reverse=True))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# **1. Affichage des R² Scores**

# Graphique des scores R²
ax[0].barh(list(sorted_scores.keys()), list(sorted_scores.values()), color='skyblue')
ax[0].set_xlabel("R² Score")
ax[0].set_title("Performance des régressions")
ax[0].invert_yaxis()  # Le meilleur modèle en haut
for i, (model, score) in enumerate(sorted_scores.items()):
    ax[0].text(score + 0.01, i, f"{score:.4f}", va='center')

# **2. Affichage des Importances des Features dans une même figure**
# Nombre de modèles avec importance disponible
n_models = len(importances)

# Déterminer la grille de subplots (par exemple 2 lignes si plus de 3 modèles)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()  # Pour un accès facile

for idx, (model_name, importance_df) in enumerate(importances.items()):
    ax = axes[idx]
    top_features = importance_df.head(10)
    ax.barh(top_features['feature'], top_features['weight'], color='lightgreen')
    ax.set_title(f"{model_name}")
    ax.invert_yaxis()
    ax.set_xlabel("Poids")

# Supprimer les axes vides s'il y en a
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Top 10 Features Importantes par Modèle", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Pour ne pas couper le titre
plt.show()