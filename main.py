# bibliothèques nécessaires
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html, State
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objs as go
import numpy as np


# Charger le jeu de données Iris
iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])
iris["species"] = iris_raw["target_names"][iris_raw["target"]]

# Créer une application Dash
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout de l'application
app.layout = dbc.Container(
    [
        # En-tête de l'application
        html.H1("Exploration des Données Iris avec Dash"),
        html.Hr(),
        # Sélection de colonne, boutons bascules et histogramme
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="column-selector",
                        options=[
                            {"label": col, "value": col} for col in iris.columns
                        ],
                        value="sepal length (cm)",
                    ),
                    md=4,
                ),
                # Case à cocher pour la normalisation
                dbc.Col(
                    dcc.Checklist(
                        id="normalize-checkbox",
                        options=[{"label": "Normaliser la colonne", "value": "normalize"}],
                        value=[],
                    ),
                    md=4,
                ),
                # Case à cocher pour la suppression des valeurs aberrantes
                dbc.Col(
                    dcc.Checklist(
                        id="outliers-checkbox",
                        options=[{"label": "Supprimer les valeurs aberrantes", "value": "outliers"}],
                        value=[],
                    ),
                    md=4,
                ),
                dbc.Col(dcc.Graph(id="histogram"), md=12),
            ],
            align="center",
        ),
        # Section de l'application pour les prédictions
        html.Div(
            [
                html.H2("Prédiction de l'espèce Iris"),
                dcc.Input(id="input-sepal-length", type="number", placeholder="Longueur du sépale (cm)"),
                dcc.Input(id="input-sepal-width", type="number", placeholder="Largeur du sépale (cm)"),
                dcc.Input(id="input-petal-length", type="number", placeholder="Longueur du pétale (cm)"),
                dcc.Input(id="input-petal-width", type="number", placeholder="Largeur du pétale (cm)"),
                html.Button("Prédire", id="predict-button"),
                html.Div(id="prediction-output"),
                html.Div(id="accuracy-output"),
            ],
            style={"margin-top": "20px"},
        ),
    ],
    fluid=True,
)

# Entraîner un modèle de régression logistique
X = iris[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Callback pour mettre à jour l'histogramme en fonction de la colonne sélectionnée
@app.callback(
    Output("histogram", "figure"),
    Input("column-selector", "value"),
    Input("normalize-checkbox", "value"),
    Input("outliers-checkbox", "value"),
)
def update_histogram(selected_column, normalize_checkbox, outliers_checkbox):
    data = iris.copy()

    # Manipulations des données en fonction des cases cochées
    if "normalize" in normalize_checkbox:
        # Normalisation de la colonne sélectionnée
        data[selected_column] = StandardScaler().fit_transform(data[[selected_column]])

    if "outliers" in outliers_checkbox:
        # Suppression des valeurs aberrantes en fonction d'un seuil défini
        threshold = 2.0  # Seuil pour définir les valeurs aberrantes (à personnaliser)
        data = data[(np.abs(data[selected_column]) < threshold)]

    if selected_column == "species":
        # Histogramme groupé par espèces pour les variables catégorielles
        fig = px.histogram(data, x=selected_column, color="species", barmode="group", histnorm="probability density")
    else:
        # Histogramme normal pour les autres variables
        fig = px.histogram(data, x=selected_column, histnorm="probability density")

    # Personnalisation de la mise en page
    fig.update_layout(
        xaxis_title=selected_column,
        yaxis_title="Densité de probabilité",
        title=f"Distribution de {selected_column}",
        hovermode="x",
    )

    return fig


# Callback pour effectuer une prédiction d'espèce en utilisant le modèle
@app.callback(
    Output("prediction-output", "children"),
    Output("accuracy-output", "children"),
    Input("predict-button", "n_clicks"),
    State("input-sepal-length", "value"),
    State("input-sepal-width", "value"),
    State("input-petal-length", "value"),
    State("input-petal-width", "value"),
)
def predict_species(n_clicks, sepal_length, sepal_width, petal_length, petal_width):
    if n_clicks is None:
        return "", f"Précision moyenne du modèle : {accuracy:.2f}"

    if None in [sepal_length, sepal_width, petal_length, petal_width]:
        return "Veuillez entrer toutes les valeurs.", ""

    # Préparer les données d'entrée pour la prédiction
    input_data = pd.DataFrame(
        {
            "sepal length (cm)": [sepal_length],
            "sepal width (cm)": [sepal_width],
            "petal length (cm)": [petal_length],
            "petal width (cm)": [petal_width],
        }
    )

    # Effectuer la prédiction
    predicted_species = model.predict(input_data)[0]

    return f"L'espèce prédite est : {predicted_species}.", f"Précision moyenne du modèle : {accuracy:.2f}"


# Lancer l'application si ce fichier est exécuté directement
if __name__ == "__main__":
    app.run_server(debug=True)
