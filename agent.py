import openai
from apikey import api_key

# Configurez votre clé API OpenAI
openai.api_key = api_key

def generer_reponse(requete):
    try:
        # Utilisez l'API Complétion de OpenAI pour générer une réponse
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Utilisez le bon ID de modèle
            prompt=requete,
            temperature=0.7,
            max_tokens=150
        )
        # Vérifiez que la réponse contient les données attendues
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['text'].strip()
        else:
            return "Erreur : Réponse inattendue de l'API"
    except openai.error.OpenAIError as e:
        return f"Erreur lors de la génération de la réponse: {e}"

def create_agent(dataframe):
    # Votre logique pour créer un agent avec le dataframe
    return dataframe

def query_agent(agent, query):
    response = generer_reponse(query)
    return response
