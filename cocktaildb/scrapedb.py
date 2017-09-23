import json
import requests

for i in range(1234):
    cocktail_json = requests.get('http://www.thecocktaildb.com/api/json/v1/1/random.php').json()['drinks'][0]
    cocktail_id = cocktail_json['idDrink']
    with open('{}.json'.format(cocktail_id), 'w') as json_file:
        json_file.write(json.dumps(cocktail_json))
