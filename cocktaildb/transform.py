import json
import os

drinks = {}

for filename in os.listdir('.'):
    if filename.endswith('.json'):
        with open(filename) as json_file:
            drink_js = json.load(json_file)
            drink_name = drink_js['strDrink']
            drink_text = ''
            if drink_js['strAlcoholic']:
                drink_text += drink_js['strAlcoholic'] + '\n'
            drink_text += drink_js['strCategory'] + '\n'
            drink_text += drink_js['strGlass'] + '\n'
            drink_text += '\n'
            for i in range(1, 16):
                if drink_js['strIngredient{}'.format(i)]:
                    drink_text += drink_js['strMeasure{}'.format(i)] + ' ' + drink_js['strIngredient{}'.format(i)] + '\n'
            drink_text += '\n'
            drink_text += drink_js['strInstructions']

            drinks[drink_name] = drink_text

with open('text2text_training_data.json', 'w') as drinks_json:
    drinks_json.write(json.dumps(drinks))
