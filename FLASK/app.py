#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


from flask import Flask, url_for
from flask import render_template
from flask import request
from flask import make_response
import json

from configs.settings import CONFIG
from modules.synonyms import add_synonyms

#---------------
print('Creating sim docs...')
from modules.simdocs import *
simdocs = SimDocs(use_stemmer=True)
simdocs.load_docs()
simdocs.load_dict_corpus()
simdocs.create_model()
#---------------
class Error(Exception):
	pass
#---------------

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET', 'POST'])
def title_page():
	return "<h2> Title page </h2>"

@app.route('/sim/', methods=['GET', 'POST'])
@app.route('/sim/<passwd>/', methods=['GET', 'POST'])
def sim_page(passwd=''):

	print('request.path:', request.path)
	print('request.method:', request.method)
	input_text = request.form.get('input_text')
	password = request.form.get('password')
	passkey = request.form.get('passkey')
	answer_type = request.form.get('answer_type')	
	lang = request.form.get('lang')
	model_type = request.form.get('model')

	if output['result']:
		#return "'answer': '{}'".format(output['result'][0])
		content = json.dumps(json_list, ensure_ascii=False)
		resp = make_response(content, 200)
		resp.headers['Content-type'] = 'application/json'
		return resp


if __name__ == "__main__":	

	#app.run(debug=True, host='185.72.146.131', port=80)
	app.run(debug=True, host=CONFIG['host'], port=int(CONFIG['port'])) # the port number should be changed
