import requests
#r = requests.post("http://87.236.23.209:5001/sim/", 
r = requests.post("http://127.0.0.1:5001/sim/",
	data={'answer_type':'json', 'input_text': 'психоз шизофрения', 'passkey': '+91(4l3Uyoh:<iBl',
		'model':'lsi', 'num_topics':'5'})
#r = requests.post("http://127.0.0.1:8888/sim/", 
#	data={'password': 12524, 'input_text': 'issue', 'answer_type':'json'})
print('text:', r.text)
print('status:', r.status_code)
print('reason:', r.reason)
print('headers:', r.headers)


if __name__ == "__main__":

	r = requests.post("http://87.236.23.209:5001/sim/",
		data={'answer_type':'json', 'input_text': 'психоз шизофрения', 'passkey': '+91(4l3Uyoh:<iBl',
			'model':'lsi', 'num_topics':'5', 'synonyms':'true', 'synonyms_threshold':'0.7',
			'w2v_model_num': '3'})
	print('text:', r.text)
	print('status:', r.status_code)
	print('reason:', r.reason)
	print('headers:', r.headers)	

	r = requests.post("http://87.236.23.209:5001/sim/",
		data={
			'answer_type':'json', 
			'input_text': 'психоз шизофрения', 
			'passkey': '+91(4l3Uyoh:<iBl',
			'model':'lsi', 
			'num_topics':'5', 
			#'synonyms':'false', 
			#'synonyms_threshold':'0.7',
			#'w2v_model_num': '3',
			})
	print('text:', r.text)
	print('status:', r.status_code)
	print('reason:', r.reason)
	print('headers:', r.headers)