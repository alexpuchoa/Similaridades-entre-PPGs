'''
Modificado em 2/3/2021
Autor: Alexandre Prestes Uchoa
'''
import re
import pandas as pd

from IPython import get_ipython

from pathlib import Path
relative_path = Path(__file__).parent

pastaWP				= {}
pastaWP['main'] 	= relative_path
pastaWP['arqs'] 	= pastaWP['main']
pastaWP['modelos'] 	= relative_path / 'modelos/'
pastaWP['estudos']	= relative_path / 'estudos/'
pastaWP['bow']		= relative_path / 'bow/'
pastaWP['tfidf']  	= relative_path / 'tfidf/'
pastaWP['hdp']  	= relative_path / 'hdp/'
pastaWP['lsi']  	= relative_path / 'lsi/'
pastaWP['lda']  	= relative_path / 'lda/'
pastaWP['tflow']	= relative_path / 'tflow/'
pastaWP['kmeans'] 	= relative_path / 'kmeans/'
pastaWP['plots']  	= relative_path / 'plots/'
pastaWP['corpus'] 	= relative_path / 'corpus/'
pastaWP['wiki']  	 = relative_path / '../WIKIPEDIA DATADUMPS/'

pasta			= {}
pasta['main'] 	= relative_path.as_posix() + '/'
pasta['arqs'] 	= pasta['main']
pasta['modelos']= relative_path.as_posix() + '/modelos/'
pasta['estudos']= relative_path.as_posix() + '/estudos/'
pasta['bow']	= relative_path.as_posix() + '/bow/'
pasta['tfidf']  = relative_path.as_posix() + '/tfidf/'
pasta['hdp']  	= relative_path.as_posix() + '/hdp/'
pasta['lsi']  	= relative_path.as_posix() + '/lsi/'
pasta['lda']  	= relative_path.as_posix() + '/lda/'
pasta['tflow']	= relative_path.as_posix() + '/tflow/'
pasta['kmeans'] = relative_path.as_posix() + '/kmeans/'
pasta['plots']  = relative_path.as_posix() + '/plots/'
pasta['corpus'] = relative_path.as_posix() + '/corpus/'
pasta['wiki']   = relative_path.as_posix() + '/../WIKIPEDIA DATADUMPS/'


print('** Capes NLP Modules loaded **')

def load_magic():
	ipython = get_ipython()
	return ipython


def auto_reload():
	ipython = get_ipython()
	ipython.magic("%load_ext autoreload")
	ipython.magic("%autoreload 2")


def load_nlp():
	import spacy
	return spacy.load('en_core_web_lg', disable=['ner'])


def load_mongo(bd=None):
	from pymongo import MongoClient
	#from pymongo.errors import BulkWriteError
	#from bson.objectid import ObjectId
	conn_mongo = MongoClient(bd)
	return conn_mongo.scopus


def _set_fname(params):
	if type(params) is not list:
		params = [params]
	params_new = []
	for p in params:
		if type(p) is list:
			params_new.append('-'.join([str(e) for e in p]))
		else:
			if p is not None and p != '':
				params_new.append(p)
	nome = '%s_' * len(params_new)
	nome = nome[:-1]
	return nome % tuple(params_new)


def latin1_to_ascii(texto):
	'''--------------------------------------------------------------------
	This replaces UNICODE Latin-1 characters with
	something equivalent in 7-bit ASCII. All characters in the standard
	7-bit ASCII range are preserved. In the 8th bit range all the Latin-1
	accented letters are stripped of their accents. Most symbol characters
	are converted to something meaningful. Anything not converted is deleted.
	=======================================================================
	Codigo original:
	http://stackoverflow.com/questions/930303/python-string-cleanup-manipulation-accented-characters
	r = ''
	for i in unicrap:
		if xlate.has_key(ord(i)):
			r += xlate[ord(i)]
		elif ord(i) >= 0x80:
			pass
		else:
			r += i
	-----------------------------------------------------------------------
	'''

	if repr(texto) == 'None':
		print('texto vazio')
		return texto

	xlate = {
		0xc0:'A', 0xc1:'A', 0xc2:'A', 0xc3:'A', 0xc4:'A', 0xc5:'A',
		0xc6:'Ae', 0xc7:'C',
		0xc8:'E', 0xc9:'E', 0xca:'E', 0xcb:'E',
		0xcc:'I', 0xcd:'I', 0xce:'I', 0xcf:'I',
		0xd0:'Th', 0xd1:'N',
		0xd2:'O', 0xd3:'O', 0xd4:'O', 0xd5:'O', 0xd6:'O', 0xd8:'O',
		0xd9:'U', 0xda:'U', 0xdb:'U', 0xdc:'U',
		0xdd:'Y', 0xde:'th', 0xdf:'ss',
		0xe0:'a', 0xe1:'a', 0xe2:'a', 0xe3:'a', 0xe4:'a', 0xe5:'a',
		0xe6:'ae', 0xe7:'c',
		0xe8:'e', 0xe9:'e', 0xea:'e', 0xeb:'e',
		0xec:'i', 0xed:'i', 0xee:'i', 0xef:'i',
		0xf0:'th', 0xf1:'n',
		0xf2:'o', 0xf3:'o', 0xf4:'o', 0xf5:'o', 0xf6:'o', 0xf8:'o',
		0xf9:'u', 0xfa:'u', 0xfb:'u', 0xfc:'u',
		0xfd:'y', 0xfe:'th', 0xff:'y',
		0xa1:'!', 0xa2:'{cent}', 0xa3:'{pound}', 0xa4:'{currency}',
		0xa5:'{yen}', 0xa6:'|', 0xa7:'{section}', 0xa8:'{umlaut}',
		0xa9:'{C}', 0xaa:'{^a}', 0xab:'<<', 0xac:'{not}',
		0xad:'-', 0xae:'{R}', 0xaf:'_', 0xb0:'{degrees}',
		0xb1:'{+/-}', 0xb2:'{^2}', 0xb3:'{^3}', 0xb4:"'",
		0xb5:'{micro}', 0xb6:'{paragraph}', 0xb7:'*', 0xb8:'{cedilla}',
		0xb9:'{^1}', 0xba:'{^o}', 0xbb:'>>',
		0xbc:'{1/4}', 0xbd:'{1/2}', 0xbe:'{3/4}', 0xbf:'?',
		0xd7:'*', 0xf7:'/'
	}
	r = ''
	i = 0
	while i < len(texto):
		char = ord(texto[i])
		if i+5 < len(texto):
			if texto[i:i+2] == '\\x':
				print('i:', i, '0+texto[i+1:i+4]:', '0'+texto[i+1:i+4])
				try:
					 char = eval('0'+texto[i+1:i+4])
				except:
					 char = '0xe1'
				i=i+3

		if char in xlate.keys():
			r += xlate[char]
		else:
			r += texto[i]

		i+=1

	return r
