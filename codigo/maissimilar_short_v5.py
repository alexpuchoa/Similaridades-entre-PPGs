'''
Criado em 1/3/2021

Derivado de maissimilar_v5

Modificações:
	Não utiliza modelo NLP
	Utiliza vetores armazenados em HDF5

Por Alexandre Uchoa
'''
# -*- coding: utf-8 -*-
import re
import h5py
import numpy as np
import pandas as pd

from time import time


from capes_nlp import pasta, pastaWP
pasta_main_WP 	= pastaWP['main']
pasta_main		= pasta['main']
pasta_modelos 	= pasta['modelos']
pasta_estudos 	= pasta['estudos']

#------------------------------------------------------------------------------
# Determina a similaridade entre tags de entidades
# a partir de um modelo já gerado 
#------------------------------------------------------------------------------
class MaisSimilar_Short(object):

	def __init__(self, tipo_producao='teses', \
					   modelo_fn=None):
		if tipo_producao is None:
			print('\n** FORNECA UM TIPO DE PRODUCAO (teses, topicos, artigos)')
		#self.AA = AreaAvaliacao()
		self.pasta_modelos 	= pasta_modelos
		self.pasta_estudos 	= pasta_estudos
		self.tipo_producao 	= tipo_producao
		self.usa_hdf5 		= True

		self.load_modelo_hdf5(modelo_fn=modelo_fn)


	# Carrega arquivos HDF5 e NPY ao inves do modelo d2v inteiro
	def load_modelo_hdf5(self, modelo_fn=None):
		#print("\n*** Carregando vetores de HDF5")

		if modelo_fn is None:

			if self.tipo_producao.startswith('tese'):
				modelo_fn = "0_d2v-teses_model.mm"

			elif self.tipo_producao == 'artigos_analise':
				modelo_fn = '7_d2v-mix-artigos-analise_model.mm'

			elif self.tipo_producao.startswith('artigo'):
				modelo_fn = "0_d2v_artigos-entidades_model.mm"

			elif self.tipo_producao.startswith('topico'):
				modelo_fn = "0_d2v-teses-topicos_model.mm"

			else:
				modelo_fn = "0_d2v-teses-topicos_model.mm"

			print("*** Assumindo modelo:", modelo_fn)

		else:
			if modelo_fn[-3:] != '.mm': modelo_fn += '.mm'

		# Carrega somente arquivos com indices e vetores DOCVECS
		self.docvecs_dv_fn 	= modelo_fn + ".docvecs.vectors_docs.hdf5"
		index_dv_fn 	= modelo_fn + ".docvecs.index_docs.npy"
		self.docvecs_wv_fn 	= modelo_fn + ".wv.vectors.hdf5"
		index_wv_fn 	= modelo_fn + ".wv.index.npy"

		# Cria DF com indices dos vetores DOCVECS e tags
		self.d2v_index = pd.DataFrame(np.load(pasta_modelos + index_dv_fn),\
									  columns=['tags'])
		self.d2v_index.set_index('tags',inplace=True,drop=True)
		self.d2v_index = self.d2v_index.assign(pos=range(len(self.d2v_index)))

		# Cria DF com indices dos vetores de WV e tags
		self.d2v_wv_index = pd.DataFrame(np.load(pasta_modelos + index_wv_fn),\
										 columns=['tags'])
		self.d2v_wv_index.set_index('tags',inplace=True,drop=True)
		self.d2v_wv_index = self.d2v_wv_index.assign(pos=range(len(self.d2v_wv_index)))

		# Abre HDF5 com vetores docvecs
		h5py_f1 = h5py.File(pasta_modelos + self.docvecs_dv_fn, 'r')
		self.d2v_docvecs = h5py_f1['docvecs']
		h5py_f2 = h5py.File(pasta_modelos + self.docvecs_wv_fn, 'r')
		self.d2v_wordvecs = h5py_f2['wv']



	def _tags_cosine_v2(self, tag1, df_tags2, primeiros=20, sim_minima=.0):
		# Separa os docvecs correspondentes
		if tag1 not in df_tags2.index:
			df_tags2 = df_tags2.append(self.d2v_index.loc[tag1], \
									   ignore_index=False)
			df_tags2.sort_values('pos', inplace=True)

		np_so_docvecs = self.d2v_docvecs[df_tags2['pos'].to_list()]

		vec_tag1 = np_so_docvecs[df_tags2.index == tag1]
		#vec_tag2 = self.d2v_docvecs[df_tags2['pos'].to_list()]
		vec_tag2 = np_so_docvecs

		cosine = self.cosine(vec_tag1[0], vec_tag2)

		# Aproveita TAGS (de areas) de vecs
		if primeiros == 0:
			primeiros = len(df_tags2)

		# ordena decrescente e retorna somente os # primeiros casos
		df_tags2['SIMILARIDADE'] = cosine
		df_tags2.sort_values('SIMILARIDADE', ascending=False, inplace=True)

		# Se tiver sido fornecida uma similaridade minima (piso), ela eh usada...
		if sim_minima is not None:
			return df_tags2[df_tags2['SIMILARIDADE'] >= sim_minima][:primeiros]
		else:
			return df_tags2[:primeiros]


	# Calcula similaridade entre 2 ou mais vetores por cosseno
	def cosine(self, vec1, vecs):
		# Cosine similarity function with NumPy
		# Se vec2 forem multiplos vetores...
		if vecs.shape[0] > 1:
			return np.dot(vecs, vec1) / (np.linalg.norm(vec1) * \
										 np.linalg.norm(vecs, axis=1))
		else:
			return np.dot(vecs, vec1) / (np.linalg.norm(vec1) * \
										 np.linalg.norm(vecs))


	#--------------------------------------------------------------------------
	# Separa POS dos tags em d2v_index
	#--------------------------------------------------------------------------
	def get_doctags_pos_por_tags_hdf5(self, df_tags):
		# df_tags: qq DF q tenha tags como indice

		if type(df_tags) == list:
			df_tags = pd.DataFrame(index=df_tags)

		# So preserva registros cujos tags estejam no d2v_model
		df_tags = df_tags[df_tags.index.isin(self.d2v_index.index)]

		# Adiciona posicao dos DOCVECS ao DF com metadados
		df_tags = df_tags.join(self.d2v_index, rsuffix='_INDEX')

		# Ordena df_dados segundo ordem crescente da posicao dos DOCVECS
		df_tags.sort_values('pos', inplace=True)

		# Separa posicoes dos tags de df_dados
		pos_vecs = df_tags['pos'].to_list()

		return pos_vecs, df_tags


	#--------------------------------------------------------------------------
	# NOVO - usa HDF5
	# Obtem posicoes dos tags_tipo
	# Retorna DF com tag(index) e pos(pos)
	# tipo_tag = sigla da entidade (areaava, areagrd, areabas, areasub,
	#             areaesp, tese, ppg, docente, discente)
	# outro_tag = uma instancia especifica de tag (ex.: codigo de area)
	#--------------------------------------------------------------------------
	def get_doctags_pos_por_tipo_hdf5(self, tipo_tag=None):
		# filtra ocorrencias de tags do tipo_tag2
		if  tipo_tag == 'areaava' or tipo_tag == 'area':
			tags_i = self.d2v_index.index.str.startswith('areaava-')
		elif tipo_tag == 'areagrd':
			tags_i = self.d2v_index.index.str.startswith('areagrd-')
		elif tipo_tag == 'areabas':
			tags_i = self.d2v_index.index.str.startswith('areabas-')
		elif tipo_tag == 'areasub':
			tags_i = self.d2v_index.index.str.startswith('areasub-')
		elif tipo_tag == 'areaesp':
			tags_i = self.d2v_index.index.str.startswith('areaesp-')
		elif tipo_tag == 'discente':
			tags_i = self.d2v_index.index.str.startswith('iddi-')
		elif tipo_tag == 'docente':
			tags_i = self.d2v_index.index.str.startswith('iddo-')
		elif tipo_tag == 'artigo' or tipo_tag == 'eid':
			tags_i = self.d2v_index.index.str.startswith('2-s2.0-')
		elif tipo_tag == 'topico':
			tags_i = self.d2v_index.index.str.startswith('t.')
		elif tipo_tag == 'ppg':
			tags_i = self.d2v_index.index.str.find('p',start=11)>-1
		elif tipo_tag == 'tese':
			tags_i = self.d2v_index.index.str.startswith('ida-')
		else:
			return pd.DataFrame()
		# retorna DF somente com tags (index) e pos dos docvecs do tipo_tag
		return self.d2v_index.loc[tags_i]
