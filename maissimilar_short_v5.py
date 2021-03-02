'''
Criado em 1/3/2021

Derivado de maissimilar_v5

Modificações:
	Não utiliza modelo NLP
	Utiliza vetores armazenados em HDF5

Por Alexandre Uchoa
'''
import re
import h5py
import numpy as np
import pandas as pd

from time import time

from pathlib import Path
relative_path = Path(__file__).parent

pasta			= {}
pasta['main'] 	= relative_path.as_posix() + '/'
pasta['modelos']= relative_path.as_posix() + '/modelos/'
pasta['estudos']= relative_path.as_posix() + '/estudos/'

pasta_main		= pasta['main']
pasta_modelos 	= pasta['modelos']
pasta_estudos 	= pasta['estudos']

#------------------------------------------------------------------------------
# Determina a similaridade entre tags de entidades
# a partir de representações de distribuição geradas preiviamente
#------------------------------------------------------------------------------
class MaisSimilar_Short(object):

	def __init__(self, modelo_fn=None):
		self.pasta_modelos 	= pasta_modelos
		self.pasta_estudos 	= pasta_estudos
		self.tipo_producao 	= 'teses'
		self.usa_hdf5 		= True
		self.load_modelo_hdf5(modelo_fn=modelo_fn)


	def load_modelo_hdf5(self, modelo_fn=None):
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

		# Forma nomes de arquivos
		self.docvecs_dv_fn 	= modelo_fn + ".docvecs.vectors_docs.hdf5"
		index_dv_fn 		= modelo_fn + ".docvecs.index_docs.npy"

		# Cria DF com indices dos docvecs e tags
		self.d2v_index = pd.DataFrame(np.load(pasta_modelos + index_dv_fn),\
									  columns=['tags'])
		self.d2v_index.set_index('tags',inplace=True,drop=True)
		self.d2v_index = self.d2v_index.assign(pos=range(len(self.d2v_index)))

		# Abre HDF5 com docvecs
		h5py_f1 = h5py.File(pasta_modelos + self.docvecs_dv_fn, 'r')
		self.d2v_docvecs = h5py_f1['docvecs']



	def _tags_cosine_v2(self, tag1, df_tags2, primeiros=20, sim_minima=.0):
		# Separa os docvecs correspondentes
		if tag1 not in df_tags2.index:
			df_tags2 = df_tags2.append(self.d2v_index.loc[tag1], \
									   ignore_index=False)
			df_tags2.sort_values('pos', inplace=True)

		np_so_docvecs = self.d2v_docvecs[df_tags2['pos'].to_list()]

		vec_tag1 = np_so_docvecs[df_tags2.index == tag1]
		vec_tag2 = np_so_docvecs

		cosine = self.cosine(vec_tag1[0], vec_tag2)

		# Aproveita TAGS (de areas) de vecs
		if primeiros == 0:
			primeiros = len(df_tags2)

		# Ordena decrescente e retorna somente os # primeiros casos
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


