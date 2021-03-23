# -*- coding: utf-8 -*-
"""
Modified on 1 Mar 2021

@author: Alexandre P. Uchoa (alex.uchoa@gmail.com)
"""
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
from pathlib import Path
import qgrid

from ipywidgets import Output, VBox, HBox, widgets, interactive
from IPython.display import display

import networkx as nx
import matplotlib.pyplot as plt

import maissimilar_short_v5 as MS


relative_path 		= Path(__file__).parent
pasta				= {}
pasta['estudos'] 	= relative_path.as_posix() + '/estudos/'
pasta_estudos = pasta['estudos']

class Graficos(object):

	def __init__(self, tipo_producao='teses', modelo_fn=None, \
					   versao=0, item_cluster='tese'):

		self.tipo_producao 	= tipo_producao
		self.item_cluster 	= item_cluster
		self.dimensoes 		= 2

		if modelo_fn is None:
			self.modelo_fn = '0_d2v-teses-topicos_model'
		else:
			self.modelo_fn = modelo_fn

		# Le dados de cada clusters e numero de artigos
		if versao > 0:
			self.versao_str = 'versao_'+str(versao)
		else:
			self.versao_str=''

		self.ms = MS.MaisSimilar_Short(modelo_fn=self.modelo_fn)

		# Le arquivo previamente preparado com metadados de PPGs
		fname='METADADOS-PPG_None_None_teses.csv'
		self.df_ppgs = pd.read_csv(pasta_estudos + fname, sep=';', encoding='utf-8')
		self.df_ppgs.set_index('_id', drop=False, inplace=True)
		self.df_ppgs = self.df_ppgs.fillna('')

		self.df_ppgs = self.df_ppgs[self.df_ppgs.index\
									.isin(self.ms.d2v_index.index.tolist())]

		# Le arquivo previamente preparado com Similaridades
		# Somente para extrair min e max
		fname = 'SIMILARIDADE_DE_PPGS_None_None_COM_PPG_teses_2D.csv'
		df = pd.read_csv(pasta_estudos + fname, sep=';', encoding='utf-8')
		sims_all = df['SIMILARIDADE'].fillna(.0)
		self.sims_all = sorted(sims_all[sims_all > 0].tolist())
		del df

		# Cores usadas
		color_list  = ['green','magenta','orange','blue','gold','violet',\
					   'olive','red','lightgrey','fuchsia','peru', 'dimgrey',\
					   'plum','cadetblue','salmon','lime','deeppink',\
					   'lightblue','yellow','khaki', 'forestgreen', \
					   'pink', 'turquoise', 'blueviolet', 'burlywood',\
					   'chartreuse', 'thistle', 'coral', 'cornflowerblue']
		#color_list += color_list
		self.color_list  = np.array(2 * color_list)

		# Cor da borda dos nos segundo area
		all_ufs 	= self.df_ppgs['sg_uf_programa'].drop_duplicates()\
													.sort_values()\
													.to_list()
		s_cor_borda = pd.Series(self.color_list[:len(all_ufs)], \
								index=all_ufs, name='cor_borda')
		self.df_ppgs = self.df_ppgs.join(s_cor_borda, on='sg_uf_programa',\
										 how='left')
		# Cor do miolo dos nos segundo UF
		all_areas 	= self.df_ppgs['cd_area_avaliacao'].drop_duplicates()\
													   .sort_values()\
													   .to_list()
		s_cor_miolo = pd.Series(self.color_list[:len(all_areas)], \
								index=all_areas, name='cor_miolo')
		self.df_ppgs = self.df_ppgs.join(s_cor_miolo, on='cd_area_avaliacao',\
										 how='left')


	def lista_ppgs(self):
		df = self.df_ppgs[['nm_programa_ies', 'sg_entidade_ensino', 'nm_area_avaliacao','sg_uf_programa','bow','tfidf']]\
                            .sort_values(by=['sg_uf_programa','nm_area_avaliacao'])
		df.columns = ['Nome','IES','Área','UF','Conceitos mais frequentes','Conceitos particulares ao PPG']
		qg = qgrid.show_grid(df,\
                show_toolbar=True,\
				grid_options={\
							'forceFitColumns': False, \
							'editable': False, \
							'fullWidthRows': True \
							}, \
				column_options={\
							'enableTextSelectionOnCells':True,\
							'enableColumnReorder':True \
							},\
				column_definitions={'Conceitos mais frequentes':{'width':500}, \
                                    'Conceitos particulares ao PPG':{'width':500}, \
                                    'Nome': {'width':240}, \
                                    '_id': {'width':120}, \
                                    'IES': {'width':40}, \
                                    'UF':{'width':40}})
		display(qg)


	def grafo_ppgs(self):
		#--------------------------------------------------------------------------
		# Colhe variedade de UFs, Areas etc.
		all_ufs 	= self.df_ppgs['sg_uf_programa'].drop_duplicates()\
													.sort_values()\
													.to_list()
		all_ufs 	= ['Todas'] + all_ufs
		all_areas 	= [(nm, cd) for nm, cd \
								in self.df_ppgs[['nm_area_avaliacao',\
												 'cd_area_avaliacao']]\
												 .drop_duplicates('cd_area_avaliacao')\
												 .sort_values(by='nm_area_avaliacao')\
												 .values]
		all_areas = [('Todas', 0)] + all_areas

		out = Output()
		@out.capture(clear_output=True)

		#----------------------------------------------------------------------
		# Atualiza plot e tabela lateral segundo escolhas nos dropboxes
		def update_droplists(Area_Aval_Pri, UF, Percentil, \
							 Mesma_UF, Mesma_IES, Area_Aval_Sec):

			msg_area = 'sejam de qualquer área'
			if Area_Aval_Pri == 0:
				area_pri = [a for n,a in all_areas]
			else:
				area_pri = [Area_Aval_Pri]
				if len(Area_Aval_Sec) == 1 and Area_Aval_Sec[0] == Area_Aval_Pri:
					msg_area = 'sejam também da área ' + str(Area_Aval_Pri)
				elif Area_Aval_Sec[0] == 0:
					msg_area = 'sejam de qualquer área'
				else:

					txt = ', '.join([a[0] for a in np.array(all_areas)[list(Area_Aval_Sec)]])
					msg_area = 'sejam da(s) área(s) ' + txt

			msg_uf = 'estejam em qualquer UF'
			if UF == 'Todas':
				uf_select = all_ufs
			else:
				uf_select = [UF]
				if Mesma_UF:
					msg_uf = 'estejam no mesmo estado ' + UF.upper()
			msg = ' e '.join([msg_uf, msg_area])

			# Inicializa grafo
			G = nx.Graph()
			plt.figure(figsize=(16, 8))

			# Similaridade minima definida pelo usuario
			sim_min = np.percentile(self.sims_all, int(Percentil))

			filtro_select = (self.df_ppgs['cd_area_avaliacao'].isin(area_pri)) & \
							(self.df_ppgs['sg_uf_programa'].isin(uf_select))

			# Se area de partida for todas, força area secundaria todas
			if Area_Aval_Pri == 0:
				Area_Aval_Sec = (0,)

			# Se os filtros ecolhidos forem muito abrangentes
			cancelada = False
			if Area_Aval_Sec[0] == 0:
				if UF == 'Todas':
					cancelada = True

			if cancelada:
				print('\n**Consulta cancelada. Recomenda-se refinar a seleção, pois essa busca pode levar muito tempo.')
				return

			if Area_Aval_Pri == 0 and UF != 'Todas' and not Mesma_UF:
				Mesma_UF= True
				print('** AVISO: Assumido que se buscam PPGs similares na mesma UF. '+\
					  'Caso contrario, a quantidade de possibilidades seria muito grande.')

			print('%s PPGs selecionados como referência de partida.' % \
				  len(self.df_ppgs[filtro_select]))
			print('Buscando casos de PPGs que %s e tenham similaridade mútua superior a %s.' % \
				 (msg, sim_min))

			# Coleta PPGs similares aos PPGs selecionados segundo filtros
			edges, nodes = self.get_ppgs_similares( \
										df_ppgs=self.df_ppgs[filtro_select], \
										mesmaies=Mesma_IES, \
										mesmauf=Mesma_UF, \
										areasec=Area_Aval_Sec, \
										sim_minima=sim_min)

			# Converte arestas e nos em DF q pode ser usado para montar Scatter
			if len(edges) == 0:
				print('\n** Nenhum PPG satisfaz o critério de filtragem usado **')
				return

			# Cria DF para apresentação
			print('Encontrados %s casos de similaridade entre %s PPGs.' % \
				 (len(edges), len(nodes)))
			new_data = pd.DataFrame.from_records(list(edges))
			new_data.columns = ['_id', '_id_SIM', 'SIMILARIDADE']
			new_data.set_index('_id', inplace=True, drop=True)

			new_data = new_data.join(self.df_ppgs,\
									 how='left')
			new_data = new_data.join(self.df_ppgs[['nm_programa_ies', \
												   'sg_entidade_ensino', \
												   'nm_area_avaliacao',\
												   'nm_municipio_programa_ies', \
												   'sg_uf_programa']] , \
									 on='_id_SIM', how='left', rsuffix='_SIM')

			# Garante q PPGs recuperados sao da mesma IES, se assim pedido
			if Mesma_IES:
				new_data = new_data.assign(mesmaies= lambda x:
										   x['sg_entidade_ensino']==\
										   x['sg_entidade_ensino_SIM'])
				new_data = new_data[new_data.mesmaies]

				# Garante q PPGs recuperados sao da mesma UF, se assim pedido
			if Mesma_UF and not new_data.empty:
				new_data = new_data.assign(mesmauf= lambda x:
										   x['sg_uf_programa']==\
										   x['sg_uf_programa_SIM'])
				new_data = new_data[new_data.mesmauf]

			if new_data.empty:
				print('\n** Nenhum PPG satisfaz o critério de filtragem usado **')
				return

			# Refas lista de nos e arestas somente com os que sobraram da
			# filtragem acima
			nodes = list(set(new_data['_id'].to_list() + \
							 new_data['_id_SIM'].to_list()))
			G.add_nodes_from(nodes)
			edges = [(n1, n2, w) for n1, n2, w in edges \
								 if n1 in nodes and n2 in nodes]
			G.add_weighted_edges_from(edges)
			nodes

			# Define padrao de posicionamento dos nos
			pos = nx.spring_layout(G, k=1.5)

			nx.draw_networkx_edges(G, pos, alpha=0.4)

			sizes 		= self.df_ppgs.loc[list(nodes)]['pubs'].to_numpy() * 10
			cor_miolo 	= self.df_ppgs.loc[list(nodes)]['cor_miolo'].to_list()
			cor_borda 	= self.df_ppgs.loc[list(nodes)]['cor_borda'].to_list()

			nx.draw_networkx_nodes(G, pos, \
								   node_color=cor_miolo, \
								   edgecolors=cor_borda, \
								   linewidths=10, \
								   node_size=sizes)

			nx.draw_networkx_labels(G, pos, font_size=11, \
									verticalalignment='top')
			plt.show()

			display_data = new_data[['nm_programa_ies', \
									 'sg_entidade_ensino', \
									 'nm_area_avaliacao',\
									 'nm_municipio_programa_ies', \
									 'sg_uf_programa', \
									 'SIMILARIDADE','_id_SIM',\
									 'nm_programa_ies_SIM', \
									 'sg_entidade_ensino_SIM', \
									 'nm_area_avaliacao_SIM',\
									 'nm_municipio_programa_ies_SIM', \
									 'sg_uf_programa_SIM']]\
									.copy()

			display_data.index.name='PPG Partida'

			display_data.columns=['Nome','IES','Área','Municipio','UF',\
								  'Similaridade','PPG Similar','Nome (Sim)',\
								  'IES (Sim)','Área (Sim)','Municipio (Sim)',\
								  'UF (Sim)']

			qw = qgrid.show_grid(display_data,\
								 show_toolbar=True,\
								 grid_options={\
										'forceFitColumns': False, \
										'editable': False, \
										'fullWidthRows': True \
										}, \
								 column_options={\
										'enableTextSelectionOnCells':True,\
										'enableColumnReorder':True \
										},
								 column_definitions={'Nome': {'width':240}, \
													 'PPG Partida': {'width':120}, \
													 'IES': {'width':70}, \
													 'Municipio': {'width':70}, \
													 'UF': {'width':40}, \
													 'Similaridade': {'width':100}, \
													 'Nome (Sim)': {'width':240}, \
													 'PPG Similar': {'width':120}, \
													 'IES (Sim)': {'width':70}, \
													 'Municipio (Sim)': {'width':70}, \
													 'UF (Sim)': {'width':40}})
			display(qw)

		# Define widgets dos dropboxes
		style = {'description_width': 'initial'}

		wareapri = widgets.Dropdown(value=0,
									options=all_areas, \
									description='Area de Partida',\
									#layout=widgets.Layout(width='30%'), \
									style=style)

		wareasec = widgets.SelectMultiple(options=all_areas,\
										  value=[0], \
										  #layout=widgets.Layout(width='70%'), \
										  rows=10, \
										  description='Areas dos Similares', \
										  style=style)

		wufs = widgets.Dropdown(value='Todas',
								options=all_ufs, \
								description='UF',\
								#layout=widgets.Layout(width='20%'), \
								style=style)

		wpercentil = widgets.IntSlider(value=85, min=0, max=100, \
									  continuous_update=False, \
									  description='Similaridade (em %)', \
									  style=style)
									  #layout=widgets.Layout(width='50%'))

		wmesmaies = widgets.Checkbox(value=False, \
									 description='Somente PPGs de mesma IES', \
									 disabled=False)

		wmesmauf = widgets.Checkbox(value=False, \
									description='Somente PPGs de mesma UF', \
									disabled=False)

		wi  = interactive(update_droplists, {'manual': True},\
							Area_Aval_Pri=wareapri, \
							Area_Aval_Sec=wareasec, \
							UF=wufs, \
							Percentil=wpercentil, \
							Mesma_UF=wmesmauf, \
							Mesma_IES=wmesmaies)
		return VBox([ \
					 HBox([VBox(wi.children[:5]),\
						   VBox(wi.children[5:7]) \
						   ]), out])

	#--------------------------------------------------------------------------
	# Encontra PPGs similares aos dados em df_ppgs e calcula suas similaridades
	#--------------------------------------------------------------------------
	def get_ppgs_similares(self, df_ppgs=pd.DataFrame(), \
								 mesmaies=False, \
								 mesmauf=True, \
								 areasec=(0,), \
								 sim_minima=.0):

		# Separa PPGs que atendem a criterio baseado em ppg fornecido
		df_ppgs_criterio = self.get_ppgs_segundo_criterio(df_ppgs=df_ppgs, \
														  areasec=areasec, \
														  mesmaies=mesmaies, \
														  mesmauf=mesmauf)
		df_ppgs_criterio.drop_duplicates(inplace=True)

		# Det. as posicoes de seus docvecs
		pos2, df_pos2 = self.ms.get_doctags_pos_por_tags_hdf5(df_ppgs_criterio\
															. drop_duplicates())
		nodes_sim = set([])
		edges_sim = set([])
		edges_ao_contrario = set([])

		for ppg in df_ppgs.index.to_list():
			# Calcula a similaridade e filtra os que estao acima do min
			df_tags2  = df_pos2[['pos']].copy()
			df_sims   = self.ms._tags_cosine_v2(tag1=ppg, \
												df_tags2=df_tags2, \
												sim_minima=sim_minima, \
												primeiros=0)
			# Se houver ppg similar dentro do criterio (alem do proprio)...
			if len(df_sims) > 1:
				# Adiciona uma aresta a cada dupla ppg e ppg_sim
				# Ignora primeiro da lista pq eh o proprio ppg
				# Assume similaridade x 10 como o peso da aresta
				df_sims = df_sims.join(df_pos2, how='left', rsuffix='_POS')

				edges_ao_contrario.update([(ppg_sim, ppg) \
								   for ppg_sim in df_sims[1:].index.to_list()])

				edges_sim.update([(ppg, ppg_sim, '%.3f' % sim) \
								  for ppg_sim, sim \
								  in df_sims[1:][['_id','SIMILARIDADE']]\
												.values
								  if (ppg, ppg_sim) not in edges_ao_contrario])
				# Inclui todos os indices do DF (ppgs, incluindo o prprio)
				# como nos
				nodes_sim.update(df_sims.index.tolist())

		return edges_sim, nodes_sim


	#--------------------------------------------------------------------------
	# Obtem lista de ppgs que atendem criterio de filtragem fornecido
	#--------------------------------------------------------------------------
	def get_ppgs_segundo_criterio(self, df_ppgs=pd.DataFrame(), \
										areasec=(0,), \
										mesmaies=False, \
										mesmauf=True):
		# Separa dados do ppg indicado
		ies = df_ppgs['sg_entidade_ensino'].drop_duplicates().tolist()
		ufs = df_ppgs['sg_uf_programa'].drop_duplicates().tolist()
		#areas = df_ppgs['cd_area_avaliacao'].drop_duplicates().tolist()

		if areasec[0] == 0:
			areasec = self.df_ppgs['cd_area_avaliacao'].drop_duplicates().tolist()

		# Det qual sao as colunas do DF de similaridades que devem ser usadas
		filtro = np.array([True] * len(self.df_ppgs))

		if mesmaies:
			filtro = filtro & (self.df_ppgs['sg_entidade_ensino'].isin(ies))
		if mesmauf:
			filtro = filtro & (self.df_ppgs['sg_uf_programa'].isin(ufs))
		if areasec != [0]:
			filtro = filtro & (self.df_ppgs['cd_area_avaliacao'].isin(areasec))

		return self.df_ppgs[filtro].copy()
