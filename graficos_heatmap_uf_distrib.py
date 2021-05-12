import numpy as np
import pandas as pd

from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode, plot

from ipywidgets import Output, VBox, HBox, widgets, interactive
from IPython.display import display
init_notebook_mode(connected=True)

import qgrid

relative_path 		= Path(__file__).parent
pasta				= {}
pasta['estudos'] 	= relative_path.as_posix() + '/estudos/'
pasta['kmeans'] 	= relative_path.as_posix() + '/kmeans/'
pasta_estudos = pasta['estudos']
pasta_kmeans  = pasta['kmeans']

################################################################################

def main():
	_ = HeatmapUF()
	return



class HeatmapUF(object):

	def __init__(self, tipo_producao='teses', item_cluster='ppg' ):

		self.tipo_producao 	= tipo_producao
		self.item_cluster 	= item_cluster
		#self.leTopicos		= LDA.LeTopicos(tipo_producao,'area',42)
		self.load_simmatrix()
		#self.load_modelo_d2v()


	#  Carrega arquivos SIMMATRIX entre PPGs
	def load_simmatrix(self):
		# Le matriz de SIM, com triangulo superior zerado
		self.df_simmatrix = pd.DataFrame.from_records(np.load(pasta['kmeans']+\
							'SIMMATRIX-PPGS-PPGS_None_None.npy',\
							 allow_pickle=True))

		fname = "SIMILARIDADE_DE_PPGS_None_None_COM_PPG_%s_2D.csv" % \
				self.tipo_producao
		df_dados_ppgs = pd.read_csv(pasta_estudos + fname, sep=';', \
									encoding='utf-8')
		self.df_dados_ppgs = df_dados_ppgs
		
		areas = df_dados_ppgs.drop_duplicates('cd_area_avaliacao')\
							 .sort_values('nm_area_avaliacao')
		areas_menu = areas[['nm_area_avaliacao','cd_area_avaliacao']].values
		self.areas_menu = [(a.title(), b) for a, b in areas_menu]

		df_dados_ppgs.drop_duplicates(subset=['_id'], inplace=True)
		df_dados_ppgs.set_index('_id', inplace=True)
		df_dados_ppgs.index.name='cd_programa_ies'

		self.ufs_all = sorted(list(set(df_dados_ppgs['sg_uf_programa'].to_list())))
		self.ufs_ALL = [u.upper() for u in self.ufs_all]

		# Acrescenta dados dos PPGs ao indice da SIMMATRIX
		simmatrix_idx 	= np.load(pasta['kmeans']+\
								 'SIMMATRIX-PPGS-PPGS_None_None-INDEX.npy',\
								  allow_pickle=True)
			
		df_sim_idx = pd.DataFrame(index=simmatrix_idx)
		self.df_sim_idx = df_sim_idx.join(df_dados_ppgs[['sg_uf_programa',\
														 'cd_area_avaliacao']])
		self.df_sim_idx.index.name='cd_programa_ies'
		self.df_sim_idx.reset_index(drop=False, inplace=True)


	#-------------------------------------------------------------------------
	# Gera visualizacao 2D ou 3D artigos por clusters anuais usando
	# coordenadas geradas com MDS a partir de vetores DOCVECS
	#-------------------------------------------------------------------------
	def uf_uf_din(self):
		
		df_mesma_area, df_outra_area, dtick_mesma, dtick_outra = \
		self.heatmap_data(area_select=45, similaridade=0.7)
		
		trace1 = go.Heatmap(z=df_outra_area,
							x=self.ufs_ALL,
							y=self.ufs_ALL,
							coloraxis = "coloraxis",
							name='Outras Áreas',
							xgap=3,
							ygap=3)

		trace2 = go.Heatmap(z=df_mesma_area,
							x=self.ufs_ALL,
							y=self.ufs_ALL,
							coloraxis = "coloraxis2",
							name='Mesma Área',
							xgap=8,
							ygap=8)
		
		self.fig = go.FigureWidget(data=[trace1, trace2])
		
		self.fig.add_shape(type="line",
							x0=0, y0=0, x1=27, y1=27,
							line=dict(
									color="MediumBlue",
									width=4,
									dash="dot",
									)
							)

		self.fig.update_layout(
					title={'text':'Casos de PPGs similares em UFs e Áreas iguais e diferentes',
							'y':0.9,
							'x':0.5,
							'xanchor': 'center',
							'yanchor': 'top'},
					xaxis_title="UF dos PPGs similares",
					yaxis_title="UF dos PPGs de referência",
					height=600,
					boxmode='overlay',
					showlegend=True,
					coloraxis2=dict(colorbar=dict(title={'text':"casos somente com PPGs da área selecionada",
														 'font':{'size':12},
														 'side':'right'},
												  thickness=10, 
												  ypad=20,
												  tick0=0,
												  dtick=dtick_mesma,
												  xpad=80
												  ),
									colorscale=[[0,'white'],[1.,'red']],
									),
					coloraxis =dict(colorbar=dict(title={'text':"casos envolvendo PPGs de outras áreas",
														 'font':{'size':12},
														 'side':'right'},
												  thickness=10,
												  tick0=0,
												  ypad=20,
												  dtick=dtick_outra,
												  ),
									colorscale=[[0,'white'],[1.,'green']]
									),
					)

		out = Output()
		out.capture(clear_output=True)
		
		# Updates the image options based on directory value
		def update_sim_range(*args):
			#wsim.max 	= self.max_slider
			wsim.value 	= self.max_slider - 0.1

		def update_heatmaps(Area, Similaridade):

			new_mesma_area, new_outra_area, dtick_mesma, dtick_outra = \
			self.heatmap_data(area_select=Area, similaridade=Similaridade)

			self.fig.data[0].z = new_outra_area
			self.fig.data[1].z = new_mesma_area

			self.fig.update_layout(coloraxis2={'colorbar':{'dtick':dtick_mesma}},
								   coloraxis ={'colorbar':{'dtick':dtick_outra}})
			
			# Se nao houver casos, tudo branco
			if new_mesma_area.max().max() == 0:
				self.fig.update_layout(coloraxis2={'colorscale':[[0,'white'],[1.,'white']]})
			else:
				self.fig.update_layout(coloraxis2={'colorscale':[[0,'white'],[1.,'red']]})

			if new_outra_area.max().max() == 0:
				self.fig.update_layout(coloraxis ={'colorscale':[[0,'white'],[1.,'white']]})
			else:
				self.fig.update_layout(coloraxis ={'colorscale':[[0,'white'],[1.,'green']]})

			self.max_slider = max(self.max_sim_mesma, self.max_sim_outra)
			#print(self.max_sim_mesma, self.max_sim_outra, self.max_slider)

			wsim.max 	= self.max_slider
			#wsim.value 	= self.max_slider - 0.1

			return self.fig

		# Create widgets
		style = {'description_width': 'initial'}
		self.max_slider = max(self.max_sim_mesma, self.max_sim_outra)
		
		# Define widgets
		warea 	= widgets.Dropdown(options=self.areas_menu,\
									value = 45,\
									layout=widgets.Layout(width='40%'), \
									description='Área', \
									style=style)
		
		wsim 	= widgets.FloatSlider(min=0.3, max=self.max_slider, step=0.02,\
									 value=self.max_slider - 0.1, \
									 layout=widgets.Layout(width='40%'), \
									 description='Similaridade mínima', \
									 style=style)

		warea.observe(update_sim_range, 'value')

		widget_dropdowns  = interactive(update_heatmaps, \
										Area=warea,\
										Similaridade=wsim)
		widget_dropdowns

		return VBox([HBox(widget_dropdowns.children), \
					 self.fig, \
					 out])


	#==========================================================================
	# Gera visualizacao Heatmap de areas e clusters
	#==========================================================================
	def heatmap_data(self, area_select=None, similaridade=.7):
		#zeros_ufs = np.zeros((len(self.ufs_all), len(self.ufs_all)), dtype='int')
		
		heat_data_mesma_area = pd.DataFrame(np.zeros((len(self.ufs_all), \
													  len(self.ufs_all)), dtype='int'), \
											columns=self.ufs_all, \
											index=self.ufs_all)
		heat_data_outra_area = pd.DataFrame(np.zeros((len(self.ufs_all), \
													  len(self.ufs_all)), dtype='int'), \
											columns=self.ufs_all, \
											index=self.ufs_all)

		# Copia SIMMATRIX
		df_sim_idx 		= self.df_sim_idx.copy()
		df_simmatrix 	= self.df_simmatrix.copy()
		
		df_simmatrix += df_simmatrix.T
		
		# Zera todas as areas diferentes da sele
		df_sim_idx.loc[df_sim_idx['cd_area_avaliacao'] != area_select, \
								  'cd_area_avaliacao'] = 0
			
		# Adiciona multiindex com area e UF nas linhas e colunas
		m_index = pd.MultiIndex.from_frame(\
						df_sim_idx[['cd_area_avaliacao','sg_uf_programa',\
									'cd_programa_ies']])
		df_simmatrix.index 		= m_index
		df_simmatrix.columns 	= m_index
		
		# Ordena SIMMATRIX descendente por area (0 fica ao final)
		df_simmatrix.sort_index(axis=0, ascending=False, inplace=True)
		df_simmatrix.sort_index(axis=1, ascending=False, inplace=True)
		
		self.df_sim = df_simmatrix
		
		# Para mesma área
		# Usa triangulo inferior de SIMMATRIX para PPGs da mesma area 
		np_dfs = np.tril(df_simmatrix)
		dfs = pd.DataFrame(np_dfs, columns=df_simmatrix.columns, index=df_simmatrix.index)

		self.max_sim_mesma = dfs.loc[area_select][area_select].max().max()

		dfs = dfs[dfs >= similaridade].loc[area_select][area_select]
		df_m = dfs.groupby(axis=0,level=0).count()
		#df_m = df_m[df_m > 0].groupby(axis=1,level=0).count()
		df_m = df_m.groupby(axis=1,level=0).sum()

		heat_data_mesma_area.at[df_m.index, df_m.columns] = df_m.values
		
		# Para outras áreas
		# Usa triangulo superior de SIMMATRIX para outras areas (=0)
		np_dfs = np.triu(df_simmatrix)
		dfs = pd.DataFrame(np_dfs, columns=df_simmatrix.columns, index=df_simmatrix.index)

		self.max_sim_outra = dfs.loc[area_select][0].max().max()

		dfs = dfs[dfs >= similaridade].loc[area_select][0]
		df_o = dfs.groupby(axis=0,level=0).count()
		#df_o = df_o[df_o > 0].groupby(axis=1,level=0).count()
		df_o = df_o.groupby(axis=1,level=0).sum()

		heat_data_outra_area.at[df_o.index, df_o.columns] = df_o.values
		
		max_mesma = heat_data_mesma_area.max().max()
		if  max_mesma > 1:
			dtick_mesma = 1 + max_mesma // 6
		else:
			dtick_mesma = 1
			
		max_outra = heat_data_outra_area.max().max()
		if  max_outra > 1:
			dtick_outra = 1 + max_outra // 6
		else:
			dtick_outra = 1
			
		self.max_slider = max(self.max_sim_mesma, self.max_sim_outra)

		return heat_data_mesma_area, heat_data_outra_area, dtick_mesma, dtick_outra


	def tabela_ppgs(self):
		df = self.df_dados_ppgs[['nm_area_avaliacao','sg_uf_programa','nm_programa_ies',\
								'nm_area_avaliacao_SIM','sg_entidade_ensino','pubs',\
								'SIMILARIDADE','sg_uf_programa_SIM','nm_programa_ies_SIM',\
								'sg_entidade_ensino_SIM','pubs_SIM']]\
								.fillna('').sort_values(by=['nm_area_avaliacao',\
								'sg_uf_programa','nm_programa_ies',\
								'nm_area_avaliacao_SIM','sg_entidade_ensino','pubs',\
								'SIMILARIDADE'])
							
		qw = qgrid.show_grid(df, show_toolbar=True,\
								 grid_options={\
										'forceFitColumns': False, \
										'editable': False, \
										'fullWidthRows': True
										}, \
								 column_options={\
										'enableTextSelectionOnCells':True,\
										'enableColumnReorder':True \
										}, \
								column_definitions={'nm_area_avaliacao':{'width':150}
										}
								)
		return display(qw)


#=========================================================================
# Formata nomes de arquivos compostos pelos parametros de geracao
#=========================================================================
def _set_fname(params):
	if type(params) is not list:
		params = [params]
	params_new = []
	for p in params:
		if type(p) is list:
			params_new.append('-'.join([str(e) for e in p]))
		else:
			params_new.append(p)
	nome = '%s_' * len(params_new)
	nome = nome[:-1]
	return nome % tuple(params_new)


