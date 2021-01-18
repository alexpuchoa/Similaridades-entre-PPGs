# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:10:09 2020

@author: Alexandre P. Uchoa (alex.uchoa@gmail.com)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ipywidgets import Output, VBox, HBox, widgets, interactive

from pathlib import Path

'''
relative_path 		= Path(__file__).parent
pasta				= {}
#pasta['main'] 		= relative_path.as_posix() + '/'
pasta['estudos'] 	= relative_path.as_posix() + '../estudos/'
'''
pasta_estudos = '../estudos/'

def grafo_mds_ppgs(df=None, dimensoes=3):

	# Le arquivo previamente preparado
	if df is None:
		fname = 'SIMILARIDADE_DE_PPGS_None_None_COM_PPG_teses_%sD.csv' % dimensoes
		df=pd.read_csv(pasta_estudos + fname, sep=';', encoding='utf-8')

	df.set_index('_id', drop=False, inplace=True)

	#--------------------------------------------------------------------------
	# Colhe variedade de UFs, Regioes, Areas etc.
	all_ufs 	= df['sg_uf_programa'].drop_duplicates().to_list()
	all_regioes = df['nm_regiao'].drop_duplicates().to_list()
	all_areas 	= df['nm_area_avaliacao'].drop_duplicates().to_list()
	#conceitos	= df['cd_conceito_programa'].drop_duplicates().to_list()
	all_decis 	= df['DECIL_MAISSIM'].drop_duplicates().to_list()
	all_decis.sort(reverse=False)
	all_ufs.sort()
	all_regioes.sort()
	all_areas.sort()

	# Prepara escalas de cores
	color_list  = 2 * ['green','magenta','orange','blue','gold','violet',\
					   'olive','red','darkgrey','fuchsia','brown', \
					   'indigo','darksalmon','darkolivegreen','deeppink',\
					   'darkblue','yellow','darkred',\
					   'black', 'pink', 'darkturquoise', \
			           'blueviolet', 'burlywood', 'cadetblue',\
			           'chartreuse', 'chocolate', 'coral', 'cornflowerblue']
	s_colors = pd.Series(color_list[:len(all_areas)], index=all_areas, name='cor')
	df = df.join(s_colors, on='nm_area_avaliacao')

	# Monta texto para hover
	def ajusta_hovertext(df):
		texto 	= df['_id'] + '<br>' + df['nm_programa_ies'].str.title() + '<br>' +\
			      'IES: '+df['sg_entidade_ensino'].str.upper() + ' - ' + \
				  'Area: '+df['nm_area_avaliacao'].str.title() + '<br>' +\
				  'UF: ' +df['sg_uf_programa'].str.upper() + ' - ' + \
				  'Conceito: '+df['cd_conceito_programa'].astype('str')
		return texto

	texto_n = ajusta_hovertext(df)

	def inclui_edges_3D_2(df):
		df2=df.join(df[['x','y','z']], on='_id_SIM', rsuffix='_SIM')
		df2.dropna(inplace=True)
		df2.drop(df2[df2['_id'].isin(df2['_id_SIM'])].index, inplace=True)
		edge_x = []
		edge_y = []
		edge_z = []
		edge_text = []
		A = []
		B = []
		for a, b, SIM, x0, y0, z0, x1, y1, z1 in df2[['_id', '_id_SIM',\
													  'SIMILARIDADE','x','y','z',\
													  'x_SIM', 'y_SIM', 'z_SIM']].values:
			A.append(a)
			B.append(b)
			edge_x.append(x0)
			edge_x.append(x1)
			edge_x.append(None)
			edge_y.append(y0)
			edge_y.append(y1)
			edge_y.append(None)
			edge_z.append(z0)
			edge_z.append(z1)
			edge_z.append(None)
			edge_text.append(str(SIM))
			edge_text.append(str(SIM))
			edge_text.append(str(SIM))
		return A, B, edge_x, edge_y, edge_z, edge_text

	def inclui_edges_2D_2(df):
		df2=df.join(df[['x','y']], on='_id_SIM', rsuffix='_SIM')
		df2.dropna(inplace=True)
		df2.drop(df2[df2['_id'].isin(df2['_id_SIM'])].index, inplace=True)
		edge_x = []
		edge_y = []
		edge_text = []
		A = []
		B = []
		for a, b, SIM, x0, y0, x1, y1 in df2[['_id', '_id_SIM', \
											  'SIMILARIDADE','x','y',\
											  'x_SIM', 'y_SIM']].values:
			A.append(a)
			B.append(b)
			edge_x.append(x0)
			edge_x.append(x1)
			edge_x.append(None)
			edge_y.append(y0)
			edge_y.append(y1)
			edge_y.append(None)
			edge_text.append(str(SIM))
			edge_text.append(str(SIM))
			edge_text.append(str(SIM))
		return A, B, edge_x, edge_y, edge_text

	df_ini = df[df['nm_regiao']=='norte']

	if dimensoes == 3:
		A, B, edge_x, edge_y, edge_z, edge_text = inclui_edges_3D_2(df_ini)

		edge_trace = go.Scatter3d(
			x=edge_x, y=edge_y, z=edge_z,
			name='Similaridade',
			line=dict(width=3,
					  color='rgb(180,180,180)'),
			hoverinfo='text',
			text=edge_text,
			textposition='top center',
			hovertemplate = "%{text}",
			mode='lines')

		node_trace = go.Scatter3d(
			x=df_ini['x'], y=df_ini['y'], z=df_ini['z'],
			mode='markers',
			name='PPG',
			hoverinfo='text',
			text=texto_n,
			hovertemplate = "%{text}<br>"+
							"Pubs: %{marker.size:,.0f}</br>",
			marker={'color': df_ini['cor'],
					#'colorscale': colorscale2,
					'sizemin': 3,
					'sizemode': 'area',
					'sizeref': .16,
					'size': df_ini['pubs']*5},
			line_width=1)
	else:
		A, B, edge_x, edge_y, edge_text = inclui_edges_2D_2(df_ini)

		edge_trace = go.Scatter(
			x=edge_x, y=edge_y,
			name='Similaridade',
			line=dict(width=3,
					  color='rgb(180,180,180)'),
			hoverinfo='text',
			text=edge_text,
			textposition='top center',
			hovertemplate = "%{text}",
			mode='lines+text')

		node_trace = go.Scatter(
			x=df_ini['x'], y=df_ini['y'],
			mode='markers',
			name='PPGs',
			hoverinfo='text',
			selected={'marker': {'color': 'black'}},
			unselected={'marker': {'opacity': 1}},
			text=texto_n,
			hovertemplate = "%{text}<br>"+
							"Pubs: %{marker.size:,.0f}</br>",
			marker={'color': df_ini['cor'],
					#'colorscale': colorscale2,
					'sizemin': 3,
					'sizemode': 'area',
					'sizeref': .16,
					'size': df_ini['pubs']},
			line_width=1)

	tab_trace = tabela_detalhe_ppg()

	if dimensoes == 3:
		fig = go.FigureWidget(data=[edge_trace, node_trace])
	else:
		fig = go.FigureWidget(data=[edge_trace, node_trace, tab_trace])

	fig.update_layout(
			        height=600,
					width=1000,
					showlegend=False,
					#template='plotly_white',
					plot_bgcolor='white',
			        dragmode='select',
					yaxis={'domain': [0.3, 1]},
					margin=dict(l=0, r=0, b=0, t=50),
					title='Similaridade entre PPGs de mesma Região, UF e/ou Área')

	out = Output()
	@out.capture(clear_output=True)

	# Atualiza plot e tabela lateral segundo escolhas nos dropboxes
	def update_droplists(Area_Aval, Regiao, UF, Decil):
		if Area_Aval == 'Todas':
			area_select = all_areas
		else:
			area_select = [Area_Aval]

		if Regiao == 'Todas':
			regiao_select = all_regioes
		else:
			regiao_select = [Regiao]

		if UF == 'Todas':
			uf_select = all_ufs
		else:
			uf_select = [UF]

		if Decil == 'Todos':
			decil_select = [i + 1 for i in range(10)]
		else:
			decil_select = [i for i in range(int(Decil), 11)]

		filtro_select = (df['nm_area_avaliacao'].isin( area_select )) & \
						(df['nm_regiao'].isin( regiao_select )) & \
						(df['sg_uf_programa'].isin( uf_select )) & \
						(df['DECIL_MAISSIM'].isin( decil_select ))

		new_data  = df[filtro_select]

		if dimensoes == 3:
			A, B, edge_x, edge_y, edge_z, edge_text 	= \
				inclui_edges_3D_2(new_data)
		else:
			A, B, edge_x, edge_y, edge_text 			= \
				inclui_edges_2D_2(new_data)

		fig.data[0].x 	= edge_x
		fig.data[0].y 	= edge_y
		fig.data[0].text = edge_text

		if dimensoes == 3:
			fig.data[0].z 	= edge_z

		fig.data[1].x = new_data['x']
		fig.data[1].y = new_data['y']
		if dimensoes == 3:
			fig.data[1].z = new_data['z']

		texto_n = ajusta_hovertext(new_data)

		fig.data[1].text = texto_n
		if dimensoes == 2:
			fig.data[1].marker.size  = new_data['pubs'].div(3)
		else:
			fig.data[1].marker.size  = new_data['pubs']
		fig.data[1].marker.color = new_data['cor']

	all_ufs 	= ['Todas'] + all_ufs
	all_regioes = ['Todas'] + all_regioes
	all_decis 	= ['Todos'] + all_decis
	all_areas 	= ['Todas'] + all_areas

	# Define widgets dos dropboxes
	widget_dropdowns  = interactive(update_droplists, \
									Regiao=all_regioes, \
									UF=all_ufs, \
									Decil=all_decis, \
									Area_Aval=all_areas)

	# Update tabela lateral ao scatter em funcao do click sobre
	def update_tabela(trace, points, state):
		fig.data[1].selectedpoints = points.point_inds

		ppg_select = [t.split('<br>')[0] for t in trace.text[points.point_inds]]

		df_ppg_select = df[df['_id'].isin(ppg_select)]\
							 [['_id', 'nm_programa_ies', 'sg_entidade_ensino',\
								'bow', 'tfidf']]
		fig.data[2].cells.values = [df_ppg_select[k].tolist() \
									for k in df_ppg_select.columns]

	if dimensoes == 2:
		fig.data[1].on_click(update_tabela)

	return VBox((HBox(widget_dropdowns.children), HBox([fig]), out))
	#return VBox((HBox(widget_dropdowns.children), HBox([fig])))


# Monta lista tabular de PPGs selecionados em plot Scatter
def tabela_detalhe_ppg(df_dados_ppg=None):

	if df_dados_ppg is None:
		dados = pd.DataFrame([5*['']], columns=['_id', 'nm_programa_ies', 'sg_entidade_ensino',\
							'bow', 'tfidf'])
	else:
		dados = df_dados_ppg[['_id', 'nm_programa_ies', 'sg_entidade_ensino',\
								'bow', 'tfidf']]
	colunas = ['CD','NM PPG','IES','Frequentes','Tipicas']
	# Constroi tabela
	figt = go.Table(domain={'y': [0, 0.3]},
					header={'values': colunas,
				            'font': {'size':12},
							'fill_color':'orange',
							'align':"left"
							},
					cells={'values':[dados[k].tolist() \
									 for k in dados.columns],
						   'font': {'size':11},
						   'fill_color':'ivory',
					       'align':"left"},
					columnwidth=[1.2,1.3,.5,3.5,3.5]
					)
	return figt

