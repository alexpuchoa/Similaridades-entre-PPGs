# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:10:09 2020

@author: auchoa
"""
import networkx as nx
import pandas as pd
#import matplotlib as plt
import numpy as np
import plotly.graph_objects as go

from ipywidgets import Output, VBox, HBox, widgets, interactive

from capes_nlp import pasta


def grafo_mds_ppgs(df=None, dimensoes=3):

	# Le arquivo previamente preparado
	if df is None:
		fname = 'SIMILARIDADE_DE_PPGS_None_None_COM_PPG_teses_%sD.csv' % dimensoes
		df=pd.read_csv(pasta['estudos']+fname, sep=';', encoding='utf-8')

	df.set_index('_id', drop=False, inplace=True)
	'''
	# Separa os dados dos PPGs filtrados pelo DECIL escolhido
	ppgs_select = set(df_sims['_id'].to_list() + df_sims['_id_SIM'].to_list())
	print('%s PPGs com Decil %s' % (len(list(ppgs_select)),decil))
	df = df[df['_id'].isin(ppgs_select)]
	df.sort_values(['SIMILARIDADE','_id'], inplace=True)
	'''
	#--------------------------------------------------------------------------
	# Identifica selecao de PPGs usada pela variedade de UFs, Regioues etc.
	all_ufs 	= df['sg_uf_programa'].drop_duplicates().to_list()
	all_regioes = df['nm_regiao'].drop_duplicates().to_list()
	all_areas 	= df['nm_area_avaliacao'].drop_duplicates().to_list()
	#conceitos	= df['cd_conceito_programa'].drop_duplicates().to_list()
	all_decis 	= df['DECIL_MAISSIM'].drop_duplicates().to_list()
	all_decis.sort(reverse=False)
	all_ufs.sort()
	all_regioes.sort()
	all_areas.sort()

	# Prepara escalas de cores e textos
	color_list  = 2 * ['green','magenta','orange','blue','gold','violet',\
					   'olive','red','darkgrey','fuchsia','brown', \
					   'indigo','darksalmon','darkolivegreen','deeppink',\
					   'darkblue','yellow','darkred',\
					   'black', 'pink', 'darkturquoise', \
			           'blueviolet', 'burlywood', 'cadetblue',\
			           'chartreuse', 'chocolate', 'coral', 'cornflowerblue']
	s_colors = pd.Series(color_list[:len(all_areas)], index=all_areas, name='cor')
	df = df.join(s_colors, on='nm_area_avaliacao')

	def ajusta_hovertext(df):
		texto 	= df['_id'] + '<br>' + df['nm_programa_ies'].str.title() + '<br>' +\
			      'IES: '+df['sg_entidade_ensino'].str.upper() + ' - ' + \
				  'Area: '+df['nm_area_avaliacao'].str.title() + '<br>' +\
				  'UF: ' +df['sg_uf_programa'].str.upper() + ' - ' + \
				  'Conceito: '+df['cd_conceito_programa'].astype('str')
		return texto

	texto_n = ajusta_hovertext(df)
	'''
	def inclui_edges_3D(df):
		edge_x = []
		edge_y = []
		edge_z = []
		edge_text = []
		for ppg_sim, SIM, x0, y0, z0 in df[['_id_SIM', 'SIMILARIDADE','x','y','z']].values:
			if ppg_sim in df.index:
				x1, y1, z1 = df.loc[ppg_sim][['x','y','z']].values
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

		return edge_x, edge_y, edge_z, edge_text

	def inclui_edges_2D(df):
		edge_x = []
		edge_y = []
		edge_text = []
		for ppg_sim, SIM, x0, y0 in df[['_id_SIM', 'SIMILARIDADE','x','y']].values:
			if ppg_sim in df.index:
				x1, y1 = df.loc[ppg_sim][['x','y']].values
				edge_x.append(x0)
				edge_x.append(x1)
				edge_x.append(None)
				edge_y.append(y0)
				edge_y.append(y1)
				edge_y.append(None)
				edge_text.append(str(SIM))

		return edge_x, edge_y, edge_text
	'''
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
		'''
		print(A)
		print(edge_text)
		print(B)
		print(fig.data[0])
		'''
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


# Monta tabela com lista de PPGs selecionados em plot Scatter
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



# EXPERIMENTO: gera grafo usando Networkx
def grafo_plotly(df=None, decil=None, k=.3, peso_area=1):

	if df is None:
		fname = 'SIMILARIDADE_DE_PPGS_None_None_COM_PPG_teses.csv'
		df=pd.read_csv(pasta['estudos']+fname, sep=';', encoding='latin-1')
	else:
		fname = 'UNICAS_SIMILARIDADES.csv'

	if decil is None:
		decil=(9,10)

	if type(decil) is int:
		df_sims = df[df['DECIL'] == decil][['_id','_id_SIM','SIMILARIDADE',\
											  'DECIL']]
	elif type(decil) is tuple:
		df_sims = df[(df['DECIL'] >= decil[0]) & (df['DECIL'] <= decil[1])]\
					  [['_id','_id_SIM','SIMILARIDADE','DECIL']]

	# Separa os dados dos PPGs filtrados pelo DECIL escolhido
	ppgs_select = set(df_sims['_id'].to_list() + df_sims['_id_SIM'].to_list())
	print('%s PPGs com Decil %s' % (len(list(ppgs_select)),decil))
	df = df[df['_id'].isin(ppgs_select)]
	df.sort_values(['SIMILARIDADE','_id'], inplace=True)

	'''
	df_unicos 	= pd.DataFrame()
	sims_unicos = []
	for i in range(len(df)):
		if set(df.iloc[i][['_id','_id_SIM']]) not in sims_unicos:
			df_unicos = df_unicos.append(df.iloc[i], ignore_index=True)
			sims_unicos.append(set(df.iloc[i][['_id','_id_SIM']]))

	# Grava versao com PPGs unicos
	df_unicos.to_csv(pasta['estudos']+'UNICAS_'+fname, sep=';', encoding='latin-1')
	'''
	#--------------------------------------------------------------------------

	G = nx.DiGraph()
	#G.add_nodes_from(df['_id'].to_list())
	G.add_nodes_from((df.iloc[i]['_id'],df.iloc[i][['cd_area_avaliacao', \
													'sg_entidade_ensino',\
													'pubs']].to_dict()) \
					 for i in range(len(df)))


	for area in df['cd_area_avaliacao'].drop_duplicates().to_list():
		df_area = df[df['cd_area_avaliacao']==area]
		ppgs = 	list(set(df_area['_id'].to_list() + df_area['_id_SIM'].to_list()))
		edges = []
		for p1 in ppgs:
			for p2 in ppgs:
				if (p1 != p2) and ((p1,p2) not in edges):
					edges.append((p1, p2, peso_area))
		G.add_weighted_edges_from(edges)

	wedges = []
	for i in range(len(df_sims)):
		a,b,w,s = df_sims.iloc[i][['_id','_id_SIM','DECIL','SIMILARIDADE']].to_list()
		data = {'similaridade': s, 'weight': w}
		wedges.append((a,b, data))

	G.add_edges_from(wedges)

	pos = nx.spring_layout(G, k=k, dim=3)


	# Prepara escalas de cores e textos
	color_list  = ['green','magenta','orange','blue','gold','violet',\
				   'olive','red','darkgrey','fuchsia','brown', \
				   'indigo','darksalmon','darkolivegreen','deeppink',\
				   'darkblue','yellow','darkred',\
				   'black', 'pink', 'darkturquoise', \
		           'blueviolet', 'burlywood', 'cadetblue',\
		           'chartreuse', 'chocolate', 'coral', 'cornflowerblue']
	area_list  = df['nm_area_avaliacao'].drop_duplicates().to_list()
	color_index = [area_list.index(df.iloc[i]['nm_area_avaliacao']) \
				   for i in range(len(df))]
	color_nodes = np.array(color_list)[color_index]
	#pos = nx.kamada_kawai_layout(G, scale=3)
	'''
	#pos_new = pos
	#print(list(pos.keys()))
	pdf = pd.DataFrame.from_dict(pos, orient='index')\
					  .assign(comb=lambda x: x[0]*10+x[1])\
					  .sort_values('comb')
	#return pos, pdf, df
	#print(pdf.index)
	#print(df[df['_id'].isin(pdf.index.to_list())]['_id'].to_list())
	pdf.index 	= df[df['_id'].isin(pdf.index.to_list())]['_id'].to_list()

	pos_new 	= {}
	for p in pdf.index:
		pos_new[p] = pdf.loc[p].to_numpy()[:2]
	'''
	# Identifica selecao de PPGs usada pela variedade de UFs, Regioues etc.
	ufs 		= df['sg_uf_programa'].drop_duplicates().to_list()
	regioes 	= df['nm_regiao'].drop_duplicates().to_list()
	areas 		= df['nm_area_avaliacao'].drop_duplicates().to_list()
	conceitos	= df['cd_conceito_programa'].drop_duplicates().to_list()


	texto_n	= df['_id'] + ' - ' + df['nm_programa_ies'].str.title() + '<br>' +\
		      df['sg_entidade_ensino'].str.upper() + ' - ' + \
			  df['nm_area_avaliacao'].str.title() + '<br>' +\
			  df['sg_uf_programa'].str.title() + ' - ' + \
			  df['cd_conceito_programa'].astype('str')

	#texto_e	= df['DECIL_MAISSIM'].astype('str') + ' - ' + df['SIMILARIDADE'].astype('str')

	# https://plotly.com/python/network-graphs/
	edge_x_vis = []
	edge_y_vis = []
	edge_z_vis = []
	edge_w_vis = []
	edge_text  = []
	for a, b, data in G.edges(data=True):
		x0, y0, z0 = pos[a]
		x1, y1, z1 = pos[b]
		w = data['weight']
		if w >= decil:
			edge_x_vis.append(x0)
			edge_x_vis.append(x1)
			edge_x_vis.append(None)
			edge_y_vis.append(y0)
			edge_y_vis.append(y1)
			edge_y_vis.append(None)
			edge_z_vis.append(z0)
			edge_z_vis.append(z1)
			edge_z_vis.append(None)
			edge_w_vis.append(w)
			edge_text.append(data['similaridade'])


	edge_trace_vis = go.Scatter3d(
		x=edge_x_vis, y=edge_y_vis, z=edge_z_vis,
		name='Similaridade',
		line=dict(width=3,
				  color='rgb(50,50,50)'),
		hoverinfo='text',
		text=edge_text,
		hovertemplate = "%{text}",
		mode='lines')
	'''
	edge_trace_ocu = go.Scatter3d(
		x=edge_x_ocu, y=edge_y_ocu, z=edge_z_ocu,
		line=dict(width=0,
				  color='rgb(255,255,255)'),
		hoverinfo='text',
		#text=texto_e,
		#hovertemplate = "%{text}",
		mode='lines')
	'''

	node_x 	= []
	node_y 	= []
	node_z 	= []
	pubs 	= []
	area 	= []
	ies 	= []
	for node, data in G.nodes(data=True):
		x, y, z = pos[node]
		ies.append(data['sg_entidade_ensino'])
		pubs.append(data['pubs'])
		area.append(data['cd_area_avaliacao'])
		node_x.append(x)
		node_y.append(y)
		node_z.append(z)

	node_trace = go.Scatter3d(
		x=node_x, y=node_y, z=node_z,
		mode='markers',
		name='PPGs',
		hoverinfo='text',
		text=texto_n,
		hovertemplate = "%{text}<br>"+
						"Pubs: %{marker.size:,.0f}</br>",
		marker={
				'color': color_nodes,
				#'colorscale': colorscale2,
				'sizemin': 3,
				'sizemode': 'area',
				'sizeref': .16,
				'size': pubs},
		#selected={'marker': {'color': 'black'}},
		#marker=dict(
			#showscale=True,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			#colorscale='YlGnBu',
			#reversescale=True,
			#color=df['cd_area_avaliacao'],
			#size=df['pubs'].div(6),
			#colorbar=dict(
			#	thickness=15,
			#	title='Area de Avaliacao',
			#	xanchor='left',
			#	titleside='right'
			#),
			line_width=1)

	fig = go.Figure(data=[edge_trace_vis, node_trace],
	             layout=go.Layout(
	                hovermode='closest',
	                margin=dict(b=20,l=5,r=5,t=40),
	                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
	                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
	                )

	fig.update_layout(
		        height=500,
				showlegend=True,
				template='plotly_white',
				plot_bgcolor='white',
		        dragmode='select',
				margin=dict(l=0, r=0, b=0, t=50),
				title='Similaridade entre PPGs')
	return fig



def desenha_grafo(G, pos, color_nodes=None):
	# https://plotly.com/python/network-graphs/
	edge_x_vis = []
	edge_y_vis = []
	edge_w_vis = []
	edge_x_ocu = []
	edge_y_ocu = []
	edge_w_ocu = []
	for a, b, data in G.edges(data=True):
		x0, y0 = pos[a]
		x1, y1 = pos[b]
		w = data['weight']
		if w > 1:
			edge_x_vis.append(x0)
			edge_x_vis.append(x1)
			edge_x_vis.append(None)
			edge_y_vis.append(y0)
			edge_y_vis.append(y1)
			edge_y_vis.append(None)
			edge_w_vis.append(w)
		else:
			edge_x_ocu.append(x0)
			edge_x_ocu.append(x1)
			edge_x_ocu.append(None)
			edge_y_ocu.append(y0)
			edge_y_ocu.append(y1)
			edge_y_ocu.append(None)
			edge_w_ocu.append(w)


	edge_trace_vis = go.Scatter(
		x=edge_x_vis, y=edge_y_vis,
		name='Similaridades',

		line=dict(width=2,
				  color='rgb(100,100,100)'),
		hoverinfo='text',
		#text=texto_e,
		#hovertemplate = "%{text}",
		mode='lines')

	edge_trace_ocu = go.Scatter(
		x=edge_x_ocu, y=edge_y_ocu,
		line=dict(width=1,
				  color='rgb(255,255,255)'),
		hoverinfo='text',
		#text=texto_e,
		#hovertemplate = "%{text}",
		mode='lines')


	node_x 	= []
	node_y 	= []
	pubs 	= []
	area 	= []
	ies 	= []
	for node, data in G.nodes(data=True):
		x, y = pos[node]
		ies.append(data['sg_entidade_ensino'])
		pubs.append(data['pubs'] / 3)
		area.append(data['cd_area_avaliacao'])
		node_x.append(x)
		node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		name='PPGs',
		hoverinfo='text',
		#text=texto_n,
		#hovertemplate = "%{text}<br>"+
		#				"Pubs: %{marker.size:,.0f}</br>",
		marker={
				'color': color_nodes,
				#'colorscale': colorscale2,
				'sizemin': 3,
				'sizemode': 'area',
				'sizeref': .16,
				#'size': df['pubs'].div(4),},
				'size': pubs},
		#selected={'marker': {'color': 'black'}},
		#marker=dict(
			#showscale=True,
			# colorscale options
			#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			#colorscale='YlGnBu',
			#reversescale=True,
			#color=df['cd_area_avaliacao'],
			#size=df['pubs'].div(6),
			#colorbar=dict(
			#	thickness=15,
			#	title='Area de Avaliacao',
			#	xanchor='left',
			#	titleside='right'
			#),
			line_width=1)

	fig = go.Figure(data=[edge_trace_ocu, edge_trace_vis, node_trace],
	             layout=go.Layout(
	                hovermode='closest',
	                margin=dict(b=20,l=5,r=5,t=40),
	                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
	                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
	                )

	fig.update_layout(
		        height=500,
				plot_bgcolor='white',
		        dragmode='select',
				margin=dict(l=0, r=0, b=0, t=50),
				title='Similaridade entre PPGs')
	return fig


"""

def grafo_nx(df, decil):
	#plt.style.use('tableau-colorblind10')
	plt.style.use('classic')

	#df = df.assign(PESO=df['SIMILARIDADE']*10/df['SIMILARIDADE'].max())

	#print(df, df.columns.array)
	#index = [i for i in range(len(df.index))]

	if type(decil) is int:
		df = df[df['DECIL'] == decil]
	elif type(decil) is tuple:
		df = df[(df['DECIL'] >= decil[0]) & (df['DECIL'] <= decil[1])]

	G = nx.Graph()
	G.add_nodes_from((df.iloc[i]['_id'],df.iloc[i][['nm_programa_ies', \
													'sg_entidade_ensino',\
													'pubs']].to_dict()) \
					 for i in range(len(df)))

	#edge_list = df[['_id','_id_SIM','DECIL']].to_numpy()
	G.add_weighted_edges_from(df[['_id','_id_SIM','DECIL']].to_numpy())

	#print(G.nodes(), G.edges())
	# Se for para trabalhar com o subgrafo de um PPG especifico...
	'''
	if ppg_filtrado != '':
		lista = np.append(df[df['CD_PPG'] == ppg_filtrado]['CD_PPG_SIM'],\
						 [ppg_filtrado])
		X = G.copy()
		#del G
		X = X.subgraph(lista)
		G = X.copy()
		#del X
		#print('G antes',G.edges())
		diferentes = [(s,o) for (s,o) in G.edges() \
					  if s != ppg_filtrado and o != ppg_filtrado]
		#print(len(diferentes), diferentes)
		G.remove_edges_from(diferentes)
		df= df[df['CD_PPG'] == ppg_filtrado]
		#print('G depois',G.nodes())
	'''
	#pos = nx.circular_layout(G, k=1.,)
	pos = nx.kamada_kawai_layout(G, scale=3)

	#pos = nx.spring_layout(G, k=1.)
	nx.draw_networkx_nodes(G, \
						   pos, \
						   node_size=3 * df['pubs'], \
						   node_color='w')

	#print(pos)
	#print(df[['CD_PPG', 'CD_PPG_SIM']])
	#print(G.nodes(data=True))
	'''
	# nodes de outras areas

	outra_area 	= df[df['AREA_SIM'].astype('int32') != int(self.area_nr)]['CD_PPG_SIM'].values
	tamanho 	= 30 * df[df['AREA_SIM'].astype('int32') != int(self.area_nr)]['PUBS_SIM'].values
	#outra_area = list(set(outra_area))
	#print(tamanho)
	if len(outra_area) != 0:
		nx.draw_networkx_nodes(G, \
							   pos, \
							   node_size=tamanho, \
							   nodelist=outra_area, \
							   node_color='r')
	# nodes da mesma area
	mesma_area 	= df[df['AREA_SIM'].astype('int32') == int(self.area_nr)]['CD_PPG_SIM'].values
	#mesma_area = list(set(mesma_area))
	tamanho 	= 30 * df[df['AREA_SIM'].astype('int32') == int(self.area_nr)]['PUBS_SIM'].values
	if len(mesma_area) != 0:
		nx.draw_networkx_nodes(G, \
							   pos, \
							   node_size=tamanho, \
							   nodelist=mesma_area, \
							   node_color='b')

	# nodes centrais/fontes
	#print(sources)
	print(pos)
	print(ppg_filtrado)
	print(G)
	nx.draw_networkx_nodes(G, \
						   pos, \
						   nodelist=[ppg_filtrado], \
						   node_color='y', \
						   node_size=1200)

	# edges
	for decil in range(10):
		edges_peso = [(s,t) for s,t,p in list(G.edges(data=True)) \
					  if p['weight'] == decil]
	'''
	for s,t,p in list(G.edges(data=True)):
		#print(s,t,p)
		nx.draw_networkx_edges(G, \
							   pos, \
							   edgelist=[(s,t)],\
							   width=p['weight']/2,\
							   alpha=p['weight'] / 20,\
							   arrow=True)
	print(len(list(G.edges(data=True))))
	'''
	colors = range(len(G.edges))
	nx.draw_networkx_edges(G, \
						   pos, \
						   edge_color=colors, \
						   edge_cmap=plt.cm.Blues \
						   )
	'''
	#plt.axis('off')
	#plt.show()
"""