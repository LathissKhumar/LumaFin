from src.reranker.model import Reranker
r=Reranker()
cands=[{'category':'Food','similarity':0.9,'merchant':'Rest A','amount':20.0},{'category':'Food','similarity':0.8,'merchant':'Rest B','amount':25.0},{'category':'Transport','similarity':0.7,'merchant':'Uber','amount':15.0}]
feats=r._features_per_category('restaurant',cands)
print('Food len',len(feats['Food']))
print('Transport len',len(feats['Transport']))
print('Food feat',feats['Food'])
