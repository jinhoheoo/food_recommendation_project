import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    foodIdx = [i[0] for i in simScore]
    recfoodList = df_reviews.iloc[foodIdx, 0]
    return recfoodList[1:11]

df_reviews = pd.read_csv('./cleaned_one_review.csv')
Tfidf_matrix = mmread('./models/Tfidf_food_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# #index 이용
# ref_idx = 10
# print(df_reviews.iloc[ref_idx, 0])
# consine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
# print(consine_sim[0])
# print(len(consine_sim))
# recommandation = getRecommendation(consine_sim)
# print(recommandation)

# keyword 이용

embedding_model = Word2Vec.load('./models/word2vec_food_review.model')
keyword = '삼겹살'
sim_word = embedding_model.wv.most_similar(keyword, topn=10)
words = [keyword]
for word, _ in sim_word:
    words.append(word)
setence = []
count = 10
for word in words:
    setence = setence + [word] * count
    count -= 1
setence = ' '.join(setence)
print(setence)
setence_vec = Tfidf.transform([setence])
cosine_sim = linear_kernel(setence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)

print(recommendation)

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    foodIdx = [i[0] for i in simScore]
    recfoodList = df_reviews.iloc[foodIdx, 0]
    return recfoodList[1:11]

df_reviews = pd.read_csv('./cleaned_naver_reviews.csv')
Tfidf_matrix = mmread('./models/Tfidf_naver_review.mtx').tocsr()
with open('./models/tfidf_naver.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 영화 index 이용
# ref_idx = 10
# print(df_reviews.iloc[ref_idx, 0])
# consine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
# print(consine_sim[0])
# print(len(consine_sim))
# recommandation = getRecommendation(consine_sim)
# print(recommandation)

# keyword 이용

embedding_model = Word2Vec.load('./models/word2vec_naver_review.model')
keyword = '삼겹살'
sim_word = embedding_model.wv.most_similar(keyword, topn=10)
words = [keyword]
for word, _ in sim_word:
    words.append(word)
setence = []
count = 10
for word in words:
    setence = setence + [word] * count
    count -= 1
setence = ' '.join(setence)
print(setence)
setence_vec = Tfidf.transform([setence])
cosine_sim = linear_kernel(setence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)

print(recommendation)




