# -*- coding: utf8 -*-

import json

from django.http import JsonResponse
from django.views import View

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class Recommend(View):
    def post(self, request):
        data = json.loads(request.body)
        title = data["title"]
        print(title)
        category = data["category"]
        print(category)
        if category == 'IT 모바일':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\IT 모바일_YES베스트.csv')
        elif category == '경제 경영':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\경제 경영_YES베스트.csv')
        elif category == '역사':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\역사_YES베스트.csv')
        elif category == '자연과학':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\자연과학_YES베스트.csv')
        elif category == '소설_시_희곡':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\소설_시_희곡_YES베스트.csv')
        elif category == '사회 정치':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\사회 정치_YES베스트.csv')
        elif category == '만화_라이트노벨':
            csv_file = pd.read_csv('\\home\\ubuntu\\csv\\만화_라이트노벨_YES베스트.csv')

        csv_file = csv_file[['상품명', 'ISBN', '설명']]

        ################## 설명으로 추천하기 ###################
        count_vector = CountVectorizer(ngram_range=(1, 5))
        c_vector_overview = count_vector.fit_transform(csv_file['설명'].values.astype('U'))
        c_vector_overview.shape

        overview_c_sim = cosine_similarity(c_vector_overview, c_vector_overview).argsort()[:, ::-1]
        overview_c_sim.shape

        def get_recommend_book_list_overview(df, book_title, top=3):
            target_book_index = df[df['상품명'] == book_title].index.values
            sim_index = overview_c_sim[target_book_index, :top].reshape(-1)
            sim_index = sim_index[sim_index != target_book_index]
            result = df.iloc[sim_index]
            return result


        ################## 제목으로 추천하기 ###################
        count_vector3 = CountVectorizer(ngram_range=(1, 3))
        c_vector_title = count_vector3.fit_transform(csv_file['상품명'])
        c_vector_title.shape

        title_c_sim = cosine_similarity(c_vector_title, c_vector_title).argsort()[:, ::-1]
        title_c_sim.shape

        def get_recommend_book_list_title(df, book_title, top=3):
            target_book_index = df[df['상품명'] == book_title].index.values
            sim_index = title_c_sim[target_book_index, :top].reshape(-1)
            sim_index = sim_index[sim_index != target_book_index]
            result = df.iloc[sim_index]
            return result

        frames = [get_recommend_book_list_title(csv_file, book_title=title),
                  get_recommend_book_list_overview(csv_file, book_title=title)]
        result = pd.concat(frames)
        result = result[['ISBN']]
        result = result.drop_duplicates()
        result.reset_index(inplace=True, drop=True)

        js = result.to_json()
        return JsonResponse(js,safe=False)
