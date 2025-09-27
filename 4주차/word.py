import pandas as pd
import wikipedia
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1.위키피디아에서 텍스트 가져오기
wikipedia.set_lang("ko") # 한글 위키피디아 설정
text = wikipedia.page("한양여자대학교").content # 예: 한양여자대학교 문서

# 2.pandas로 텍스트 데이터프레임 생성(분석 및 전처리를 위해)
df = pd.DataFrame({'text':[text]})

# 3.텍스트 전차리
words = ' '.join(df['text']).replace('\n',' ').replace('==',' ')
# 필요 시 불용어 제거, 한글 형태소 분석 추가 가능

# 4.워드 클라우드 생성
wordcloud = WordCloud(
    font_path='NanumGothic.ttf',
    width=800, height=400,
    background_color='white'
).generate(words)

# 5.시각화
plt.figure(figsize=(12,6)) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('한양여자대학교 위키피디아 워드클라우드')
plt.show()