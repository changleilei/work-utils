# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/5/15 10:56 AM
==================================="""

# 导入所需库
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 将分词结果转换为字符串
texts = """AIGC， AGI， StableDiffusion， OpenAI， ChatGPT， ChatGLM， LLama， Vicuna， 大模型，LLM，midjourney，
AI 赋能， 智能聊天，智能客服，人工智能，PaLM2，多模态，人机对话，星火大模型，文心一言"""

texts = [text.strip() for text in texts.split('，')]
words_str = ' '.join(texts)
# 生成词云
wc = WordCloud(font_path='./SimHei.ttf',  background_color='white', width=800, height=600).generate(words_str)

# 显示词云
plt.imshow(wc)
plt.axis('off')
plt.show()