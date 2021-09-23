# Transformers, what can they do?

Transforemr library에 기본적은 object는 **pipeline**이다! 

pipeline은 필수적인 전처리 및 후처리와 함께 모델을 연결해준다!

기본적으로 pipeline은 자동으로 fine-tuning된 모델을 가져와 task 시행! 

**모델 1번 다운 시, 다시 실행하면 캐시된 모델이 사용되니 다시 다운 필요 X**

**가능한 pipeline**

> * feature-extraction
> * fill-mask
> * ner
> * QA
> * 요약
> * 텍스트 생성
> * 번역
> * zero-shot-classification

* 감정 분석 pipeline 생성 예시

~~~
from transformers import pipeline

classifier = pipeline("sentiment-analysis") # 감정분석 pipeline 생성
classifier("I've been waiting for a HuggingFace course my whole life.") # 해당 문장 감정 분석 및 score 도출
~~~

~~~
classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
]) #여러 문장도 가능!
~~~

**pipeline에 text를 통과시킬 시, 주된 step!**

1. text는 모델이 이해할 수 있는 format으로 전처리 된다!

2. 전처리된 input은 model에 통과!

3. 모델 예측값은 후처리 되어 이해 가능할 것이다!

## 1. Zero-shot classification

**라벨이 부착되지 않은 텍스트 분류하는 어려운 과제 해결!**

사전 모델의 label에 의존하지 않는다! => 문장을 음수 or 양수로 분류 / 다른 레이블 집합을 사용하여 텍스트 분류

fine-tuning이 필요없고 원하는 레이블 list에 대한 확률 점수를 직접 반환!

* Zero-shto classification 예시

~~~
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
~~~

## 2. Text generation

**프롬프트를 제공하면 모델이 나머지 텍스트를 생성하여 프롬프트를 자동 완료**

* Text generation 예시

~~~
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
~~~

'num_return_sequences' 인자로 많은 sequence들을 얼마나 생성하는지 컨트롤 가능

'max_length' 인자로 output text의 총 길이를 컨트롤 가능

## 3. Using any model from the Hub in a pipeline

distilgpt2 모델을 load 해보자!

~~~
from transformers import pipeline

generator = pipeline("text-generation",model = "distilgpt2")
generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2)
~~~

## 4. The Inference API

HuggingFace Website에서 이 api를 통해 직접 test가 가능!

## 5. Mask filling

text에 공백(<MASK> 부분 word로 채우기!) 채우기!

top_k는 score 높은 k개의 단어 후보를 출력해준다!
  
* Mask filling 예시

~~~
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
~~~
  
## 6. Named entity recognition
  
**NER은 모델의 입력 텍스트의 어떤 부분이 persons, locations, organization인지 판별하는 작업**

NER 시, **grouped_entities=True**하면 pipeline에 동일한 entity에 해당하는 부분은 regroup을 시킨다! ex) Hugging + Face => ORG / False 할 시에는 Hugging이 더 작은 단어들로 나눠서 보인다.(Wordpiece Embedding 느낌)

* Named entity recognition 예시  

~~~
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
~~~
  
## 7. Question answering

주어진 context로부터 question 대답을 추출
  
* QA 예시

~~~
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn"
)
~~~

## 8. Summarization

**max_length** or **min_length**를 통해서 text 생성을 조정할 수 있다!
  
* Summarization 예시
  
~~~
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer("""
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
""")
~~~
  
## 9. Translation

"translation_en_to_fr"와 같이 task 이름에 언어 쌍을 제공하면 defalut model을 사용 가능!

마찬가지로 "max_length" or "min_length"로 text 생성과 요약 길이 조정 가능!
  
* Translation 예시

~~~
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
~~~
