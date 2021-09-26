# Using pretrained models

* 예로, mask filling을 수행하는 French-based model을 찾아보자!

![fre](https://user-images.githubusercontent.com/59636424/134816147-c300a002-b708-4ca3-b384-6ba4b85e44e2.PNG)

-> 그림을 통해 camembert-base checkpoint 선정!

~~~
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
~~~

camembert-base chekcpoint는 fill-mask pipeline에 loading 되었다!

-> 그러나, text-classification pipeline에 load하면, camembert-base의 head는 그 task를 이해할 수 없을 것이다!

**따라서, 적절한 checkpoints를 HuggingFace Hub에서 task selector를 사용함으로써 추천받아라!**

![ttt](https://user-images.githubusercontent.com/59636424/134816264-2113ef8e-bc00-4929-8b76-0f067776addd.PNG)

~~~
from transformers import CamembertTokenizer, CamembertForMaskedLM 

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
~~~

~~~
ImportError: 
CamembertTokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
~~~

이러한 에러가 발생하므로 Auto class를 사용하여 설계에 구애받지 않는 것이 좋다!

Auto class를 통해, checkpoint를 쉽게 전환할 수 있다!!

~~~
from transformers import AutoTokenizer, AutoModelForMaskedLM 

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
~~~

