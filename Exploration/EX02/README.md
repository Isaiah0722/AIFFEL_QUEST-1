# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이윤상
- 리뷰어 : 김소연


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다.
          ```python
          # 등장 빈도수가 6회 미만인 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인
            threshold = 6
            total_cnt = len(tar_tokenizer.word_index) # 단어의 수
            rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
            total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
            rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
            
            # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
            for key, value in tar_tokenizer.word_counts.items():
                total_freq = total_freq + value
            
                # 단어의 등장 빈도수가 threshold보다 작으면
                if(value < threshold):
                    rare_cnt = rare_cnt + 1
                    rare_freq = rare_freq + value
            
            print('단어 집합(vocabulary)의 크기 :', total_cnt)
            print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
            print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
            print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
            print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
            단어 집합(vocabulary)의 크기 : 29903
            등장 빈도가 5번 이하인 희귀 단어의 수: 19668
            단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 10235
            단어 집합에서 희귀 단어의 비율: 65.77266495000501
            전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 5.896234529480391
            tar_vocab = 10000
            tar_tokenizer = Tokenizer(num_words=tar_vocab)
            tar_tokenizer.fit_on_texts(decoder_input_train)
            tar_tokenizer.fit_on_texts(decoder_target_train)
            
            # 텍스트 시퀀스를 정수 시퀀스로 변환
            decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train)
            decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)
            decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
            decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)
            
            # 잘 변환되었는지 확인
            print('input')
            print('input ',decoder_input_train[:5])
            print('target')
            print('decoder ',decoder_target_train[:5])
            input
            input  [[1, 91, 133, 168, 623, 42, 5549, 375], [1, 5, 48, 5302, 3079, 59], [1, 222, 9375, 1273, 6, 15, 815, 851], [1, 776, 1550, 2410, 4720, 1155, 49], [1, 8, 5, 2158, 4721, 23, 73]]
            target
            decoder  [[91, 133, 168, 623, 42, 5549, 375, 2], [5, 48, 5302, 3079, 59, 2], [222, 9375, 1273, 6, 15, 815, 851, 2], [776, 1550, 2410, 4720, 1155, 49, 2], [8, 5, 2158, 4721, 23, 73, 2]]
            # 훈련 데이터와 테스트 데이터에 대해서 요약문의 길이가 1인 경우의 인덱스를 각각 drop_train과 drop_test에 라는 변수에 저장
            # 이 샘플들은 모두 삭제함
            drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
            drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]
            
            print('삭제할 훈련 데이터의 개수 :', len(drop_train))
            print('삭제할 테스트 데이터의 개수 :', len(drop_test))
            
            encoder_input_train = [sentence for index, sentence in enumerate(encoder_input_train) if index not in drop_train]
            decoder_input_train = [sentence for index, sentence in enumerate(decoder_input_train) if index not in drop_train]
            decoder_target_train = [sentence for index, sentence in enumerate(decoder_target_train) if index not in drop_train]
            
            encoder_input_test = [sentence for index, sentence in enumerate(encoder_input_test) if index not in drop_test]
            decoder_input_test = [sentence for index, sentence in enumerate(decoder_input_test) if index not in drop_test]
            decoder_target_test = [sentence for index, sentence in enumerate(decoder_target_test) if index not in drop_test]
            
            print('훈련 데이터의 개수 :', len(encoder_input_train))
            print('훈련 레이블의 개수 :', len(decoder_input_train))
            print('테스트 데이터의 개수 :', len(encoder_input_test))
            print('테스트 레이블의 개수 :', len(decoder_input_test))
            삭제할 훈련 데이터의 개수 : 1
            삭제할 테스트 데이터의 개수 : 1
            훈련 데이터의 개수 : 78596
            훈련 레이블의 개수 : 78596
            테스트 데이터의 개수 : 19648
            테스트 레이블의 개수 : 19648
          ```
        - 텍스트 요약모델이 성공적으로 학습되었음을 확인하였다.
          ```python
          308/308 [==============================] - 33s 107ms/step - loss: 2.2572 - val_loss: 3.1424
          Epoch 00025: early stopping
          # 훈련 데이터의 손실과 검증 데이터의 손실이 줄어드는 과정을 시각화
          plt.plot(history.history['loss'], label='train')
          plt.plot(history.history['val_loss'], label='test')
          plt.legend()
          plt.show()
          ```
        - Extractive 요약을 시도해 보고 Abstractive 요약 결과과 함께 비교해 보았다.
          ```python
          # 추상적 요약을 하는 함수
            def textSummary(text) :
                text_preprocess = preprocess_sentence(text)
                text_preprocess = src_tokenizer.texts_to_sequences([text])
                text_preprocess = pad_sequences(text_preprocess, maxlen=text_max_len, padding='post')
                summa = decode_sequence(text_preprocess)
                return summa
            for i in range(50, 100):
                print("원문 :", data_summa['text'][i])
                print("실제 요약 :", data_summa['headlines'][i])
                print("추상 요약 :", textSummary(data_summa['text'][i]))
                print("추출 요약 :", summarize(data_summa['text'][i], ratio=0.3))
                print("\n")
          ```
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
      - 불용어 처리 function에서 각각의 기능들에 대한 주석을 잘 작성하였다.
        ```python
        # 데이터 전처리 함수
        def preprocess_sentence(sentence, remove_stopwords=True):
            sentence = sentence.lower() # 텍스트 소문자화
            sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
            sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
            sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
            sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
            sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
            sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
            sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah

        # 불용어 제거 (Text)
        if remove_stopwords:
            tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
        # 불용어 미제거 (Summary)
        else:
            tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
        return tokens
        ```
        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 추출 요약부분에서 더 나은 성능을 내는 ratio를 찾기 위해 값을 변경히보며 여러번 실험을 진행하였다.
          ```python
          for i in range(50, 100):
            print("원문 :", seq2text(encoder_input_test[i]))
            print("실제 요약 :", seq2headlines(decoder_input_test[i]))
            print("추상 요약 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
            print("추출 요약 :", summarize(data_summa['text'][i], ratio=0.4))  # ratio값을 0.4로 조정
            print("\n")
          ...

          for i in range(50, 100):
            print("원문 :", seq2text(encoder_input_test[i]))
            print("실제 요약 :", seq2headlines(decoder_input_test[i]))
            print("추상 요약 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
            print("추출 요약 :", summarize(data_summa['text'][i], ratio=0.7))  # ratio값을 0.7로 조정
            print("\n")

          ...
          
          for i in range(50, 100):
            print("원문 :", data_summa['text'][i])
            print("실제 요약 :", data_summa['headlines'][i])
            print("추상 요약 :", textSummary(data_summa['text'][i]))
            print("추출 요약 :", summarize(data_summa['text'][i], ratio=0.3))
            print("\n")
          
          ```
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 추출요약 ratio실험과정에서 있었던 특징들에 대한 회고가 잘 진행되었다.
          ```
          기계는 같은 의미라고 생각하지만, 인간이 보기에 같은 의미라고 볼 수 없는 요약들이 다수 존재했다. NLP 작업을 통한 요약은 최종적인 loss 값이 낮게 나오더라도 인간이 이해하지 못하거나 잘못된 내용을 전달할 수도 있겠다는 생각이 들었다. 마지막에 summa를 이용하는 과정에서 ratio 값에 따라 추출 요약이 나오거나 나오지 않는 경우가 있었다. 이는 요약을 하는 과정에서 ratio의 값이 너무 낮으면 문법적으로 올바른 내용을 만들기도 어렵고 내용상으로도 요약이라기보다는 키워드 몇 개를 추출하는 형태로 코드가 실행되어 요약이라고 할 수 없는 경우가 발생하게 된다. 따라서 ratio 값을 적절히 조절하여 사용자가 읽기 편하면서 추출 요약이 실행되는 최적의 ratio 값을 찾아서 대입해야 한다.
          ```
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 반복적으로 사용되어야 했던 추출요약 함수를 작성하여 반복적으로 코드를 쓸 필요가 없도록 만들었다.
          ```python
          # 추출 요약을 하는 함수
            def textSummary(text) :
                text_preprocess = preprocess_sentence(text)
                text_preprocess = src_tokenizer.texts_to_sequences([text])
                text_preprocess = pad_sequences(text_preprocess, maxlen=text_max_len, padding='post')
                summa = decode_sequence(text_preprocess)
                return summa
          ```


# 참고 링크 및 코드 개선
```

회고부분에서 'NLP 작업을 통한 요약은 최종적인 loss 값이 낮게 나오더라도 인간이 이해하지 못하거나 잘못된 내용을 전달할 수도 있겠다는 생각이 들었다' 라는 정리가 있었지만,
이해할 수 없게 요약된 내용들에 대한 명시가 없어서 어떤 부분이 잘 안되어서 해당 내용을 작성 하였는지 알 수 없었다. 해당 내용이 추가될 수 있다면 좋을 것 같다.

```
