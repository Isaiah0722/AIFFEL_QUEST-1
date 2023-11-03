# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이윤상님
- 리뷰어 : 이진영


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네. 문제를 깔끔히 해결했습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. 함수 하나하나 주석을 달아주셔서 이해하기 쉬웠습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 잘 작동하기 때문에 에러를 유발할 가능성은 없다고 생각합니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네. 궁금한 것을 물어봤을 때 하나하나 설명해주셨습니다.
- [X] 코드가 간결한가요?
  > 함수를 간결하게 만들고 마지막에 while문을 보고 소름이 돋았습니다. 제너레이터 마지막 출력값 None을 제외하는 방법을 알게되어 유익했습니다. 

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
from time import sleep                                                                     # time 모듈의 sleep 함수를 import함.


def show_fish_movement_comprehension(fish_list):                                           # show_fish_movement_comprehension함수를 지정
    return [f"{i['이름']} is swimming at a speed of {i['speed']} m/s" for i in fish_list]   # '이름'과 'spped'를 for 문을 이용해서 fish_list에서 가져옴.

                                                                                           # 컴프리헨션을 이용하여 for문을 작성함.
def show_fish_movement_Generator(fish_list):                                               # show_fish_movement-Generator 함수를 지정
    for i in fish_list:                                                                    # for문을 이용하면서 yield
        yield (f"{i['이름']} is swimming at a speed of {i['speed']} m/s")

fish_list = [{"이름": "Nemo", "speed": 3}, {"이름": "Dory", "speed": 5}]                     # fish_list 지정

print("Using Comprehension : ")
for i in show_fish_movement_comprehension(fish_list):                                      # for문을 이용해서 반복하고 sleep함수를 이용해서 2초 뒤에 print함수가 실행되도록 작성함.
    sleep(2)
    print(i)

fish_gen = show_fish_movement_Generator(fish_list)                                         # 제너레이터 객체 생성
sleep(2)
print("Using Generator :")
while True:                                                                                # 제너레이터 객체의 값을 하나씩 출력합니다.
    try:
        sleep(2)
        fish_movement = next(fish_gen)
        print(fish_movement)
    except StopIteration:                                                                  # 제너레이터에서 더 이상 반환한 값이 없을때 break
        break
```

# 참고 링크 및 코드 개선
'from time import sleep as sl'를 사용하면 sleep함수를 더 간결하게 쓸 수 있을 것 같습니다. 
