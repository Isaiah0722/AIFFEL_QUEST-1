# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 이윤상
- 리뷰어 : 김소연

# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 1. 인물모드 사진을 성공적으로 제작하였다.
             - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/a7065898-d2ab-4246-92aa-f19fb21de4fa)
        - 2. 제작한 인물모드 사진들에서 나타나는 문제점을 정확히 지적하였다.
             - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/62e11f0a-a6d8-43a8-a837-1e414c3a0b83)
             - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/ec4a997d-77b5-4037-8bae-1525e44d83a0)

        - 3. 인물모드 사진의 문제점을 개선할 수 있는 솔루션을 적절히 제시하였다.
             - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/b427494d-0339-4223-97c5-d4edc98b495a)

    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/ec94e24b-fb58-4b64-8116-7ecf88603adb)
        - 각 변수와 코드들이 어떠한 기능을 수행하고 있는지 이해할 수 있었습니다.

        
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/47c52d9b-8929-490d-aa56-4a02e12e46fb)
        - 다른 인물사진과 동물사진을 사용하여 추가 실험을 진행하였습니다.

        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/b1b230f6-7993-4388-9df6-9ef617ba3d16)

        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - ![image](https://github.com/elliekim9881/AIFFEL_QUESTys/assets/137244968/0b437f17-6395-4695-abeb-d1ed04fc13bc)
        - 라이브러리를 간결하게 불러올 수 있도록 as 사용.



# 참고 링크 및 코드 개선
- 이미지별로 세그멘테이션, 블러, 크로마키 효과를 중복적으로 적용하고 있는데 해당 명령어들이 함수화 되어있지 않아 매번 다시 코드를 작성하고 있는 형식으로 쓰여 있다.
    - 블러, 크로마키 효과를 함수로 선언하여 처리한다면 더 간결하게 쓰일 수 있을 것이라 생각된다.
```python
#개선 예시

...

def ChromaKey2(img_path, img_bg_path, category):
    segvalues, output = model_segmentation.segmentAsPascalvoc(img_path)
    indices = np.where(LABEL_NAMES == category)[0]
    
    if indices.size == 0:
        print('잘못된 category입니다')
        print(LABEL_NAMES)
        return
    if indices[0] not in segvalues['class_ids']:
        print(category + '는 찾지 못했습니다.')
        return
    
    seg_map = np.all(output == colormap[indices], axis=-1)
    
    # 여기에서 seg_map에 오프닝 연산 적용
    seg_map = apply_opening(seg_map, kernel_size=5)
    
    img_orig = cv2.imread(img_path)
    img_bg = cv2.imread(img_bg_path)
    img_bg = cv2.resize(img_bg, (img_orig.shape[1], img_orig.shape[0]))
    img_mask = seg_map.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    img_mask = cv2.dilate(img_mask, kernel, iterations=3)
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    
    stickObject(img_orig, img_bg, img_mask_color)
...


```
- 회고에도 쓰여있는 것과 같이 제안한 해결방법을 토대로 장비가 갖추어진다면 해결해 보고 추가해볼 수 있으면 좋겠다.

### 참고링크
- 리뷰어 작성
https://github.com/elliekim9881/Exploration/blob/main/EX03/EX03.ipynb
