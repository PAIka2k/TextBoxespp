## TextBoxes++ Tutorial
---
본 페이지는 https://github.com/mvoelk/ssd_detectors 의 코드를 차용해 만든 [TextBoxes++](https://arxiv.org/pdf/1801.02765.pdf)논문 튜토리얼입니다. scripts 폴더내의 notebook을 실행시키면 튜토리얼을 시작하실 수 있습니다.

### DataSet
---
튜토리얼에 사용된 데이터 셋은 [Synthtext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)데이터 셋으로, 7,266,866개의 단어들로 이루어진 858,750장의 합성 이미지 파일이 200개의 폴더에 나누어져 담겨있으며 Ground-truth annotation은 gt.mat파일에 담겨있습니다.<br><br>
데이터셋은 다운받으신 후 `data` 폴더에 넣어주세요.

### Expected results
---
본 튜토리얼을 마쳤을 때 얻을 수 있는 결과값입니다.

<img src="https://i.imgur.com/YgA4EJe.png" width="800">