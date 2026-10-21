# 대기 CO2 농도 시계열 트렌드 분석

마우나로아 관측소의 월별 대기 CO2 농도(1958-03 ~ 2026-08, 822개월)를 분석한 프로젝트입니다.
분석 내용과 인사이트는 [REPORT.md](REPORT.md)에 있습니다.

## 폴더 구조

```
co2-trend-analysis/
├── data/
│   └── co2_mlo_monthly_raw.csv   # 원본 데이터
├── images/                       # analysis.py 실행 시 생성되는 그래프 6개
├── results/
│   ├── co2_clean.csv             # 정제 + 파생변수(이동평균, YoY) 포함 데이터
│   └── summary.json              # 리포트에 쓴 모든 수치
├── analysis.py
├── REPORT.md
├── README.md
└── requirements.txt
```

## 실행 방법

Python 3.10 이상

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python analysis.py
```

실행하면 `images/`의 그래프와 `results/summary.json`이 다시 생성됩니다.
한글 폰트는 Windows의 맑은 고딕, macOS의 AppleGothic, Linux의 Noto Sans CJK 중 있는 것을 자동으로 사용합니다.

## 데이터 출처 / 수집 방법 / 라이선스

- 원 출처: NOAA Global Monitoring Laboratory, Mauna Loa CO2 월평균 (https://gml.noaa.gov/ccgg/trends/data.html)
- 수집 경로: NOAA 데이터를 CSV로 정리해 둔 공개 저장소 datasets/co2-ppm
  (https://raw.githubusercontent.com/datasets/co2-ppm/main/data/co2-mm-mlo.csv)를 내려받아 `data/`에 저장
- 라이선스: NOAA 데이터는 미국 정부 공개 데이터로 자유롭게 사용할 수 있으나, NOAA는 사용 시 출처 표기와
  최신 데이터 확인을 요청합니다. 최근 값은 사후 보정으로 바뀔 수 있습니다.
