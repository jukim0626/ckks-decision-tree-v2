# Pima Diabetes Decision Tree Rules

- max_depth setting: `20`
- actual tree depth: `13`

```python
if glucose <= 154.5000 (standardized <= 1.0951):
    if bmi <= 27.3500 (standardized <= -0.7477):
        if pregnancies <= 12.5000 (standardized <= 2.6214):
            if glucose <= 133.5000 (standardized <= 0.3946):
                if bmi <= 26.9000 (standardized <= -0.8137):
                    if insulin <= 48.5000 (standardized <= -1.1335):
                        if insulin <= 45.5000 (standardized <= -1.1716):
                            return class=0 (diabetes_prob=0.000, samples=9, gini=0.000)
                        else:  # insulin > 45.5000
                            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                    else:  # insulin > 48.5000
                        return class=0 (diabetes_prob=0.000, samples=94, gini=0.000)
                else:  # bmi > 26.9000
                    if bmi <= 27.0500 (standardized <= -0.7917):
                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                    else:  # bmi > 27.0500
                        return class=0 (diabetes_prob=0.000, samples=5, gini=0.000)
            else:  # glucose > 133.5000
                if pregnancies <= 2.5000 (standardized <= -0.3984):
                    return class=0 (diabetes_prob=0.000, samples=8, gini=0.000)
                else:  # pregnancies > 2.5000
                    if pregnancies <= 5.0000 (standardized <= 0.3566):
                        return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                    else:  # pregnancies > 5.0000
                        return class=0 (diabetes_prob=0.000, samples=4, gini=0.000)
        else:  # pregnancies > 12.5000
            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
    else:  # bmi > 27.3500
        if age <= 28.5000 (standardized <= -0.4116):
            if glucose <= 127.5000 (standardized <= 0.1944):
                if bmi <= 45.4000 (standardized <= 1.8995):
                    if diabetes_pedigree <= 0.5095 (standardized <= 0.0972):
                        if blood_pressure <= 83.5000 (standardized <= 0.9262):
                            if diabetes_pedigree <= 0.1975 (standardized <= -0.8482):
                                if diabetes_pedigree <= 0.1945 (standardized <= -0.8573):
                                    if glucose <= 118.0000 (standardized <= -0.1225):
                                        return class=0 (diabetes_prob=0.000, samples=17, gini=0.000)
                                    else:  # glucose > 118.0000
                                        return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                                else:  # diabetes_pedigree > 0.1945
                                    return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                            else:  # diabetes_pedigree > 0.1975
                                return class=0 (diabetes_prob=0.000, samples=61, gini=0.000)
                        else:  # blood_pressure > 83.5000
                            if diabetes_pedigree <= 0.2825 (standardized <= -0.5906):
                                if diabetes_pedigree <= 0.2465 (standardized <= -0.6997):
                                    return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                else:  # diabetes_pedigree > 0.2465
                                    return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                            else:  # diabetes_pedigree > 0.2825
                                return class=0 (diabetes_prob=0.000, samples=3, gini=0.000)
                    else:  # diabetes_pedigree > 0.5095
                        if bmi <= 32.7000 (standardized <= 0.0369):
                            if pregnancies <= 3.5000 (standardized <= -0.0964):
                                return class=0 (diabetes_prob=0.000, samples=22, gini=0.000)
                            else:  # pregnancies > 3.5000
                                return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                        else:  # bmi > 32.7000
                            if bmi <= 34.3000 (standardized <= 0.2716):
                                if glucose <= 79.0000 (standardized <= -1.4233):
                                    return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                else:  # glucose > 79.0000
                                    return class=1 (diabetes_prob=1.000, samples=5, gini=0.000)
                            else:  # bmi > 34.3000
                                if glucose <= 118.5000 (standardized <= -0.1058):
                                    if glucose <= 103.5000 (standardized <= -0.6061):
                                        if insulin <= 112.5000 (standardized <= -0.3203):
                                            return class=0 (diabetes_prob=0.000, samples=5, gini=0.000)
                                        else:  # insulin > 112.5000
                                            return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                                    else:  # glucose > 103.5000
                                        if glucose <= 110.5000 (standardized <= -0.3726):
                                            return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                                        else:  # glucose > 110.5000
                                            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                else:  # glucose > 118.5000
                                    return class=0 (diabetes_prob=0.000, samples=7, gini=0.000)
                else:  # bmi > 45.4000
                    if pregnancies <= 2.5000 (standardized <= -0.3984):
                        return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                    else:  # pregnancies > 2.5000
                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
            else:  # glucose > 127.5000
                if blood_pressure <= 73.0000 (standardized <= 0.0701):
                    if pregnancies <= 0.5000 (standardized <= -1.0023):
                        return class=1 (diabetes_prob=1.000, samples=7, gini=0.000)
                    else:  # pregnancies > 0.5000
                        if blood_pressure <= 69.0000 (standardized <= -0.2560):
                            if blood_pressure <= 59.0000 (standardized <= -1.0713):
                                return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                            else:  # blood_pressure > 59.0000
                                return class=0 (diabetes_prob=0.000, samples=6, gini=0.000)
                        else:  # blood_pressure > 69.0000
                            if bmi <= 31.8000 (standardized <= -0.0951):
                                return class=1 (diabetes_prob=1.000, samples=4, gini=0.000)
                            else:  # bmi > 31.8000
                                return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                else:  # blood_pressure > 73.0000
                    if bmi <= 43.4000 (standardized <= 1.6062):
                        if skin_thickness <= 32.5000 (standardized <= 0.3892):
                            return class=0 (diabetes_prob=0.000, samples=10, gini=0.000)
                        else:  # skin_thickness > 32.5000
                            if diabetes_pedigree <= 0.3730 (standardized <= -0.3164):
                                return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                            else:  # diabetes_pedigree > 0.3730
                                return class=0 (diabetes_prob=0.000, samples=3, gini=0.000)
                    else:  # bmi > 43.4000
                        return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
        else:  # age > 28.5000
            if glucose <= 107.5000 (standardized <= -0.4727):
                if diabetes_pedigree <= 0.6355 (standardized <= 0.4790):
                    if glucose <= 93.5000 (standardized <= -0.9397):
                        if pregnancies <= 11.5000 (standardized <= 2.3195):
                            return class=0 (diabetes_prob=0.000, samples=21, gini=0.000)
                        else:  # pregnancies > 11.5000
                            return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                    else:  # glucose > 93.5000
                        if glucose <= 95.5000 (standardized <= -0.8730):
                            return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                        else:  # glucose > 95.5000
                            if diabetes_pedigree <= 0.2895 (standardized <= -0.5694):
                                if blood_pressure <= 80.0000 (standardized <= 0.6408):
                                    return class=0 (diabetes_prob=0.000, samples=12, gini=0.000)
                                else:  # blood_pressure > 80.0000
                                    return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                            else:  # diabetes_pedigree > 0.2895
                                if blood_pressure <= 62.0000 (standardized <= -0.8267):
                                    return class=0 (diabetes_prob=0.000, samples=3, gini=0.000)
                                else:  # blood_pressure > 62.0000
                                    if blood_pressure <= 80.0000 (standardized <= 0.6408):
                                        if insulin <= 157.5000 (standardized <= 0.2515):
                                            return class=1 (diabetes_prob=1.000, samples=4, gini=0.000)
                                        else:  # insulin > 157.5000
                                            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                    else:  # blood_pressure > 80.0000
                                        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                else:  # diabetes_pedigree > 0.6355
                    if insulin <= 122.0000 (standardized <= -0.1996):
                        if diabetes_pedigree <= 0.9165 (standardized <= 1.3304):
                            return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                        else:  # diabetes_pedigree > 0.9165
                            return class=0 (diabetes_prob=0.000, samples=5, gini=0.000)
                    else:  # insulin > 122.0000
                        if age <= 31.5000 (standardized <= -0.1579):
                            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                        else:  # age > 31.5000
                            return class=1 (diabetes_prob=1.000, samples=8, gini=0.000)
            else:  # glucose > 107.5000
                if bmi <= 43.0000 (standardized <= 1.5475):
                    if pregnancies <= 7.5000 (standardized <= 1.1115):
                        if age <= 42.5000 (standardized <= 0.7725):
                            if pregnancies <= 1.5000 (standardized <= -0.7004):
                                if bmi <= 31.8500 (standardized <= -0.0877):
                                    if diabetes_pedigree <= 0.4090 (standardized <= -0.2073):
                                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                    else:  # diabetes_pedigree > 0.4090
                                        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                else:  # bmi > 31.8500
                                    if diabetes_pedigree <= 0.9560 (standardized <= 1.4501):
                                        return class=1 (diabetes_prob=1.000, samples=6, gini=0.000)
                                    else:  # diabetes_pedigree > 0.9560
                                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                            else:  # pregnancies > 1.5000
                                if diabetes_pedigree <= 0.8975 (standardized <= 1.2728):
                                    if glucose <= 118.5000 (standardized <= -0.1058):
                                        if glucose <= 111.5000 (standardized <= -0.3393):
                                            return class=0 (diabetes_prob=0.000, samples=5, gini=0.000)
                                        else:  # glucose > 111.5000
                                            if blood_pressure <= 76.0000 (standardized <= 0.3147):
                                                return class=1 (diabetes_prob=1.000, samples=5, gini=0.000)
                                            else:  # blood_pressure > 76.0000
                                                if age <= 37.5000 (standardized <= 0.3496):
                                                    return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                                else:  # age > 37.5000
                                                    return class=0 (diabetes_prob=0.000, samples=3, gini=0.000)
                                    else:  # glucose > 118.5000
                                        if age <= 29.5000 (standardized <= -0.3270):
                                            return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                                        else:  # age > 29.5000
                                            if insulin <= 99.5000 (standardized <= -0.4855):
                                                return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                            else:  # insulin > 99.5000
                                                return class=0 (diabetes_prob=0.000, samples=19, gini=0.000)
                                else:  # diabetes_pedigree > 0.8975
                                    return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                        else:  # age > 42.5000
                            if age <= 59.0000 (standardized <= 2.1680):
                                if bmi <= 39.9500 (standardized <= 1.1002):
                                    if diabetes_pedigree <= 0.2220 (standardized <= -0.7740):
                                        if glucose <= 139.0000 (standardized <= 0.5780):
                                            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                        else:  # glucose > 139.0000
                                            return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                                    else:  # diabetes_pedigree > 0.2220
                                        if pregnancies <= 0.5000 (standardized <= -1.0023):
                                            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                        else:  # pregnancies > 0.5000
                                            return class=1 (diabetes_prob=1.000, samples=14, gini=0.000)
                                else:  # bmi > 39.9500
                                    return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                            else:  # age > 59.0000
                                if glucose <= 143.5000 (standardized <= 0.7281):
                                    return class=0 (diabetes_prob=0.000, samples=6, gini=0.000)
                                else:  # glucose > 143.5000
                                    return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                    else:  # pregnancies > 7.5000
                        if diabetes_pedigree <= 0.5200 (standardized <= 0.1290):
                            if bmi <= 35.1000 (standardized <= 0.3889):
                                if insulin <= 107.5000 (standardized <= -0.3838):
                                    return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                                else:  # insulin > 107.5000
                                    if bmi <= 28.0500 (standardized <= -0.6450):
                                        return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
                                    else:  # bmi > 28.0500
                                        if pregnancies <= 9.5000 (standardized <= 1.7155):
                                            return class=1 (diabetes_prob=1.000, samples=7, gini=0.000)
                                        else:  # pregnancies > 9.5000
                                            if bmi <= 31.1500 (standardized <= -0.1904):
                                                return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                                            else:  # bmi > 31.1500
                                                if glucose <= 118.5000 (standardized <= -0.1058):
                                                    return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                                                else:  # glucose > 118.5000
                                                    return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                            else:  # bmi > 35.1000
                                return class=0 (diabetes_prob=0.000, samples=4, gini=0.000)
                        else:  # diabetes_pedigree > 0.5200
                            if pregnancies <= 12.5000 (standardized <= 2.6214):
                                return class=1 (diabetes_prob=1.000, samples=10, gini=0.000)
                            else:  # pregnancies > 12.5000
                                return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                else:  # bmi > 43.0000
                    if diabetes_pedigree <= 0.2325 (standardized <= -0.7421):
                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                    else:  # diabetes_pedigree > 0.2325
                        return class=1 (diabetes_prob=1.000, samples=11, gini=0.000)
else:  # glucose > 154.5000
    if bmi <= 23.1000 (standardized <= -1.3710):
        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
    else:  # bmi > 23.1000
        if insulin <= 80.0000 (standardized <= -0.7332):
            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
        else:  # insulin > 80.0000
            if insulin <= 544.0000 (standardized <= 5.1625):
                if age <= 62.5000 (standardized <= 2.4640):
                    if diabetes_pedigree <= 0.1345 (standardized <= -1.0391):
                        if insulin <= 146.5000 (standardized <= 0.1117):
                            return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                        else:  # insulin > 146.5000
                            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                    else:  # diabetes_pedigree > 0.1345
                        if blood_pressure <= 92.0000 (standardized <= 1.6192):
                            if skin_thickness <= 15.5000 (standardized <= -1.5242):
                                return class=1 (diabetes_prob=0.667, samples=3, gini=0.444)
                            else:  # skin_thickness > 15.5000
                                if age <= 57.5000 (standardized <= 2.0411):
                                    if pregnancies <= 9.5000 (standardized <= 1.7155):
                                        if skin_thickness <= 26.5000 (standardized <= -0.2862):
                                            if diabetes_pedigree <= 0.5900 (standardized <= 0.3411):
                                                return class=1 (diabetes_prob=1.000, samples=9, gini=0.000)
                                            else:  # diabetes_pedigree > 0.5900
                                                return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                        else:  # skin_thickness > 26.5000
                                            return class=1 (diabetes_prob=1.000, samples=54, gini=0.000)
                                    else:  # pregnancies > 9.5000
                                        if diabetes_pedigree <= 0.2690 (standardized <= -0.6315):
                                            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                                        else:  # diabetes_pedigree > 0.2690
                                            return class=1 (diabetes_prob=1.000, samples=4, gini=0.000)
                                else:  # age > 57.5000
                                    if bmi <= 34.2000 (standardized <= 0.2569):
                                        return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
                                    else:  # bmi > 34.2000
                                        return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
                        else:  # blood_pressure > 92.0000
                            if glucose <= 177.0000 (standardized <= 1.8456):
                                return class=1 (diabetes_prob=1.000, samples=4, gini=0.000)
                            else:  # glucose > 177.0000
                                return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                else:  # age > 62.5000
                    if blood_pressure <= 82.0000 (standardized <= 0.8039):
                        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
                    else:  # blood_pressure > 82.0000
                        return class=1 (diabetes_prob=1.000, samples=2, gini=0.000)
            else:  # insulin > 544.0000
                return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
```
