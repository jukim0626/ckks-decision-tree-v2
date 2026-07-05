# Pima Diabetes Decision Tree Rules

- max_depth setting: `5`
- actual tree depth: `5`

```python
if glucose <= 154.5000 (standardized <= 1.0951):
    if bmi <= 27.3500 (standardized <= -0.7477):
        if pregnancies <= 12.5000 (standardized <= 2.6214):
            if glucose <= 133.5000 (standardized <= 0.3946):
                if bmi <= 26.9000 (standardized <= -0.8137):
                    return class=0 (diabetes_prob=0.010, samples=105, gini=0.019)
                else:  # bmi > 26.9000
                    return class=0 (diabetes_prob=0.143, samples=7, gini=0.245)
            else:  # glucose > 133.5000
                if pregnancies <= 2.5000 (standardized <= -0.3984):
                    return class=0 (diabetes_prob=0.000, samples=8, gini=0.000)
                else:  # pregnancies > 2.5000
                    return class=0 (diabetes_prob=0.333, samples=6, gini=0.444)
        else:  # pregnancies > 12.5000
            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
    else:  # bmi > 27.3500
        if age <= 28.5000 (standardized <= -0.4116):
            if glucose <= 127.5000 (standardized <= 0.1944):
                if bmi <= 45.4000 (standardized <= 1.8995):
                    return class=0 (diabetes_prob=0.099, samples=141, gini=0.179)
                else:  # bmi > 45.4000
                    return class=1 (diabetes_prob=0.750, samples=4, gini=0.375)
            else:  # glucose > 127.5000
                if blood_pressure <= 73.0000 (standardized <= 0.0701):
                    return class=1 (diabetes_prob=0.636, samples=22, gini=0.463)
                else:  # blood_pressure > 73.0000
                    return class=0 (diabetes_prob=0.222, samples=18, gini=0.346)
        else:  # age > 28.5000
            if glucose <= 107.5000 (standardized <= -0.4727):
                if diabetes_pedigree <= 0.6355 (standardized <= 0.4790):
                    return class=0 (diabetes_prob=0.151, samples=53, gini=0.256)
                else:  # diabetes_pedigree > 0.6355
                    return class=1 (diabetes_prob=0.556, samples=18, gini=0.494)
            else:  # glucose > 107.5000
                if bmi <= 43.0000 (standardized <= 1.5475):
                    return class=1 (diabetes_prob=0.517, samples=118, gini=0.499)
                else:  # bmi > 43.0000
                    return class=1 (diabetes_prob=0.923, samples=13, gini=0.142)
else:  # glucose > 154.5000
    if bmi <= 23.1000 (standardized <= -1.3710):
        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
    else:  # bmi > 23.1000
        if insulin <= 80.0000 (standardized <= -0.7332):
            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
        else:  # insulin > 80.0000
            if insulin <= 544.0000 (standardized <= 5.1625):
                if diabetes_pedigree <= 0.1345 (standardized <= -1.0391):
                    return class=0 (diabetes_prob=0.500, samples=4, gini=0.500)
                else:  # diabetes_pedigree > 0.1345
                    return class=1 (diabetes_prob=0.909, samples=88, gini=0.165)
            else:  # insulin > 544.0000
                return class=0 (diabetes_prob=0.333, samples=3, gini=0.444)
```
