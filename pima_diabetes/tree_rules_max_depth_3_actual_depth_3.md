# Pima Diabetes Decision Tree Rules

- max_depth setting: `3`
- actual tree depth: `3`

```python
if glucose <= 154.5000 (standardized <= 1.0951):
    if bmi <= 27.3500 (standardized <= -0.7477):
        if pregnancies <= 12.5000 (standardized <= 2.6214):
            return class=0 (diabetes_prob=0.032, samples=126, gini=0.061)
        else:  # pregnancies > 12.5000
            return class=0 (diabetes_prob=0.500, samples=2, gini=0.500)
    else:  # bmi > 27.3500
        if age <= 28.5000 (standardized <= -0.4116):
            return class=0 (diabetes_prob=0.189, samples=185, gini=0.307)
        else:  # age > 28.5000
            return class=0 (diabetes_prob=0.450, samples=202, gini=0.495)
else:  # glucose > 154.5000
    if bmi <= 23.1000 (standardized <= -1.3710):
        return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
    else:  # bmi > 23.1000
        if insulin <= 80.0000 (standardized <= -0.7332):
            return class=0 (diabetes_prob=0.000, samples=2, gini=0.000)
        else:  # insulin > 80.0000
            return class=1 (diabetes_prob=0.874, samples=95, gini=0.221)
```
