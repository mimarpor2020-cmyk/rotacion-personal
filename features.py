"""Ingeniería de características.

DEBE ser idéntica a la función `ingenieria_caracteristicas` del notebook:
el preprocesador guardado espera exactamente estas columnas derivadas.
"""
import numpy as np


def ingenieria_caracteristicas(d):
    d = d.copy()
    d["IngresoPorNivel"] = d["MonthlyIncome"] / d["JobLevel"]
    d["LogIngresoMensual"] = np.log1p(d["MonthlyIncome"])
    d["AniosPorEmpresa"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1)
    d["ProporcionAntiguedad"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["ProporcionSinAscenso"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["SatisfaccionTotal"] = (d["EnvironmentSatisfaction"] + d["JobSatisfaction"]
                              + d["RelationshipSatisfaction"] + d["WorkLifeBalance"]
                              + d["JobInvolvement"])
    d["MenorDe30"] = (d["Age"] < 30).astype(int)
    return d
