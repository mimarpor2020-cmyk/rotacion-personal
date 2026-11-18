"""Definición del formulario: etiquetas en español, grupos, opciones y ejemplos."""

ESCALA_SATISF = [(1, "Baja (1)"), (2, "Media (2)"), (3, "Alta (3)"), (4, "Muy alta (4)")]

GRUPOS = [
    ("personal", "Datos personales"),
    ("puesto", "Puesto de trabajo"),
    ("compensacion", "Compensación"),
    ("satisfaccion", "Satisfacción"),
    ("trayectoria", "Trayectoria"),
]

# tipo "sel" = lista desplegable (valor que espera el modelo, texto que ve el usuario)
# tipo "num" = número entero (los rangos se leen de config_modelo.json)
CAMPOS = [
    # --- Datos personales ---
    dict(nombre="Age", etiqueta="Edad", grupo="personal", tipo="num", sufijo="años"),
    dict(nombre="Gender", etiqueta="Género", grupo="personal", tipo="sel",
         opciones=[("Female", "Femenino"), ("Male", "Masculino")]),
    dict(nombre="MaritalStatus", etiqueta="Estado civil", grupo="personal", tipo="sel",
         opciones=[("Single", "Soltero/a"), ("Married", "Casado/a"), ("Divorced", "Divorciado/a")]),
    dict(nombre="DistanceFromHome", etiqueta="Distancia casa–trabajo", grupo="personal", tipo="num", sufijo="km"),
    dict(nombre="Education", etiqueta="Nivel educativo", grupo="personal", tipo="sel",
         opciones=[(1, "Secundaria (1)"), (2, "Técnico (2)"), (3, "Pregrado (3)"), (4, "Maestría (4)"), (5, "Doctorado (5)")]),
    dict(nombre="EducationField", etiqueta="Área de formación", grupo="personal", tipo="sel",
         opciones=[("Life Sciences", "Ciencias de la vida"), ("Medical", "Medicina"), ("Marketing", "Mercadeo"),
                   ("Technical Degree", "Título técnico"), ("Human Resources", "Recursos humanos"), ("Other", "Otra")]),
    # --- Puesto ---
    dict(nombre="Department", etiqueta="Departamento", grupo="puesto", tipo="sel",
         opciones=[("Research & Development", "Investigación y desarrollo"), ("Sales", "Ventas"),
                   ("Human Resources", "Recursos humanos")]),
    dict(nombre="JobRole", etiqueta="Cargo", grupo="puesto", tipo="sel",
         opciones=[("Sales Executive", "Ejecutivo de ventas"), ("Research Scientist", "Científico investigador"),
                   ("Laboratory Technician", "Técnico de laboratorio"), ("Manufacturing Director", "Director de manufactura"),
                   ("Healthcare Representative", "Representante de salud"), ("Manager", "Gerente"),
                   ("Sales Representative", "Representante de ventas"), ("Research Director", "Director de investigación"),
                   ("Human Resources", "Recursos humanos")]),
    dict(nombre="JobLevel", etiqueta="Nivel del cargo", grupo="puesto", tipo="sel",
         opciones=[(1, "Inicial (1)"), (2, "Intermedio (2)"), (3, "Senior (3)"), (4, "Gerencial (4)"), (5, "Directivo (5)")]),
    dict(nombre="OverTime", etiqueta="Hace horas extra", grupo="puesto", tipo="sel",
         opciones=[("No", "No"), ("Yes", "Sí")]),
    dict(nombre="BusinessTravel", etiqueta="Viajes de trabajo", grupo="puesto", tipo="sel",
         opciones=[("Non-Travel", "No viaja"), ("Travel_Rarely", "Rara vez"), ("Travel_Frequently", "Con frecuencia")]),
    dict(nombre="JobInvolvement", etiqueta="Compromiso con el trabajo", grupo="puesto", tipo="sel", opciones=ESCALA_SATISF),
    dict(nombre="PerformanceRating", etiqueta="Evaluación de desempeño", grupo="puesto", tipo="sel",
         opciones=[(3, "Excelente (3)"), (4, "Sobresaliente (4)")]),  # el dataset solo contiene 3 y 4
    # --- Compensación ---
    dict(nombre="MonthlyIncome", etiqueta="Salario mensual", grupo="compensacion", tipo="num", sufijo="USD"),
    dict(nombre="PercentSalaryHike", etiqueta="Último aumento salarial", grupo="compensacion", tipo="num", sufijo="%"),
    dict(nombre="StockOptionLevel", etiqueta="Nivel de acciones", grupo="compensacion", tipo="sel",
         opciones=[(0, "Ninguna (0)"), (1, "Bajo (1)"), (2, "Medio (2)"), (3, "Alto (3)")]),
    dict(nombre="DailyRate", etiqueta="Tarifa diaria", grupo="compensacion", tipo="num", sufijo="USD"),
    dict(nombre="HourlyRate", etiqueta="Tarifa por hora", grupo="compensacion", tipo="num", sufijo="USD"),
    dict(nombre="MonthlyRate", etiqueta="Tarifa mensual", grupo="compensacion", tipo="num", sufijo="USD"),
    # --- Satisfacción ---
    dict(nombre="JobSatisfaction", etiqueta="Con el trabajo", grupo="satisfaccion", tipo="sel", opciones=ESCALA_SATISF),
    dict(nombre="EnvironmentSatisfaction", etiqueta="Con el ambiente", grupo="satisfaccion", tipo="sel", opciones=ESCALA_SATISF),
    dict(nombre="RelationshipSatisfaction", etiqueta="Con los compañeros", grupo="satisfaccion", tipo="sel", opciones=ESCALA_SATISF),
    dict(nombre="WorkLifeBalance", etiqueta="Equilibrio vida–trabajo", grupo="satisfaccion", tipo="sel",
         opciones=[(1, "Malo (1)"), (2, "Aceptable (2)"), (3, "Bueno (3)"), (4, "Excelente (4)")]),
    # --- Trayectoria ---
    dict(nombre="TotalWorkingYears", etiqueta="Años de experiencia total", grupo="trayectoria", tipo="num", sufijo="años"),
    dict(nombre="NumCompaniesWorked", etiqueta="Empresas anteriores", grupo="trayectoria", tipo="num", sufijo=""),
    dict(nombre="YearsAtCompany", etiqueta="Años en esta empresa", grupo="trayectoria", tipo="num", sufijo="años"),
    dict(nombre="YearsInCurrentRole", etiqueta="Años en el cargo actual", grupo="trayectoria", tipo="num", sufijo="años"),
    dict(nombre="YearsSinceLastPromotion", etiqueta="Años desde el último ascenso", grupo="trayectoria", tipo="num", sufijo="años"),
    dict(nombre="YearsWithCurrManager", etiqueta="Años con el jefe actual", grupo="trayectoria", tipo="num", sufijo="años"),
    dict(nombre="TrainingTimesLastYear", etiqueta="Capacitaciones el último año", grupo="trayectoria", tipo="num", sufijo=""),
]

ETIQUETAS = {c["nombre"]: c["etiqueta"] for c in CAMPOS}

# Perfiles de ejemplo (los mismos de la sección 14.2 del notebook)
EJEMPLOS = {
    "A": ("Joven con horas extra y salario bajo", dict(
        Age=23, OverTime="Yes", MonthlyIncome=2100, JobLevel=1, MaritalStatus="Single",
        BusinessTravel="Travel_Frequently", Department="Sales", JobRole="Sales Representative",
        StockOptionLevel=0, TotalWorkingYears=2, YearsAtCompany=1, YearsInCurrentRole=0,
        YearsWithCurrManager=0, YearsSinceLastPromotion=0, NumCompaniesWorked=3, DistanceFromHome=22,
        JobSatisfaction=1, EnvironmentSatisfaction=1, WorkLifeBalance=1, JobInvolvement=2)),
    "B": ("Gerente estable", dict(
        Age=45, OverTime="No", MonthlyIncome=16000, JobLevel=4, MaritalStatus="Married",
        BusinessTravel="Travel_Rarely", Department="Research & Development", JobRole="Manager",
        StockOptionLevel=1, TotalWorkingYears=22, YearsAtCompany=15, YearsInCurrentRole=8,
        YearsWithCurrManager=7, YearsSinceLastPromotion=3, NumCompaniesWorked=2, DistanceFromHome=3,
        JobSatisfaction=4, EnvironmentSatisfaction=3, WorkLifeBalance=3, JobInvolvement=3)),
    "C": ("Técnico de laboratorio con horas extra", dict(
        Age=29, OverTime="Yes", MonthlyIncome=2800, JobLevel=1, MaritalStatus="Single",
        BusinessTravel="Travel_Rarely", Department="Research & Development", JobRole="Laboratory Technician",
        StockOptionLevel=0, TotalWorkingYears=5, YearsAtCompany=3, YearsInCurrentRole=2,
        YearsWithCurrManager=2, YearsSinceLastPromotion=1, NumCompaniesWorked=2, DistanceFromHome=10,
        JobSatisfaction=2, EnvironmentSatisfaction=2, WorkLifeBalance=3, JobInvolvement=3)),
    "D": ("Ejecutivo de ventas a mitad de carrera", dict(
        Age=38, OverTime="No", MonthlyIncome=7200, JobLevel=2, MaritalStatus="Married",
        BusinessTravel="Travel_Rarely", Department="Sales", JobRole="Sales Executive",
        StockOptionLevel=1, TotalWorkingYears=12, YearsAtCompany=7, YearsInCurrentRole=4,
        YearsWithCurrManager=4, YearsSinceLastPromotion=1, NumCompaniesWorked=2, DistanceFromHome=7,
        JobSatisfaction=3, EnvironmentSatisfaction=3, WorkLifeBalance=3, JobInvolvement=3)),
}
