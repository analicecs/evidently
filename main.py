import pandas as pd

from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import ColumnCount
from evidently.presets import DataSummaryPreset
from src.evidently.metrics.classification_bias_calculation import ClassImbalanceMetric
from src.evidently.metrics.classification_bias_calculation import ConditionalDemographicDisparityMetric
from src.evidently.metrics.classification_bias_calculation import KLDivergenceMetric

# 1. Carregar dataset
df = pd.read_csv("pcpe_03.csv", sep=";", low_memory=False)

df = df_binario = df[df["RAMO_ATIVIDADE_1"].isin([1, 3])].copy()

# 2. Criar Evidently Dataset
df = Dataset.from_pandas(df, data_definition=DataDefinition())


# 3. Criar o report usando as métricas
report = Report(
    [
        ClassImbalanceMetric(
            target_column="I-d",
            protected_attribute_column="RAMO_ATIVIDADE_1",
            privileged_group=3,
        ),
        KLDivergenceMetric(
            target_column="I-d",
            protected_attribute_column="RAMO_ATIVIDADE_1",
            privileged_group=3,
        ),
        ConditionalDemographicDisparityMetric(
            target_column="I-d",
            protected_attribute_column="RAMO_ATIVIDADE_1",
            # --- CORREÇÃO ---
            # A coluna de categoria (estrato) não deve ser o outcome ('I-d').
            # Vamos usar 'RAMO_ATIVIDADE_1' como a categoria.
            category_column="RAMO_ATIVIDADE_1",
        ),
        DataSummaryPreset(),
        ColumnCount(),
    ]
)

# 4. Rodar e salvar o relatório
my_eval = report.run(df, None)
my_eval.save_html("aif360_class_imbalance_report.html")
print("Report salvo em 'aif360_class_imbalance_report.html'!")
