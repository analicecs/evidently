from typing import List
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from aif360.sklearn.metrics import class_imbalance  # type: ignore
from aif360.sklearn.metrics import conditional_demographic_disparity  # type: ignore
from aif360.sklearn.metrics import kl_divergence  # type: ignore
from plotly.express import line

from evidently import Dataset
from evidently.core.metric_types import BoundTest
from evidently.core.metric_types import SingleValue
from evidently.core.metric_types import SingleValueCalculation
from evidently.core.metric_types import SingleValueMetric
from evidently.core.report import Context
from evidently.legacy.renderers.html_widgets import plotly_figure
from evidently.tests import Reference
from evidently.tests import eq


class MyMaxMetric(SingleValueMetric):
    column: str

    def _default_tests(self) -> List[BoundTest]:
        return [eq(0).bind_single(self.get_fingerprint())]

    def _default_tests_with_reference(self) -> List[BoundTest]:
        return [eq(Reference(relative=0.1)).bind_single(self.get_fingerprint())]


# implementation
class MaxMetricImplementation(SingleValueCalculation[MyMaxMetric]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Optional[Dataset]) -> SingleValue:
        x = current_data.column(self.metric.column).data
        value = x.max()
        result = self.result(value=value)
        figure = line(x)
        figure.add_hrect(6, 10)
        result.widget = [plotly_figure(title=self.display_name(), figure=figure)]  # skip this to get a simple counter
        return result

    def display_name(self) -> str:
        return f"Max value for {self.metric.column}"


class ClassImbalanceMetric(SingleValueMetric):
    target_column: str
    protected_attribute_column: str
    privileged_group: str

    def _default_tests(self) -> List[BoundTest]:
        return [eq(0).bind_single(self.get_fingerprint())]

    def _default_tests_with_reference(self) -> List[BoundTest]:
        return [eq(Reference(relative=0.1)).bind_single(self.get_fingerprint())]


class ClassImbalanceImplementation(SingleValueCalculation[ClassImbalanceMetric]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Optional[Dataset]) -> SingleValue:
        y_true = current_data.column(self.metric.target_column).data
        prot_attr = current_data.column(self.metric.protected_attribute_column).data
        prot_attr = prot_attr.astype(str)

        value = class_imbalance(
            y_true=y_true,
            prot_attr=prot_attr,
            priv_group=self.metric.privileged_group,  # pyright: ignore[reportArgumentType]
        )

        result = self.result(value=value)

        # Criar DataFrame manualmente a partir dos dados
        df_current = pd.DataFrame(
            {
                self.metric.target_column: y_true,
                self.metric.protected_attribute_column: prot_attr,
            }
        )

        # Calcular as proporções por grupo
        imbalance_data = df_current.groupby(self.metric.protected_attribute_column)[self.metric.target_column].mean()

        figure = go.Figure()

        # Adicionar barras para cada grupo
        for group in imbalance_data.index:
            is_privileged = group == self.metric.privileged_group
            color = "blue" if is_privileged else "red"
            figure.add_trace(
                go.Bar(
                    x=[group],
                    y=[imbalance_data[group]],
                    name=group,
                    marker_color=color,
                    text=[f"{imbalance_data[group]:.3f}"],
                    textposition="auto",
                )
            )

        figure.update_layout(
            title=f"Class Imbalance - Approval Rate by Group<br>Ratio: {value:.4f}",
            xaxis_title="Protected Group",
            yaxis_title="Approval Rate",
            showlegend=False,
            yaxis=dict(range=[0, 1]),
        )

        result.widget = [plotly_figure(title=self.display_name(value), figure=figure)]
        return result

    def display_name(self, value: float) -> str:
        return f"Class Imbalance ({self.metric.target_column}) - Value {value:.4f}"


class KLDivergenceMetric(SingleValueMetric):
    target_column: str
    protected_attribute_column: str
    privileged_group: str

    def _default_tests(self) -> List[BoundTest]:
        return [eq(0).bind_single(self.get_fingerprint())]

    def _default_tests_with_reference(self) -> List[BoundTest]:
        return [eq(Reference(relative=0.1)).bind_single(self.get_fingerprint())]


class KLDivergenceImplementation(SingleValueCalculation[KLDivergenceMetric]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Optional[Dataset]) -> SingleValue:
        y_true = current_data.column(self.metric.target_column).data
        prot_attr = current_data.column(self.metric.protected_attribute_column).data
        prot_attr = prot_attr.astype(str)

        value = kl_divergence(
            y_true=y_true,
            prot_attr=prot_attr,
            priv_group=self.metric.privileged_group,  # pyright: ignore[reportArgumentType]
        )

        result = self.result(value=value)

        df_current = pd.DataFrame(
            {
                self.metric.target_column: y_true,
                self.metric.protected_attribute_column: prot_attr,
            }
        )

        imbalance_data = df_current.groupby(self.metric.protected_attribute_column)[self.metric.target_column].mean()

        figure = go.Figure()

        for group in imbalance_data.index:
            is_privileged = group == self.metric.privileged_group
            color = "blue" if is_privileged else "red"
            figure.add_trace(
                go.Bar(
                    x=[group],
                    y=[imbalance_data[group]],
                    name=group,
                    marker_color=color,
                    text=[f"{imbalance_data[group]:.3f}"],
                    textposition="auto",
                )
            )

        figure.update_layout(
            title=f"KL divergence - Approval Rate by Group<br>Ratio: {value:.4f}",
            xaxis_title="Protected Group",
            yaxis_title="Approval Rate",
            showlegend=False,
            yaxis=dict(range=[0, 1]),
        )

        result.widget = [plotly_figure(title=self.display_name(), figure=figure)]
        return result

    def display_name(self) -> str:
        return f"KL Divergence ({self.metric.target_column})"


class ConditionalDemographicDisparityMetric(SingleValueMetric):
    target_column: str
    protected_attribute_column: str
    category_column: str  # coluna categórica que condiciona a disparidade

    def _default_tests(self) -> List[BoundTest]:
        return [eq(0).bind_single(self.get_fingerprint())]

    def _default_tests_with_reference(self) -> List[BoundTest]:
        return [eq(Reference(relative=0.1)).bind_single(self.get_fingerprint())]


class ConditionalDemographicDisparityImplementation(SingleValueCalculation[ConditionalDemographicDisparityMetric]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Optional[Dataset]) -> SingleValue:
        y_true = current_data.column(self.metric.target_column).data
        prot_attr = current_data.column(self.metric.protected_attribute_column).data
        prot_attr = prot_attr.astype(str)
        categories = current_data.column(self.metric.category_column).data

        # Calcular Conditional Demographic Disparity usando AIF360
        value = conditional_demographic_disparity(y_true=y_true, prot_attr=prot_attr)

        result = self.result(value=value)

        # Criar DataFrame para calcular proporções por categoria e grupo
        df_current = pd.DataFrame({"categoria": categories, "target": y_true, "grupo": prot_attr})

        grouped = df_current.groupby(["categoria", "grupo"])["target"].mean().reset_index()

        # Gráfico de barras agrupadas
        figure = go.Figure()
        for group in grouped["grupo"].unique():
            group_data = grouped[grouped["grupo"] == group]
            color = "blue" if group == grouped["grupo"].iloc[0] else "red"
            figure.add_trace(
                go.Bar(
                    x=group_data["categoria"],
                    y=group_data["target"],
                    name=str(group),
                    marker_color=color,
                    text=[f"{v:.3f}" for v in group_data["target"]],
                    textposition="auto",
                )
            )

        figure.update_layout(
            title=f"Conditional Demographic Disparity (CDD) - {self.metric.target_column}<br>Value: {value:.4f}",
            xaxis_title="Categoria",
            yaxis_title="Proporção target=1",
            barmode="group",
            yaxis=dict(range=[0, 1]),
        )

        result.widget = [plotly_figure(title=self.display_name(), figure=figure)]
        return result

    def display_name(self) -> str:
        return f"Conditional Demographic Disparity ({self.metric.target_column})"
