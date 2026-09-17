import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

BRAZIL_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

# Columns that must always be present -- their absence means the request itself is malformed
STRICTLY_REQUIRED_COLUMNS = [
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "customer_state",
]

# Columns known to have some real-world sparsity (matches notebook 5's missing-value summary) --
# your SimpleImputer already handles these, so we tolerate up to 2% missing here
SPARSE_NUMERIC_COLUMNS = [
    "order_approved_at", "state", "latitude", "longitude",
    "item_count", "unique_products", "unique_sellers",
    "total_item_price", "mean_item_price", "max_item_price",
    "total_freight_value", "mean_freight_value", "mean_product_weight_g",
    "product_category_count", "seller_state_count",
    "payment_count", "total_payment_value", "mean_payment_value", "max_payment_installments",
]

NON_NEGATIVE_COLUMNS = [
    "item_count", "unique_products", "unique_sellers",
    "total_item_price", "mean_item_price", "max_item_price",
    "total_freight_value", "mean_freight_value", "mean_product_weight_g",
    "product_category_count", "seller_state_count",
    "payment_count", "total_payment_value", "mean_payment_value", "max_payment_installments",
]


def build_order_suite():
    suite = gx.ExpectationSuite(name="raw_order_suite")

    for col in STRICTLY_REQUIRED_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    for col in SPARSE_NUMERIC_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=0.98))

    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="customer_state", value_set=BRAZIL_STATES, mostly=0.98))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="state", value_set=BRAZIL_STATES, mostly=0.98))

    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="latitude", min_value=-34, max_value=10, mostly=0.98))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="longitude", min_value=-74, max_value=-32, mostly=0.98))

    for col in NON_NEGATIVE_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=0, mostly=0.98))

    return suite


def validate_orders(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("orders")
    asset = source.add_dataframe_asset(name="orders")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    suite = build_order_suite()
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    result = batch.validate(suite)

    return result
