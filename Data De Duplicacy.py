# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Step 1: Get all available catalogs
catalogs = [row.catalog for row in spark.sql("SHOW CATALOGS").collect()]

# Step 2: For each catalog, get table names
table_list = []

for catalog in catalogs:
    try:
        df = spark.sql(f"""
            SELECT
                '{catalog}' AS catalog_name,
                table_schema,
                table_name
            FROM {catalog}.information_schema.tables
        """)
        table_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipping catalog {catalog} due to error: {str(e)}")

# Step 3: Union all results
if table_list:
    all_tables_df = table_list[0]
    for df in table_list[1:]:
        all_tables_df = all_tables_df.unionByName(df)

    # Step 4: Save to Delta table (change catalog.schema.table if needed)
    target_table = "information_technology_test.silver_platform_usage.duplicate_table_stage1"  # e.g. "your_catalog.your_schema.table_nam"
    all_tables_df.write.format("delta").mode("overwrite").saveAsTable(target_table)
    print(f"✅ Saved all table names to: {target_table}")
else:
    print("❌ No table data collected.")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from information_technology_test.silver_platform_usage.duplicate_table_stage1
# MAGIC where catalog_name = 'marketing_prod'

# COMMAND ----------

from pyspark.sql.functions import col, lit
from pyspark.sql import Row
from pyspark.sql.utils import AnalysisException

# Configuration
source_table = "information_technology_test.silver_platform_usage.duplicate_table_stage1"
output_table = "information_technology_test.silver_platform_usage.duplicate_table_stage2"
target_catalogs = ['connected_experience', 'customer', 'acquire']
batch_size = 100

# Step 1: Read and filter input
tables_df = spark.table(source_table).filter(col("catalog_name").isin(target_catalogs))
tables = tables_df.collect()
total_tables = len(tables)

print(f"📦 Total tables to process: {total_tables}")

# Step 2: Process in 100-table batches
for i in range(0, total_tables, batch_size):
    batch = tables[i:i + batch_size]
    results = []
    print(f"\n🔁 Processing batch {i//batch_size + 1} ({i} to {i+len(batch)-1})")

    for row in batch:
        catalog = row["catalog_name"]
        schema = row["table_schema"]
        table = row["table_name"]
        full_table = f"{catalog}.{schema}.{table}"

        try:
            # Try to read the table
            df = spark.table(full_table)

            # Extract schema details
            schema_fields = df.schema.fields
            column_names = ",".join([f.name for f in schema_fields])
            column_types = ",".join([f.dataType.simpleString() for f in schema_fields])
            schema_fingerprint = ",".join([f"{f.name}:{f.dataType.simpleString()}" for f in schema_fields])

            # Count rows
            row_count = df.count()

            # Get owner
            try:
                info_schema = spark.table(f"{catalog}.information_schema.tables").select(
                    "table_schema", "table_name", "table_owner"
                )
                owner_row = info_schema.filter(
                    (col("table_schema") == schema) & (col("table_name") == table)
                ).first()
                owner = owner_row["table_owner"] if owner_row else "unknown"
            except:
                owner = "unknown"

            results.append(Row(
                catalog_name=catalog,
                table_schema=schema,
                table_name=table,
                column_names=column_names,
                column_types=column_types,
                schema_fingerprint=schema_fingerprint,
                row_count=row_count,
                owner=owner
            ))

        except AnalysisException as e:
            print(f"⚠️ Skipped {full_table}: AnalysisException - {str(e)}")
            continue
        except Exception as e:
            print(f"⚠️ Skipped {full_table}: {str(e)}")
            continue

    # Step 3: Save the current batch
    if results:
        batch_df = spark.createDataFrame(results)
        batch_df.write.format("delta").mode("append").saveAsTable(output_table)
        print(f"✅ Batch {i//batch_size + 1} saved with {len(results)} records.")
    else:
        print(f"❌ Batch {i//batch_size + 1}: No valid tables processed.")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from  information_technology_test.silver_platform_usage.duplicate_table_stage2
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Load the metadata table
df = spark.table("information_technology_test.silver_platform_usage.duplicate_table_stage2")

# Add fully qualified table name
df = df.withColumn("full_table_name", F.concat_ws(".", "catalog_name", "table_schema", "table_name"))

# Self-join (all unique pairs)
df_a = df.alias("a")
df_b = df.alias("b")
joined = df_a.crossJoin(df_b).filter(F.col("a.full_table_name") < F.col("b.full_table_name"))

# Jaccard Similarity UDF
def jaccard_similarity(a, b):
    set_a = set(a.split(",")) if a else set()
    set_b = set(b.split(",")) if b else set()
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b)) / len(set_a | set_b)

jaccard_udf = F.udf(jaccard_similarity, DoubleType())

# Compute match percentages
results = joined \
    .withColumn("column_name_match_percent", jaccard_udf("a.column_names", "b.column_names") * 100) \
    .withColumn("column_type_match_percent", jaccard_udf("a.column_types", "b.column_types") * 100) \
    .withColumn("schema_match_percent", jaccard_udf(
        F.concat_ws(":", "a.column_names", "a.column_types"),
        F.concat_ws(":", "b.column_names", "b.column_types")
    ) * 100) \
    .withColumn("row_count_match_percent",
                (F.try_divide(F.least("a.row_count", "b.row_count"), F.greatest("a.row_count", "b.row_count")) * 100)) \
    .withColumn("owner_match_percent",
                F.when(F.col("a.owner") == F.col("b.owner"), F.lit(100.0)).otherwise(0.0)) \
    .filter(F.col("schema_match_percent") > 90)

# Select final required columns
final = results.select(
    F.col("a.full_table_name").alias("table_full_name"),
    F.col("b.full_table_name").alias("duplicate_table_full_name"),
    F.col("column_name_match_percent"),
    F.col("column_type_match_percent"),
    F.col("schema_match_percent"),
    F.col("row_count_match_percent"),
    F.col("owner_match_percent"),
    F.col("a.schema_fingerprint").alias("schema_fingerprint_table_1"),
    F.col("b.schema_fingerprint").alias("schema_fingerprint_table_2"),
    F.col("a.row_count").alias("row_count_table_1"),
    F.col("b.row_count").alias("row_count_table_2")
)

# Save result into Delta table
final.write.mode("overwrite").format("delta").saveAsTable("information_technology_test.silver_platform_usage.duplicate_table_stage3")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from information_technology_test.silver_platform_usage.duplicate_table_stage3

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import builtins
import traceback
import re

# Load duplicate pairs
pairs_df = spark.table("information_technology_test.silver_platform_usage.duplicate_table_stage3").collect()

batch_size = 100
total_batches = (len(pairs_df) + batch_size - 1) // batch_size

results = []
failures = []

def extract_timestamp_candidates(fingerprint):
    """Return list of timestamp-like columns from schema fingerprint."""
    cols = fingerprint.split('|')
    ts_candidates = []
    for col in cols:
        parts = col.split(':')
        if len(parts) == 2:
            col_name, dtype = parts
            if 'timestamp' in dtype.lower() or 'date' in dtype.lower() or re.search(r'(ts|time)', col_name.lower()):
                ts_candidates.append((col_name.strip(), dtype.strip()))
    return ts_candidates

for batch_index in range(total_batches):
    print(f"🔁 Processing batch {batch_index + 1}/{total_batches}")
    batch = pairs_df[batch_index * batch_size : (batch_index + 1) * batch_size]

    for row in batch:
        table_1 = row['table_full_name']
        table_2 = row['duplicate_table_full_name']
        row_count_1 = row['row_count_table_1']
        row_count_2 = row['row_count_table_2']
        col_match = row['column_name_match_percent']
        schema_match = row['schema_match_percent']
        fingerprint_1 = row['schema_fingerprint_table_1']
        fingerprint_2 = row['schema_fingerprint_table_2']

        try:
            df1 = spark.table(table_1)
            df2 = spark.table(table_2)
            common_cols = list(set(df1.columns) & set(df2.columns))
            if not common_cols:
                failures.append((table_1, table_2, "no_common_columns"))
                continue

            # METHOD 1: Exact match
            if col_match == 100.0 or schema_match == 100.0:
                df_union = df1.select(common_cols).unionByName(df2.select(common_cols))
                dup_row_count = df_union.groupBy(common_cols).count().filter("count > 1").count()
                base_count = builtins.min(row_count_1, row_count_2)
                dup_pct = (dup_row_count / base_count) * 100 if base_count > 0 else 0.0
                method = "method_1"

            # METHOD 2: Partial match, use timestamp sampling
            else:
                ts_1 = extract_timestamp_candidates(fingerprint_1)
                ts_2 = extract_timestamp_candidates(fingerprint_2)

                ts1_chosen, ts2_chosen = None, None

                # choose column with latest timestamp in each table
                for col_name, dtype in ts_1:
                    if col_name in df1.columns:
                        ts1_chosen = col_name
                        break
                for col_name, dtype in ts_2:
                    if col_name in df2.columns:
                        ts2_chosen = col_name
                        break

                if not ts1_chosen or not ts2_chosen:
                    failures.append((table_1, table_2, "no_timestamp_column"))
                    continue

                df1_sample = df1.orderBy(F.col(ts1_chosen).desc_nulls_last()).limit(5000)
                df2_sample = df2.orderBy(F.col(ts2_chosen).desc_nulls_last()).limit(5000)
                sample_union = df1_sample.select(common_cols).unionByName(df2_sample.select(common_cols))
                dup_row_count = sample_union.groupBy(common_cols).count().filter("count > 1").count()
                base_count = builtins.min(df1_sample.count(), df2_sample.count())
                dup_pct = (dup_row_count / base_count) * 100 if base_count > 0 else 0.0
                method = "method_2"

            results.append((
                table_1, table_2, row_count_1, row_count_2, dup_row_count,
                builtins.round(dup_pct, 2), ",".join(common_cols), method
            ))

        except Exception as e:
            print(f"❌ Failed for {table_1} vs {table_2}")
            traceback.print_exc()
            failures.append((table_1, table_2, "unexpected_error"))
            continue

    # Save after each batch
    if results:
        result_schema = StructType([
            StructField("table_1", StringType(), True),
            StructField("table_2", StringType(), True),
            StructField("row_count_table_1", LongType(), True),
            StructField("row_count_table_2", LongType(), True),
            StructField("num_duplicate_rows", LongType(), True),
            StructField("duplicate_row_percentage", DoubleType(), True),
            StructField("common_columns", StringType(), True),
            StructField("method_used", StringType(), True)
        ])
        spark.createDataFrame(results, schema=result_schema).write.mode("append").format("delta").saveAsTable(
            "information_technology_test.silver_platform_usage.duplicate_table_stage4"
        )
        results.clear()

    if failures:
        fail_schema = StructType([
            StructField("table_1", StringType(), True),
            StructField("table_2", StringType(), True),
            StructField("failure_reason", StringType(), True)
        ])
        spark.createDataFrame(failures, schema=fail_schema).write.mode("append").format("delta").saveAsTable(
            "information_technology_test.silver_platform_usage.duplicate_table_stage4_failed"
        )
        failures.clear()

print("✅ Stage-4 processing complete.")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from information_technology_test.silver_platform_usage.duplicate_table_stage4

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import Row

# Step 1: Load Stage-4 and filter rows with >90% match
stage4_df = spark.table("information_technology_test.silver_platform_usage.duplicate_table_stage4")

high_match_tables = (
    stage4_df
    .filter(col("duplicate_row_percentage") > 90)
    .select(col("table_2").alias("full_table_name"))
    .distinct()
)

# Step 2: Use DESCRIBE DETAIL to get size info
results = []

for row in high_match_tables.collect():
    table_name = row["full_table_name"]
    try:
        desc_df = spark.sql(f"DESCRIBE DETAIL {table_name}")
        detail = desc_df.collect()[0]
        size_bytes = detail['sizeInBytes']
        size_gb = round(size_bytes / (1024 ** 3), 2)
        cost_per_gb = 0.023  # Change per your cloud pricing
        monthly_cost = round(size_gb * cost_per_gb, 4)

        results.append(Row(
            full_table_name=table_name,
            size_bytes=size_bytes,
            size_gb=size_gb,
            estimated_monthly_cost_usd=monthly_cost
        ))
    except Exception:
        results.append(Row(
            full_table_name=table_name,
            size_bytes=None,
            size_gb=None,
            estimated_monthly_cost_usd=None
        ))

# Step 3: Create final dataframe and save as Delta/Unity table
table2_highmatch_storage = spark.createDataFrame(results)

# Save table for Power BI dashboard consumption
table2_highmatch_storage.write.mode("overwrite").saveAsTable("information_technology_test.silver_platform_usage.table2_highmatch_storage_cost")


# COMMAND ----------

# MAGIC %sql
# MAGIC drop table information_technology_test.silver_platform_usage.table2_storage_cost