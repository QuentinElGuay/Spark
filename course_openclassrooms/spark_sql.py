from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local") \
                    .appName("Agents") \
                    .getOrCreate()
agents = spark.read.json('Spark/data/agents.json')

# Describe
print('Loading DataFrame <agents.json>:', agents.describe())

# Count
print(f"There are {agents.count()} agents in total.")

# Filter
french_agents = agents.filter(agents.country_name == "France")
print(f"There are {french_agents.count()} French agents.")

female_fr_agents = agents.filter((agents.country_name == "France") & (agents.sex == "Female"))
print(f"There are {female_fr_agents.count()} French female agents.")

male_fr_agents = agents.filter(agents.country_name == "France").filter(agents.sex == "Male")
print(f"There are {male_fr_agents.count()} French male agents.")

# Order By and Limit
print("Result of a query using the ORM:")
female_fr_agents.orderBy('id').limit(5).show()

# Create temporary view
print("Result of the same query using a temporary view and SQL:")
agents.createTempView('agents_table')
spark.sql(
    "SELECT * FROM agents_table \
    WHERE country_name = 'France' AND sex = 'Female' \
    ORDER BY id \
    LIMIT 5"
).show()
