# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Testing Out Webscraping for GenAI
# MAGIC
# MAGIC Some stub code for getting playwright up

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y \
# MAGIC     libwoff-dev \
# MAGIC     libgstreamer1.0-dev \
# MAGIC     gstreamer1.0-gl \
# MAGIC     libwebp-dev \
# MAGIC     libgstreamer-plugins-bad1.0-0

# COMMAND ----------

# MAGIC %pip install pytest-playwright

# COMMAND ----------

# MAGIC %sh
# MAGIC playwright install