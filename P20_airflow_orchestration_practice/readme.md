This folder contains my practice and notes on using Apache Airflow. There are three parts:
- some commonly used Airflow CLI commands
- A use case to pipeline several small ETL functions on a local executor that read data from a csv, clean them in a py file, and write different versions into SQL tables; some features are: conditional trigger (sensor), retries, Xcoms, and concurrency settings
- A customized plugin operator(i.e. nodes on an ETL pipeline) that can be replaced by any ELT or ML functions when needed