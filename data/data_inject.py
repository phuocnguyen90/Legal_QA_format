import utils.file_handler as file_handler
import pandas as pd

df=file_handler.create_documents_dataframe("data\preprocessed\raw_law")


df.to_excel("law_list.xlsx", index=False)